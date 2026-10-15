"""
PPO training of the racing policy on the batched RaceEnv.

    python3 train.py --timesteps 5_000_000 --n-envs 512 --seed 0 --name run1 [--resume runs/run0]

Writes rl/runs/<name>/ (override the parent with --out-dir):
    model.zip, vecnormalize.pkl   final policy + observation statistics
    config.json                   every env / reward / PPO / CLI setting
    checkpoints/                  model + vecnormalize every --checkpoint-every steps
    tb/                           tensorboard logs (tensorboard --logdir runs/<name>/tb)
Evaluate with  python3 fly_course.py --track tracks/oval6.json --controller rl --model runs/<name>

CONVENTIONS: as race_env.py (ENU, FLU, WXYZ, CTBR command).

CHOICES (and why)
    Activation tanh: the SB3/PPO default for continuous control; bounded
        hidden units behave well with VecNormalize and a Gaussian policy head.
        ELU is available with --activation elu (smoother gradients, unbounded).
    log_std_init -1.0: SB3's default std of 1.0 is as wide as the whole action
        range [-1,1] (thrust +-1 g, rates +-full), which crashes almost every
        early rollout; std ~0.37 explores around hover instead.
    Fixed tracks are NOT in the training pool by default (RaceEnvConfig), so the
        periodic evaluation below measures generalisation to unseen tracks.
    Rollout = n_envs * n_steps samples, minibatch = rollout / 4, so each rollout
        gives 4 * n_epochs gradient steps.
    --resume takes a run directory (or its model.zip); --timesteps then counts
        ADDITIONAL steps and the step counter continues.
"""

import argparse
import dataclasses
import json
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecNormalize

from course import Course, TRACKS_DIR
from drone_params import QuadParams
from fly_course import run_controller
from race_env import RaceEnvConfig, RaceVecEnv
from rl_policy import RlPolicyController

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
TRACK_NAMES = ["straight3", "oval6", "slalom5"]


class EvalTracksCallback(BaseCallback):
    """Every `every` timesteps fly the current policy deterministically on the
    fixed tracks from Course.start_pose() and log completion / lap time."""

    def __init__(self, every: int, params: QuadParams, max_time: float = 40.0):
        super().__init__()
        self.every, self.params, self.max_time = every, params, max_time
        self.next_eval = None    # set on the first step: after --resume, num_timesteps
        # does not start at 0, and starting the schedule at `every` made the
        # callback fire on every rollout until it caught up
        self.courses = [Course.load(os.path.join(TRACKS_DIR, f"{n}.json")) for n in TRACK_NAMES]

    def _on_step(self) -> bool:
        if self.next_eval is None:
            self.next_eval = self.num_timesteps + self.every
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.every
            self.evaluate()
        return True

    def evaluate(self):
        done = []
        for course in self.courses:
            ctl = RlPolicyController(self.params, course, policy=self.model.policy,
                                     vecnorm=self.model.get_vec_normalize_env())
            _, s = run_controller(course, ctl, self.params, max_time=self.max_time)
            self.logger.record(f"eval/{course.name}/completed", float(s["completed"]))
            self.logger.record(f"eval/{course.name}/gates_passed_frac",
                               s["gates_passed"] / s["n_gates"])
            if s["completed"]:
                self.logger.record(f"eval/{course.name}/lap_time", s["lap_time"])
            done.append(s["completed"])
            print(f"  [eval @ {self.num_timesteps}] {course.name}: completed={s['completed']} "
                  f"gates={s['gates_passed']}/{s['n_gates']} lap={s['lap_time']:.2f}s", flush=True)
        self.logger.record("eval/completed_frac", float(np.mean(done)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=lambda s: int(s.replace("_", "")), default=5_000_000)
    ap.add_argument("--n-envs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--resume", default=None, help="run dir (or model.zip) to continue from")
    ap.add_argument("--out-dir", default=RUNS_DIR)
    ap.add_argument("--n-steps", type=int, default=64)
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w-crash", type=float, default=RaceEnvConfig.w_crash,
                    help="crash penalty (reward weight)")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--activation", choices=["tanh", "elu"], default="tanh")
    ap.add_argument("--log-std-init", type=float, default=-1.0)
    ap.add_argument("--eval-every", type=int, default=250_000)
    ap.add_argument("--eval-max-time", type=float, default=40.0)
    ap.add_argument("--checkpoint-every", type=int, default=500_000)
    ap.add_argument("--n-courses", type=int, default=256)
    ap.add_argument("--course-seed", type=int, default=0)
    ap.add_argument("--yaw-jitter", type=float, default=0.0,
                    help="+- deg of random gate yaw off the travel direction in training courses")
    ap.add_argument("--train-on-fixed", action="store_true",
                    help="add the fixed tracks to the training pool (then eval is not held-out)")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    run_dir = os.path.join(args.out_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)
    params = QuadParams()
    cfg = RaceEnvConfig(n_courses=args.n_courses, course_seed=args.course_seed,
                        include_fixed_tracks=args.train_on_fixed,
                        course_yaw_jitter_deg=args.yaw_jitter, w_crash=args.w_crash)
    batch_size = max(1, args.n_envs * args.n_steps // 4)
    ppo_kwargs = dict(n_steps=args.n_steps, batch_size=batch_size, n_epochs=args.n_epochs,
                      learning_rate=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
                      clip_range=args.clip_range, seed=args.seed, device="cpu")
    act_fn = {"tanh": torch.nn.Tanh, "elu": torch.nn.ELU}[args.activation]
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]), activation_fn=act_fn,
                         log_std_init=args.log_std_init)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(dict(cli=vars(args), env=dataclasses.asdict(cfg),
                       quad_params=dataclasses.asdict(params),
                       ppo=dict(ppo_kwargs, policy="MlpPolicy", activation=args.activation,
                                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                                log_std_init=args.log_std_init),
                       vecnormalize=dict(norm_obs=True, norm_reward=True)),
                  f, indent=2, default=str)

    venv = RaceVecEnv(args.n_envs, cfg, params, seed=args.seed)
    if args.resume:
        src = args.resume if os.path.isdir(args.resume) else os.path.dirname(args.resume)
        venv = VecNormalize.load(os.path.join(src, "vecnormalize.pkl"), venv)
        venv.training, venv.norm_reward = True, True
        # PPO.load restores the SAVED hyperparameters; override the ones a
        # fine-tune is likely to change, or --lr etc. would be silently ignored
        lr = args.lr
        model = PPO.load(os.path.join(src, "model.zip"), env=venv, device="cpu",
                         tensorboard_log=os.path.join(run_dir, "tb"),
                         custom_objects={"learning_rate": lr, "lr_schedule": lambda _: lr,
                                         "n_epochs": args.n_epochs})
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=args.gamma)
        model = PPO("MlpPolicy", venv, policy_kwargs=policy_kwargs, verbose=1,
                    tensorboard_log=os.path.join(run_dir, "tb"), **ppo_kwargs)

    callbacks = [CheckpointCallback(save_freq=max(1, args.checkpoint_every // args.n_envs),
                                    save_path=os.path.join(run_dir, "checkpoints"),
                                    name_prefix="ppo", save_vecnormalize=True),
                 EvalTracksCallback(args.eval_every, params, args.eval_max_time)]
    t0 = time.time()
    model.learn(args.timesteps, callback=CallbackList(callbacks), reset_num_timesteps=not args.resume)
    model.save(os.path.join(run_dir, "model.zip"))
    venv.save(os.path.join(run_dir, "vecnormalize.pkl"))
    print(f"done in {time.time() - t0:.0f} s -> {run_dir}")


if __name__ == "__main__":
    main()
