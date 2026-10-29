"""
Push a new Course into a running controller WITHOUT resetting it (the runner's
vision mode does this whenever the gate map changes).

    update_course(controller, course, pos=None, vel=None) -> str   # how it was done

Uniform hook: a controller with its own `update_course(course, pos, vel)` method
(the MPC) is called through it. Otherwise small adapters here mutate the
controller in place, because pid_controller.py / rl_policy.py are not edited:

    PidController   reset() caches `centers` and `normals` (N gates x 3 tensors,
                    device/dtype from reset). Rebuilt from the new course; the
                    stage machine (`stage`, `last_idx`), the velocity integrator
                    and `prev_vel` are kept, so the flight is not disturbed
                    except by the gate having moved.
    RlPolicyController   reset() caches `pool = CoursePool([course])`. Rebuilt;
                    `prev_action` (part of the observation) is kept.

A controller that has not been reset yet only gets `.course` replaced (reset()
builds its caches from it). The gate COUNT must not change.
"""

import os
import sys

import numpy as np
import torch

_RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RL not in sys.path:
    sys.path.insert(0, _RL)


def _check_count(controller, course):
    old = getattr(controller, "course", None)
    if old is not None and len(old) != len(course):
        raise ValueError(f"gate count changed: {len(old)} -> {len(course)}")


def update_pid(controller, course):
    _check_count(controller, course)
    controller.course = course
    if getattr(controller, "centers", None) is None:
        return
    kw = dict(device=controller.device, dtype=controller.dtype)
    controller.centers = torch.as_tensor(np.stack([g.center for g in course.gates]), **kw)
    controller.normals = torch.as_tensor(np.stack([g.normal for g in course.gates]), **kw)


def update_rl(controller, course):
    from race_env import CoursePool
    _check_count(controller, course)
    controller.course = course
    if getattr(controller, "pool", None) is None:
        return
    controller.pool = CoursePool([course], controller.device, controller.dtype)


def update_course(controller, course, pos=None, vel=None):
    """Returns "method" (controller.update_course), "pid" or "rl"."""
    own = getattr(controller, "update_course", None)
    if callable(own):
        own(course, pos, vel)
        return "method"
    if hasattr(controller, "centers") or type(controller).__name__ == "PidController":
        update_pid(controller, course)
        return "pid"
    if hasattr(controller, "pool") or type(controller).__name__ == "RlPolicyController":
        update_rl(controller, course)
        return "rl"
    raise TypeError(f"don't know how to update the course of {type(controller).__name__}")
