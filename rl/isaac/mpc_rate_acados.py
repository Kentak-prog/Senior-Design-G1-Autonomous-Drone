"""
acados backend for RateAwareMpc (mpc_rate.py): same model, limits and cost,
solved with SQP-RTI (one Gauss-Newton SQP step per tick, HPIPM QP) on
generated C code. Selected with RateAwareMpc(..., backend="acados").

Needs acados built (ACADOS_SOURCE_DIR, LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib)
and acados_template installed; see README_mpc_rate.md. The generated code
goes to $ACADOS_CODEGEN_DIR (default ~/.cache/acados_rate_mpc/<hash>), one
build per model/weights/horizon, reused on later runs.

DELTA-U FORMULATION. The commands [T_cmd, w_cmd] are 4 extra STATES and the
optimizer's input is their rate of change du (per second): uc' = du. This
gives the IPOPT backend's penalty on the change between consecutive inputs
(|U_k - U_k-1|^2, weight "du") as a stage cost on (du * h). A first version
with the commands as inputs and only a same-stage smoothing term chattered
at 100-190 deg/s when the model was wrong (40 ms delay the MPC was not told,
or a 2x faster rate loop), where IPOPT stayed at 3-7 deg/s; more SQP
iterations per tick did not help, the missing stage-to-stage penalty did.
The initial command state is the last command sent; the new command is
that state moved one control tick along du: uc0 + du0 * control_dt.

COST as nonlinear least squares, 0.5 * ||y||^2_W:
    y = [p - p_ref (3), v - v_ref (3), vec(q_ref^-1 (x) q) (3), w - w_ref (3),
         T_cmd - T_ff, w_cmd (3), du * h (4)]
    |vec(q_ref^-1 (x) q)|^2 = 1 - <q, q_ref>^2 for unit quaternions, the IPOPT
    backend's attitude term. W = 2 x the IPOPT weights (IPOPT has no 0.5), and
    cost_scaling = 1 per stage (acados' default scales stage costs by h).
"""
import hashlib
import json
import os

import casadi as ca
import numpy as np

from mpc_rate import model_rhs


def _quat_err_vec(q, qr):
    """Vector part of qr^-1 (x) q (WXYZ)."""
    w1, x1, y1, z1 = qr[0], -qr[1], -qr[2], -qr[3]
    w2, x2, y2, z2 = q[0], q[1], q[2], q[3]
    return ca.vertcat(w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


class AcadosRateSolver:
    NX = 18                                       # 14 model states + 4 command states

    def __init__(self, ctl):
        from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
        p, N, h, w = ctl.p, ctl.Nh, ctl.dtm, ctl.wts
        self.N, self.ctrl_dt = N, ctl.ctrl_dt
        xm, um, xdot_m = model_rhs(p, ctl.tau_w, ctl.tau_T)
        f = ca.Function("f", [xm, um], [xdot_m])
        uc = ca.SX.sym("uc", 4)                   # [T_cmd, w_cmd] as states
        du = ca.SX.sym("du", 4)                   # their rate of change: the optimizer's input
        x = ca.vertcat(xm, uc)
        xdot = ca.vertcat(f(xm, uc), du)
        ref = ca.SX.sym("ref", 14)
        model = AcadosModel()
        model.x, model.u, model.p, model.f_expl_expr = x, du, ref, xdot
        model.xdot = ca.SX.sym("xdot", self.NX)
        model.f_impl_expr = model.xdot - xdot
        mr = np.asarray(p.max_rate, dtype=float)
        tag = json.dumps(dict(m=p.mass, I=list(p.inertia_diag), c=list(p.drag_coeff), tw=list(map(float, ctl.tau_w)),
                              tT=ctl.tau_T, N=N, h=h, w=w, mr=list(mr), Tmax=p.max_total_thrust(), formulation="du-1"),
                         sort_keys=True)
        hsh = hashlib.sha1(tag.encode()).hexdigest()[:10]
        model.name = f"rate_mpc_{hsh}"

        track = ca.vertcat(xm[0:3] - ref[0:3], xm[3:6] - ref[3:6], _quat_err_vec(xm[6:10], ref[6:10]),
                           xm[10:13] - ref[10:13])
        y_stage = ca.vertcat(track, uc[0] - ref[13], uc[1:4], du * h)
        Wt = np.concatenate([w["p"], w["v"], [w["att"]] * 3, [w["w"]] * 3])
        Ws = 2.0 * np.concatenate([Wt, [w["T"]], [w["wc"]] * 3, [w["du"]] * 4])
        We = 2.0 * w["term"] * Wt

        ocp = AcadosOcp()
        ocp.model = model
        so = ocp.solver_options
        so.N_horizon, so.tf = N, N * h
        ocp.cost.cost_type, ocp.cost.cost_type_e = "NONLINEAR_LS", "NONLINEAR_LS"
        model.cost_y_expr, model.cost_y_expr_e = y_stage, track
        ocp.cost.W, ocp.cost.W_e = np.diag(Ws), np.diag(We)
        ocp.cost.yref, ocp.cost.yref_e = np.zeros(y_stage.shape[0]), np.zeros(track.shape[0])
        lo = np.array([0.1 * p.mass * p.gravity, -mr[0], -mr[1], -mr[2]])
        hi = np.array([p.max_total_thrust(), mr[0], mr[1], mr[2]])
        ocp.constraints.idxbx = np.arange(14, 18)
        ocp.constraints.lbx, ocp.constraints.ubx = lo, hi
        ocp.constraints.idxbx_e = np.arange(14, 18)
        ocp.constraints.lbx_e, ocp.constraints.ubx_e = lo, hi
        ocp.constraints.x0 = np.zeros(self.NX)
        ocp.parameter_values = np.zeros(14)
        so.integrator_type, so.sim_method_num_stages, so.sim_method_num_steps = "ERK", 4, 2
        so.nlp_solver_type = "SQP_RTI"
        so.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        so.hessian_approx = "GAUSS_NEWTON"
        so.qp_solver_cond_N = max(1, N // 4)
        so.cost_scaling = np.ones(N + 1)          # sum stage costs unscaled, like the IPOPT backend
        so.print_level = 0
        self.lo, self.hi = lo, hi

        base = os.environ.get("ACADOS_CODEGEN_DIR", os.path.expanduser("~/.cache/acados_rate_mpc"))
        out = os.path.join(base, hsh)
        os.makedirs(out, exist_ok=True)
        ocp.code_export_directory = os.path.join(out, "c_generated_code")
        json_name = f"{model.name}.json"           # acados writes it inside code_export_directory
        built = os.path.exists(os.path.join(ocp.code_export_directory, json_name)) and \
            os.path.exists(os.path.join(ocp.code_export_directory, f"libacados_ocp_solver_{model.name}.so"))
        cwd = os.getcwd()
        os.chdir(out)                              # keep acados' relative outputs out of the repo
        try:
            self.solver = AcadosOcpSolver(ocp, json_file=json_name, generate=not built, build=not built,
                                          verbose=False)
        finally:
            os.chdir(cwd)
        self.n_failures = 0
        self.Xg = None

    def reset(self, u_hover, N):
        self.Xg = None

    def solve(self, x0, ref, u_prev):
        s, N = self.solver, self.N
        xa = np.concatenate([x0, u_prev])          # the command state starts at the last command sent
        cold = self.Xg is None
        if cold:
            self.Xg = np.tile(xa.reshape(-1, 1), (1, N + 1))
        for k in range(N + 1):
            s.set(k, "p", ref[:, k])
            s.set(k, "x", self.Xg[:, k])
            if k < N:
                s.set(k, "u", np.zeros(4) if cold else self.Ug[:, k])
        s.set(0, "lbx", xa)
        s.set(0, "ubx", xa)
        iters = int(os.environ.get("MPC_RATE_RTI_ITERS", "1"))
        for _ in range(8 if cold else iters):      # the first call starts from a flat guess: iterate
            status = s.solve()
        if status not in (0, 2):
            self.n_failures += 1
        self.Xg = np.column_stack([s.get(k, "x") for k in range(N + 1)])
        self.Ug = np.column_stack([s.get(k, "u") for k in range(N)])
        u = u_prev + self.Ug[:, 0] * self.ctrl_dt  # the command one control tick along du
        return np.clip(u, self.lo, self.hi)
