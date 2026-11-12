# Run once in the Isaac Script Editor:  exec(open('/workspace/race/r.py').read())
# Installs a Kit update-loop poller: whenever /workspace/race/isaac_next.py appears, it is
# renamed to isaac_next.running.py and exec'd in a fresh globals dict (async code: define
# and schedule your own coroutine). Output/tracebacks go to /workspace/race/isaac_next.log.
import os, traceback, builtins, omni.kit.app

_NEXT = "/workspace/race/isaac_next.py"
_RUN = "/workspace/race/isaac_next.running.py"
_LOG = "/workspace/race/isaac_next.log"

old = getattr(builtins, "_race_poller_sub", None)
if old is not None:
    old.unsubscribe()


def _race_poll(_e):
    if not os.path.exists(_NEXT):
        return
    try:
        os.replace(_NEXT, _RUN)
        src = open(_RUN).read()
        with open(_LOG, "a") as f:
            f.write(f"--- exec {len(src)} bytes\n")
        g = {"__name__": "__main__", "__file__": _RUN}
        exec(compile(src, _RUN, "exec"), g)
        with open(_LOG, "a") as f:
            f.write("--- ok\n")
    except Exception:
        with open(_LOG, "a") as f:
            f.write(traceback.format_exc())


builtins._race_poller_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
    _race_poll, name="race_poller")
with open(_LOG, "a") as f:
    f.write("--- poller installed\n")
print("[race] poller installed")
