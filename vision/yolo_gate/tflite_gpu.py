"""
Run a .tflite model from Python on the VOXL 2's GPU (OpenCL), through libgate_tflite.so
(docker/voxl2-tflite-gpu/: TFLite built from source with the GPU delegate, plus a small C API).

    from tflite_gpu import TFLiteModel
    m = TFLiteModel("pink_det_v1_480x640_float16.tflite", delegate="gpu")
    out = m.run(nhwc_float32)            # numpy in, numpy out

The library is found via $GATE_TFLITE_LIB, else /usr/local/lib/libgate_tflite.so, else next to
this file. Only numpy and the standard library are needed.
"""

import ctypes
import os

import numpy as np

_LIB_CANDIDATES = (
    os.environ.get("GATE_TFLITE_LIB", ""),
    "/usr/local/lib/libgate_tflite.so",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "libgate_tflite.so"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "libgate_tflite.dylib"),
)

_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    tried = []
    for p in _LIB_CANDIDATES:
        if p and os.path.isfile(p):
            lib = ctypes.CDLL(p)
            break
        if p:
            tried.append(p)
    else:
        raise OSError("libgate_tflite not found (set GATE_TFLITE_LIB); tried: " + ", ".join(tried))
    lib.gt_create.restype = ctypes.c_void_p
    lib.gt_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.c_char_p, ctypes.c_int]
    ip = ctypes.POINTER(ctypes.c_int)
    lib.gt_info.restype = ctypes.c_int
    lib.gt_info.argtypes = [ctypes.c_void_p, ip, ip, ip, ip, ip]
    fp = ctypes.POINTER(ctypes.c_float)
    lib.gt_run.restype = ctypes.c_int
    lib.gt_run.argtypes = [ctypes.c_void_p, fp, ctypes.c_int, fp, ctypes.c_int]
    lib.gt_destroy.restype = None
    lib.gt_destroy.argtypes = [ctypes.c_void_p]
    _lib = lib
    return lib


class TFLiteModel:
    """One float32-in, float32-out .tflite model. delegate: "gpu" (OpenCL) or "cpu" (XNNPACK).
    A GPU that can't take the model raises (no silent CPU fallback): check .plan_nodes == 1 to
    know the whole graph is on the GPU."""

    def __init__(self, path, delegate="gpu", threads=0, fp16=True):
        lib = _load()
        err = ctypes.create_string_buffer(512)
        d = {"cpu": 0, "gpu": 1}[delegate]
        self._h = lib.gt_create(os.fsencode(path), d, int(threads), int(bool(fp16)), err, len(err))
        if not self._h:
            raise RuntimeError(f"gate_tflite: {err.value.decode(errors='replace')}")
        ind, outd = (ctypes.c_int * 4)(), (ctypes.c_int * 4)()
        rank, gpu, nodes = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        if lib.gt_info(self._h, ind, outd, ctypes.byref(rank), ctypes.byref(gpu), ctypes.byref(nodes)):
            raise RuntimeError("gate_tflite: unexpected tensor shapes (need a 4-D input)")
        self.in_shape = tuple(ind)
        self.out_shape = tuple(outd[: rank.value])
        self.gpu = bool(gpu.value)
        self.plan_nodes = nodes.value
        self._out = np.empty(self.out_shape, dtype=np.float32)
        self._lib = lib

    def run(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.shape != self.in_shape:
            raise ValueError(f"input shape {x.shape}, model wants {self.in_shape}")
        fp = ctypes.POINTER(ctypes.c_float)
        rc = self._lib.gt_run(self._h, x.ctypes.data_as(fp), x.size, self._out.ctypes.data_as(fp),
                              self._out.size)
        if rc:
            raise RuntimeError(f"gate_tflite: run failed ({rc})")
        return self._out.copy()

    def close(self):
        if getattr(self, "_h", None):
            self._lib.gt_destroy(self._h)
            self._h = None

    def __del__(self):
        self.close()
