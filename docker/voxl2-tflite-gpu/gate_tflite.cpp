// gate_tflite: a minimal C API around TensorFlow Lite, so Python (ctypes + numpy) can run a
// .tflite model on the VOXL 2's Adreno GPU through TFLite's OpenCL GPU delegate.
//
// TFLite's Python packages (tflite-runtime, ai-edge-litert) do not ship the GPU delegate, and
// ModalAI's qrb5165-tflite package has no Python bindings. This file links TFLite (built from
// source, with the GPU delegate) into one shared library: vision/yolo_gate/tflite_gpu.py loads it.
//
// One model, one float32 input, one float32 output (the YOLO detect head). Not thread-safe:
// one handle per thread.

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#ifdef GT_WITH_GPU
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif

namespace {

struct Handle {
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interp;
  TfLiteDelegate* gpu = nullptr;
  int gpu_used = 0;
  ~Handle() {
    interp.reset();  // the interpreter must go before the delegate it uses
#ifdef GT_WITH_GPU
    if (gpu) TfLiteGpuDelegateV2Delete(gpu);
#endif
  }
};

void set_err(char* err, int err_len, const std::string& msg) {
  if (err && err_len > 0) {
    std::snprintf(err, static_cast<size_t>(err_len), "%s", msg.c_str());
  }
}

}  // namespace

extern "C" {

// delegate: 0 = CPU (XNNPACK), 1 = GPU (OpenCL only; fails instead of falling back to CPU).
// allow_fp16: GPU computes in fp16 (fastest); 0 = fp32 on the GPU.
// num_threads: CPU threads (CPU ops, and the part of the graph the GPU does not take); <= 0 = default.
// Returns NULL on failure, with the reason in err.
void* gt_create(const char* model_path, int delegate, int num_threads, int allow_fp16, char* err,
                int err_len) {
  auto h = std::make_unique<Handle>();
  h->model = tflite::FlatBufferModel::BuildFromFile(model_path);
  if (!h->model) {
    set_err(err, err_len, std::string("cannot load model ") + model_path);
    return nullptr;
  }
  tflite::ops::builtin::BuiltinOpResolver resolver;
  if (tflite::InterpreterBuilder(*h->model, resolver)(&h->interp) != kTfLiteOk || !h->interp) {
    set_err(err, err_len, "InterpreterBuilder failed (an op this TFLite version does not know?)");
    return nullptr;
  }
  if (num_threads > 0) h->interp->SetNumThreads(num_threads);

  if (delegate == 1) {
#ifdef GT_WITH_GPU
    TfLiteGpuDelegateOptionsV2 opt = TfLiteGpuDelegateOptionsV2Default();
    opt.is_precision_loss_allowed = allow_fp16 ? 1 : 0;
    opt.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
    opt.inference_priority1 = allow_fp16 ? TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY
                                         : TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
    opt.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY;  // OpenCL, never OpenGL
    h->gpu = TfLiteGpuDelegateV2Create(&opt);
    if (!h->gpu) {
      set_err(err, err_len, "TfLiteGpuDelegateV2Create failed");
      return nullptr;
    }
    if (h->interp->ModifyGraphWithDelegate(h->gpu) != kTfLiteOk) {
      set_err(err, err_len,
              "GPU delegate could not take the graph (no OpenCL device? check /dev/kgsl-3d0 and "
              "libOpenCL.so in the container)");
      return nullptr;
    }
    h->gpu_used = 1;
#else
    set_err(err, err_len, "built without the GPU delegate (GT_WITH_GPU off)");
    return nullptr;
#endif
  }

  if (h->interp->AllocateTensors() != kTfLiteOk) {
    set_err(err, err_len, "AllocateTensors failed");
    return nullptr;
  }
  if (h->interp->inputs().size() != 1 || h->interp->outputs().size() < 1) {
    set_err(err, err_len, "expected 1 input and at least 1 output");
    return nullptr;
  }
  if (h->interp->input_tensor(0)->type != kTfLiteFloat32 ||
      h->interp->output_tensor(0)->type != kTfLiteFloat32) {
    set_err(err, err_len, "input and output 0 must be float32 (export float32/float16, not int8)");
    return nullptr;
  }
  return h.release();
}

// in_dims[4], out_dims[4] (padded with 0), *out_rank, *gpu_used, *plan_nodes.
// plan_nodes = nodes in the execution plan: 1 with the GPU = the whole graph runs on the GPU; more =
// the GPU took only part of it and the rest runs on the CPU (slower: copies between the two).
int gt_info(void* vh, int* in_dims, int* out_dims, int* out_rank, int* gpu_used, int* plan_nodes) {
  auto* h = static_cast<Handle*>(vh);
  const TfLiteTensor* in = h->interp->input_tensor(0);
  const TfLiteTensor* out = h->interp->output_tensor(0);
  if (in->dims->size != 4 || out->dims->size > 4) return -1;
  for (int i = 0; i < 4; ++i) {
    in_dims[i] = in->dims->data[i];
    out_dims[i] = i < out->dims->size ? out->dims->data[i] : 0;
  }
  *out_rank = out->dims->size;
  *gpu_used = h->gpu_used;
  *plan_nodes = static_cast<int>(h->interp->execution_plan().size());
  return 0;
}

// input: n_in floats in the input tensor's layout (NHWC); output: n_out floats. 0 = OK.
int gt_run(void* vh, const float* input, int n_in, float* output, int n_out) {
  auto* h = static_cast<Handle*>(vh);
  TfLiteTensor* in = h->interp->input_tensor(0);
  if (static_cast<size_t>(n_in) * sizeof(float) != in->bytes) return -1;
  std::memcpy(in->data.f, input, in->bytes);
  if (h->interp->Invoke() != kTfLiteOk) return -2;
  const TfLiteTensor* out = h->interp->output_tensor(0);
  if (static_cast<size_t>(n_out) * sizeof(float) != out->bytes) return -3;
  std::memcpy(output, out->data.f, out->bytes);
  return 0;
}

void gt_destroy(void* vh) { delete static_cast<Handle*>(vh); }

}  // extern "C"
