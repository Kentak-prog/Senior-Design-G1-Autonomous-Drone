/* opencl_check: is the GPU usable through OpenCL here (on the VOXL 2 or in the container)?
 * Lists every OpenCL platform and device, then runs a small kernel on the first GPU and checks
 * the result. Exit code 0 = the GPU computed correctly.
 *
 *   gcc -O2 -Wall -o opencl_check opencl_check.c -lOpenCL
 *
 * On the VOXL 2 expect one platform, one device: QUALCOMM Adreno(TM), OpenCL 2.0 Adreno(TM) 650. */

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1048576

static const char *SRC =
    "__kernel void saxpy(const float a, __global const float *x, __global float *y) {\n"
    "  int i = get_global_id(0);\n"
    "  y[i] = a * x[i] + y[i];\n"
    "}\n";

#define CHECK(err, what)                                              \
  do {                                                                \
    if ((err) != CL_SUCCESS) {                                        \
      fprintf(stderr, "FAIL: %s (OpenCL error %d)\n", (what), (int)(err)); \
      return 1;                                                       \
    }                                                                 \
  } while (0)

int main(void) {
  cl_uint np = 0;
  cl_int err = clGetPlatformIDs(0, NULL, &np);
  if (err != CL_SUCCESS || np == 0) {
    fprintf(stderr, "FAIL: no OpenCL platform (error %d). In Docker: are /dev/kgsl-3d0 and "
                    "/dev/ion passed in, and the qti-adreno driver installed?\n", (int)err);
    return 1;
  }
  cl_platform_id plats[8];
  if (np > 8) np = 8;
  clGetPlatformIDs(np, plats, NULL);
  printf("OpenCL platforms: %u\n", np);

  cl_device_id gpu = NULL;
  for (cl_uint p = 0; p < np; ++p) {
    char name[256] = "";
    clGetPlatformInfo(plats[p], CL_PLATFORM_NAME, sizeof name, name, NULL);
    cl_uint nd = 0;
    clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 0, NULL, &nd);
    printf("platform %u: %s, %u device(s)\n", p, name, nd);
    cl_device_id devs[8];
    if (nd > 8) nd = 8;
    if (nd == 0) continue;
    clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, nd, devs, NULL);
    for (cl_uint d = 0; d < nd; ++d) {
      char dname[256] = "", hw[256] = "", drv[256] = "";
      cl_uint cu = 0;
      cl_ulong mem = 0;
      cl_device_type type = 0;
      clGetDeviceInfo(devs[d], CL_DEVICE_NAME, sizeof dname, dname, NULL);
      clGetDeviceInfo(devs[d], CL_DEVICE_VERSION, sizeof hw, hw, NULL);
      clGetDeviceInfo(devs[d], CL_DRIVER_VERSION, sizeof drv, drv, NULL);
      clGetDeviceInfo(devs[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cu, &cu, NULL);
      clGetDeviceInfo(devs[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof mem, &mem, NULL);
      clGetDeviceInfo(devs[d], CL_DEVICE_TYPE, sizeof type, &type, NULL);
      printf("  device %u: %s | %s | driver %s | %u compute units | %.0f MB\n", d, dname, hw, drv,
             cu, mem / 1048576.0);
      if (!gpu && (type & CL_DEVICE_TYPE_GPU)) gpu = devs[d];
    }
  }
  if (!gpu) {
    fprintf(stderr, "FAIL: no GPU device\n");
    return 1;
  }

  cl_context ctx = clCreateContext(NULL, 1, &gpu, NULL, NULL, &err);
  CHECK(err, "clCreateContext");
  cl_command_queue q = clCreateCommandQueueWithProperties(ctx, gpu, NULL, &err);
  CHECK(err, "clCreateCommandQueue");
  cl_program prog = clCreateProgramWithSource(ctx, 1, &SRC, NULL, &err);
  CHECK(err, "clCreateProgramWithSource");
  err = clBuildProgram(prog, 1, &gpu, NULL, NULL, NULL);
  CHECK(err, "clBuildProgram");
  cl_kernel k = clCreateKernel(prog, "saxpy", &err);
  CHECK(err, "clCreateKernel");

  float *x = malloc(N * sizeof(float)), *y = malloc(N * sizeof(float));
  for (int i = 0; i < N; ++i) { x[i] = (float)i; y[i] = 1.0f; }
  cl_mem bx = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(float), x, &err);
  CHECK(err, "clCreateBuffer x");
  cl_mem by = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(float), y, &err);
  CHECK(err, "clCreateBuffer y");
  float a = 2.0f;
  clSetKernelArg(k, 0, sizeof a, &a);
  clSetKernelArg(k, 1, sizeof bx, &bx);
  clSetKernelArg(k, 2, sizeof by, &by);
  size_t gsz = N;
  err = clEnqueueNDRangeKernel(q, k, 1, NULL, &gsz, NULL, 0, NULL, NULL);
  CHECK(err, "clEnqueueNDRangeKernel");
  err = clEnqueueReadBuffer(q, by, CL_TRUE, 0, N * sizeof(float), y, 0, NULL, NULL);
  CHECK(err, "clEnqueueReadBuffer");

  int bad = 0;
  for (int i = 0; i < N; ++i) {
    if (y[i] != 2.0f * (float)i + 1.0f) { ++bad; }
  }
  printf(bad ? "FAIL: %d of %d results wrong\n" : "OK: the GPU ran a %d-element kernel correctly\n",
         bad ? bad : N, N);

  clReleaseMemObject(bx); clReleaseMemObject(by); clReleaseKernel(k); clReleaseProgram(prog);
  clReleaseCommandQueue(q); clReleaseContext(ctx); free(x); free(y);
  return bad ? 1 : 0;
}
