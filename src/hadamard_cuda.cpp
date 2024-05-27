// https://jax.readthedocs.io/en/latest/Custom_Operation_for_GPUs.html#gpu-ops-code-listing

#include <cuda_runtime.h>  // Include CUDA runtime header
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <cstring> // For std::memcpy
#include <cmath> // For std::log2

void fwtBatchGPU(float *x, int batchSize, int log2N);

enum ElementType { BF16, F16, F32, F64 };

struct HadamardTransformDescriptor {
  int batch_size;
  int n;
  ElementType x_type;
};

void hadamard_transform(
  cudaStream_t stream, void **buffers, const char *opaque,
                        std::size_t opaque_len
) {
  void *x = buffers[0];
  void *out = buffers[1];

  HadamardTransformDescriptor args;
  assert(sizeof(HadamardTransformDescriptor) == opaque_len);
  memcpy(&args, opaque, opaque_len);

  auto log2N = long(log2(args.n));

  // Synchronously copy data from input to output buffer
  size_t num_bytes = args.n * args.batch_size * sizeof(float);
  cudaMemcpy(out, x, num_bytes, cudaMemcpyDeviceToDevice);

  // TODO: maybe try this to see if it increases performance...?
  // Alternatively we could do the copy asynchronously but then we'd need to
  // pass the stream  into fwtBatchGPU and also into the cuda kernel calls.

  // cudaMemcpyAsync(out, x, num_bytes, cudaMemcpyDeviceToDevice, stream);

  fwtBatchGPU(static_cast<float *>(out), args.batch_size, log2N);

  // optinoally check for any errors.
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
  }
}

template <typename T> pybind11::capsule EncapsulateFunction(T *fn) {
  return pybind11::capsule(reinterpret_cast<void *>(fn),
                           "xla._CUSTOM_CALL_TARGET");
}

template <typename T>
inline std::string PackDescriptorAsString(const T &descriptor) {
  return std::string(reinterpret_cast<const char *>(&descriptor), sizeof(T));
}


template <typename T> pybind11::bytes PackDescriptor(const T &descriptor) {
  return pybind11::bytes(PackDescriptorAsString(descriptor));
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["hadamard_transform"] = EncapsulateFunction(hadamard_transform);
  return dict;
}

PYBIND11_MODULE(hadamard_transform_cuda, m) {
  m.doc() = "Hadamard Transform";
  m.def("get_registrations", &Registrations);
  m.def("create_hadamard_transform_descriptor", [](int batch_size, int n,
                                                   ElementType x_type) {
    return PackDescriptor(HadamardTransformDescriptor{batch_size, n, x_type});
  });

  pybind11::enum_<ElementType>(m, "ElementType")
      .value("BF16", ElementType::BF16)
      .value("F16", ElementType::F16)
      .value("F32", ElementType::F32)
      .value("F64", ElementType::F64);
}
