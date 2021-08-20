#ifndef GPU_TEST_HELPER_H
#define GPU_TEST_HELPER_H

#include "Serializer.h"

#ifdef EIGEN_GPUCC
#define EIGEN_USE_GPU
#include <unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>
#endif // EIGEN_GPUCC

namespace Eigen {

namespace internal {

// C++14 integer/index_sequence.
#if defined(__cpp_lib_integer_sequence) && __cpp_lib_integer_sequence >= 201304L && EIGEN_MAX_CPP_VER >= 14

using std::integer_sequence;
using std::make_integer_sequence;

using std::index_sequence;
using std::make_index_sequence;

#else 

template <typename T, T... Ints>
struct integer_sequence {
  static EIGEN_CONSTEXPR size_t size() EIGEN_NOEXCEPT { return sizeof...(Ints); }
};

namespace {

template <typename T, typename Sequence, T N>
struct append_integer;

template<typename T, T... Ints, T N>
struct append_integer<T, integer_sequence<T, Ints...>, N> {
  using type = integer_sequence<T, Ints..., N>;
};

template<typename T, size_t N>
struct generate_integer_sequence {
  using type = typename append_integer<T, typename generate_integer_sequence<T, N-1>::type, N-1>::type;
};

template<typename T>
struct generate_integer_sequence<T, 0> {
  using type = integer_sequence<T>;
};

} // namespace

template <typename T, size_t N>
using make_integer_sequence = typename generate_integer_sequence<T, N>::type;

template<size_t... Ints>
using index_sequence = integer_sequence<size_t, Ints...>;

template<size_t N>
using make_index_sequence = make_integer_sequence<size_t, N>;

#endif

// Determine if a type can be considered an "output" parameter - i.e. is
// a non-const l-value reference.
template<typename T, typename EnableIf = void>
struct is_output_parameter_type : std::false_type {
  enum { value = 0 };
};
template<typename T>
struct is_output_parameter_type<T, 
    typename std::enable_if <
      std::is_lvalue_reference<T>::value 
      && !std::is_const<typename std::remove_reference<T>::type>::value
    >::type
  > : public std::true_type {
  enum { value = 1 };
};

// Size of buffer if serializing only output parameters.
template<typename T = void>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE size_t serialize_outputs_only_size() {
  return 0;
}
template<typename T, typename... Args>
EIGEN_DEVICE_FUNC size_t serialize_outputs_only_size(const T& value, const Args&... args) {
  size_t size = is_output_parameter_type<T>::value ? serialize_size(value) : 0;
  return size + serialize_outputs_only_size<Args...>(args...);
}

// Serialize only output parameters.
template<typename T = void>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
uint8_t* serialize_outputs_only(uint8_t* buffer) {
  return buffer;
}
template<typename T, typename... Args>
EIGEN_DEVICE_FUNC
uint8_t* serialize_outputs_only(uint8_t* buffer, const T& value, const Args&... args) {
  if (is_output_parameter_type<T>::value) { buffer = serialize(buffer, value); }
  return serialize_outputs_only<Args...>(buffer, args...);
}

// Only call deserialize on output types (otherwise could violate const).
template<typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
uint8_t* maybe_deserialize_output(uint8_t* buffer, T& value, std::true_type) {
  return deserialize(buffer, value);
}
template<typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
uint8_t* maybe_deserialize_output(uint8_t* buffer, T&, std::false_type) {
  return buffer;
}

// Deserialize only output parameter types.
template<typename T = void>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
uint8_t* deserialize_outputs_only(uint8_t* buffer) { return buffer; }
template<typename T, typename... Args>
EIGEN_DEVICE_FUNC 
uint8_t* deserialize_outputs_only(uint8_t* buffer, T& value, Args&... args) {
  if (is_output_parameter_type<T>::value) {
    buffer = maybe_deserialize_output<T>(buffer, value, is_output_parameter_type<T>());
  }
  return deserialize_outputs_only<Args...>(buffer, args...);
}

// Helper struct for dealing with Generic functors that may return void.
struct void_helper {
  struct Void {};
  
  // void -> Void, T otherwise.
  template<typename T>
  using return_type = typename std::conditional<std::is_same<T, void>::value,
                                                Void, T>::type;
  
  // Non-void return value.
  template<typename Func, typename... Args>
  static EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  auto call(Func&& func, Args&&... args) -> 
      typename std::enable_if<!std::is_same<decltype(func(args...)), void>::value, 
                              decltype(func(args...))>::type {
    return func(std::forward<Args>(args)...);
  }
  
  // Void return value.
  template<typename Func, typename... Args>
  static EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  auto call(Func&& func, Args&&... args) -> 
      typename std::enable_if<std::is_same<decltype(func(args...)), void>::value,
                              Void>::type {
    func(std::forward<Args>(args)...);
    return Void{};
  }
  
  // Restores the original return type, Void -> void, T otherwise.
  template<typename T>
  static EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  typename std::enable_if<!std::is_same<typename std::decay<T>::type, Void>::value, T>::type
  restore(T&& val) {
    return val;
  }
  
  // Void specialization.
  template<typename T = void>
  static EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  void restore(const Void&) {}
};

// Runs a kernel via serialized buffer.  Does this by deserializing the buffer
// to construct the arguments, calling the kernel, then re-serialing the outputs.
// The buffer contains
//     [ input_buffer_size, args ]
// After the kernel call, it is then populated with
//     [ output_buffer_size, output_parameters, return_value ]
// If the output_buffer_size exceeds the buffer's capacity, then only the
// output_buffer_size is populated.
template<typename Kernel, typename... Args, size_t... Indices>
EIGEN_DEVICE_FUNC
void run_serialized(Kernel kernel, uint8_t* buffer, size_t capacity,
                    index_sequence<Indices...>) {
  // Deserialize input size and inputs.
  size_t input_size;
  uint8_t* buff_ptr = Eigen::deserialize(buffer, input_size);
  std::tuple<typename std::decay<Args>::type...> args; // Value-type instances.
  EIGEN_UNUSED_VARIABLE(args) // Avoid NVCC compile warning.
  buff_ptr = Eigen::deserialize(buff_ptr, std::get<Indices>(args)...);
  
  // Call function, with void->Void conversion so we are guaranteed a complete
  // output type.
  auto result = void_helper::call(kernel, std::get<Indices>(args)...);
  
  // Serialize required buffer size and outputs.
  size_t output_size = sizeof(size_t);
  output_size += serialize_outputs_only_size<Args...>(std::get<Indices>(args)...);
  output_size += serialize_size(result);
  buff_ptr = Eigen::serialize(buffer, output_size);
  if (output_size <= capacity) {
    // Collect outputs and result.
    buff_ptr = serialize_outputs_only<Args...>(buff_ptr, std::get<Indices>(args)...);
    buff_ptr = Eigen::serialize(buff_ptr, result);
  }
}

#ifdef EIGEN_GPUCC

// Checks for GPU errors and asserts / prints the error message.
#define GPU_CHECK(expr)                                                \
do {                                                                   \
  gpuError_t err = expr;                                               \
  if (err != gpuSuccess) {                                             \
    printf("%s: %s\n", gpuGetErrorName(err), gpuGetErrorString(err));  \
    gpu_assert(false);                                                 \
  }                                                                    \
} while(0)

// Calls run_serialized on the GPU.
template<typename Kernel, typename... Args>
__global__
EIGEN_HIP_LAUNCH_BOUNDS_1024
void run_serialized_on_gpu_meta_kernel(
    const Kernel kernel, uint8_t* buffer, size_t capacity)
{
  run_serialized<Kernel, Args...>(kernel, buffer, capacity,
                                  make_index_sequence<sizeof...(Args)>{});
}

// Runs kernel(args...) on the GPU via the serialization mechanism.  This may
// end up calling the kernel multiple times if the initial output buffer is
// not large enough to hold the outputs.
template<typename Kernel, typename... Args>
auto run_serialized_on_gpu(Kernel kernel, Args&&... args)
    -> decltype(kernel(args...)) {  
  // Compute required serialization buffer capacity.
  // Round up input size to next power of two to give a little extra room
  // for outputs.
  size_t input_data_size = Eigen::serialize_size(args...) + sizeof(size_t);
  size_t capacity = 1;
  while (capacity <= input_data_size) {
    capacity *= 2;
  }
  std::vector<uint8_t> buffer(capacity);
  
  uint8_t* host_data = nullptr;
  uint8_t* host_ptr = nullptr;
  uint8_t* device_data = nullptr;
  size_t output_data_size = 0;
  
  do {
    // Allocate buffers and copy input data.
    capacity = std::max<size_t>(capacity, output_data_size);
    buffer.resize(capacity);
    host_data = buffer.data();
    host_ptr = Eigen::serialize(host_data, input_data_size);
    host_ptr = Eigen::serialize(host_ptr, args...);
    
    // Copy inputs to host.
    gpuFree(device_data);
    gpuMalloc((void**)(&device_data), capacity);
    gpuMemcpy(device_data, buffer.data(), input_data_size, gpuMemcpyHostToDevice);
    GPU_CHECK(gpuDeviceSynchronize());
        
    // Run kernel.
    #ifdef EIGEN_USE_HIP
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(run_serialized_on_gpu_meta_kernel<Kernel, Args...>), 
          1, 1, 0, 0, kernel, device_data, capacity);
    #else
      run_serialized_on_gpu_meta_kernel<Kernel, Args...><<<1,1>>>(
          kernel, device_data, capacity);
    #endif
    // Check pre-launch and kernel execution errors.
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());
    // Copy back new output to host.
    gpuMemcpy(host_data, device_data, capacity, gpuMemcpyDeviceToHost);
    GPU_CHECK(gpuDeviceSynchronize());
    
    // Determine output buffer size.
    host_ptr = Eigen::deserialize(host_data, output_data_size);
  } while (output_data_size > capacity);
  gpuFree(device_data);
  
  // Deserialize outputs.
  host_ptr = deserialize_outputs_only<Args...>(host_ptr, args...);
  // Maybe deserialize return value, properly handling void.
  typename void_helper::return_type<decltype(kernel(args...))> result;
  Eigen::deserialize(host_ptr, result);
  return void_helper::restore(result);
}

#endif // EIGEN_GPUCC

} // namespace internal

/**
 * Runs a kernel on the CPU, returning the results.
 * \param kernel kernel to run.
 * \param args ... input arguments.
 * \return kernel(args...).
 */
template<typename Kernel, typename... Args>
auto run_on_cpu(Kernel kernel, Args&&... args) -> decltype(kernel(args...)){  
  return kernel(std::forward<Args>(args)...);
}

#ifdef EIGEN_GPUCC

/**
 * Runs a kernel on the GPU, returning the results.
 * 
 * The kernel must be able to be passed directly as an input to a global
 * function (i.e. empty or POD).  Its inputs must be "Serializable" so we
 * can transfer them to the device, and the output must be a Serializable value
 * type so it can be transfered back from the device.
 * 
 * \param kernel kernel to run.
 * \param args ... input arguments, must be "Serializable".
 * \return kernel(args...).
 */
template<typename Kernel, typename... Args>
auto run_on_gpu(Kernel kernel, Args&&... args) -> decltype(kernel(args...)){  
  return internal::run_serialized_on_gpu<Kernel, Args...>(kernel, std::forward<Args>(args)...);
}

/**
 * Kernel for determining basic Eigen compile-time information
 * (i.e. the cuda/hip arch)
 */
struct CompileTimeDeviceInfoKernel {
  struct Info {
    int cuda;
    int hip;
  };
  
  EIGEN_DEVICE_FUNC
  Info operator()() const
  {
    Info info = {-1, -1};
    #if defined(__CUDA_ARCH__)
    info.cuda = int(__CUDA_ARCH__ +0);
    #endif
    #if defined(EIGEN_HIP_DEVICE_COMPILE)
    info.hip = int(EIGEN_HIP_DEVICE_COMPILE +0);
    #endif
    return info;
  }
};

/**
 * Queries and prints the compile-time and runtime GPU info.
 */
void print_gpu_device_info()
{
  int device = 0;
  gpuDeviceProp_t deviceProp;
  gpuGetDeviceProperties(&deviceProp, device);

  auto info = run_on_gpu(CompileTimeDeviceInfoKernel());

  std::cout << "GPU compile-time info:\n";
  
  #ifdef EIGEN_CUDACC
  std::cout << "  EIGEN_CUDACC:                " << int(EIGEN_CUDACC) << std::endl;
  #endif
  
  #ifdef EIGEN_CUDA_SDK_VER
  std::cout << "  EIGEN_CUDA_SDK_VER:          " << int(EIGEN_CUDA_SDK_VER) << std::endl;
  #endif

  #ifdef EIGEN_COMP_NVCC
  std::cout << "  EIGEN_COMP_NVCC:             " << int(EIGEN_COMP_NVCC) << std::endl;
  #endif
  
  #ifdef EIGEN_HIPCC
  std::cout << "  EIGEN_HIPCC:                 " << int(EIGEN_HIPCC) << std::endl;
  #endif

  std::cout << "  EIGEN_CUDA_ARCH:             " << info.cuda << std::endl;  
  std::cout << "  EIGEN_HIP_DEVICE_COMPILE:    " << info.hip << std::endl;

  std::cout << "GPU device info:\n";
  std::cout << "  name:                        " << deviceProp.name << std::endl;
  std::cout << "  capability:                  " << deviceProp.major << "." << deviceProp.minor << std::endl;
  std::cout << "  multiProcessorCount:         " << deviceProp.multiProcessorCount << std::endl;
  std::cout << "  maxThreadsPerMultiProcessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
  std::cout << "  warpSize:                    " << deviceProp.warpSize << std::endl;
  std::cout << "  regsPerBlock:                " << deviceProp.regsPerBlock << std::endl;
  std::cout << "  concurrentKernels:           " << deviceProp.concurrentKernels << std::endl;
  std::cout << "  clockRate:                   " << deviceProp.clockRate << std::endl;
  std::cout << "  canMapHostMemory:            " << deviceProp.canMapHostMemory << std::endl;
  std::cout << "  computeMode:                 " << deviceProp.computeMode << std::endl;
}

#endif // EIGEN_GPUCC

/**
 * Runs a kernel on the GPU (if EIGEN_GPUCC) or CPU otherwise.
 * 
 * This is to better support creating generic tests.
 * 
 * The kernel must be able to be passed directly as an input to a global
 * function (i.e. empty or POD).  Its inputs must be "Serializable" so we
 * can transfer them to the device, and the output must be a Serializable value
 * type so it can be transfered back from the device.
 * 
 * \param kernel kernel to run.
 * \param args ... input arguments, must be "Serializable".
 * \return kernel(args...).
 */
template<typename Kernel, typename... Args>
auto run(Kernel kernel, Args&&... args) -> decltype(kernel(args...)){
#ifdef EIGEN_GPUCC
  return run_on_gpu(kernel, std::forward<Args>(args)...);
#else
  return run_on_cpu(kernel, std::forward<Args>(args)...);
#endif
}


} // namespace Eigen

#endif // GPU_TEST_HELPER_H
