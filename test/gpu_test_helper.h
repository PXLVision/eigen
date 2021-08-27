#ifndef GPU_TEST_HELPER_H
#define GPU_TEST_HELPER_H

#include <Eigen/Core>

#ifdef EIGEN_GPUCC
#define EIGEN_USE_GPU
#include <unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>
#endif // EIGEN_GPUCC

namespace Eigen {

namespace internal {

template<size_t N, size_t Idx, typename OutputIndexSequence, typename... Ts>
struct extract_output_indices_helper;

/**
 * Extracts a set of indices corresponding to non-const l-value reference
 * output types.
 *
 * \internal
 * \tparam N the number of types {T1, Ts...}.
 * \tparam Idx the "index" to append if T1 is an output type.
 * \tparam OutputIndices the current set of output indices.
 * \tparam T1 the next type to consider, with index Idx.
 * \tparam Ts the remaining types.
 */
template<size_t N, size_t Idx, size_t... OutputIndices, typename T1, typename... Ts>
struct extract_output_indices_helper<N, Idx, index_sequence<OutputIndices...>, T1, Ts...> {
  using type = typename
    extract_output_indices_helper<
      N - 1, Idx + 1,
      typename std::conditional<
        // If is a non-const l-value reference, append index.
        std::is_lvalue_reference<T1>::value 
          && !std::is_const<typename std::remove_reference<T1>::type>::value,
        index_sequence<OutputIndices..., Idx>,
        index_sequence<OutputIndices...> >::type,
      Ts...>::type;
};

// Base case.
template<size_t Idx, size_t... OutputIndices>
struct extract_output_indices_helper<0, Idx, index_sequence<OutputIndices...> > {
  using type = index_sequence<OutputIndices...>;
};

// Extracts a set of indices into Types... that correspond to non-const
// l-value references.
template<typename... Types>
using extract_output_indices = typename extract_output_indices_helper<sizeof...(Types), 0, index_sequence<>, Types...>::type;

// Helper struct for dealing with Generic functors that may return void.
struct void_helper {
  struct Void {};
  
  // Converts void -> Void, T otherwise.
  template<typename T>
  using ReturnType = typename std::conditional<std::is_same<T, void>::value, Void, T>::type;
  
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
  
  // Void case.
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
template<typename Kernel, typename... Args, size_t... Indices, size_t... OutputIndices>
EIGEN_DEVICE_FUNC
void run_serialized(index_sequence<Indices...>, index_sequence<OutputIndices...>,
                    Kernel kernel, uint8_t* buffer, size_t capacity) {
  // Deserialize input size and inputs.
  size_t input_size;
  uint8_t* buff_ptr = Eigen::deserialize(buffer, input_size);
  Eigen::Tuple<typename std::decay<Args>::type...> args = 
    Eigen::make_tuple(typename std::decay<Args>::type{}...); // Value-type instances.
  EIGEN_UNUSED_VARIABLE(args) // Avoid NVCC compile warning.
  // NVCC 9.1 requires us to spell out the template parameters explicitly.
  buff_ptr = Eigen::deserialize(buff_ptr, get<Indices, typename std::decay<Args>::type...>(args)...);
  
  // Call function, with void->Void conversion so we are guaranteed a complete
  // output type.
  auto result = void_helper::call(kernel, get<Indices, typename std::decay<Args>::type...>(args)...);
  
  // Determine required output size.
  size_t output_size = sizeof(size_t);
  output_size += Eigen::serialize_size(Eigen::get<OutputIndices, typename std::decay<Args>::type...>(args)...);
  output_size += Eigen::serialize_size(result);
  
  // Always serialize required buffer size.
  buff_ptr = Eigen::serialize(buffer, output_size);
  // Serialize outputs if they fit in the buffer.
  if (output_size <= capacity) {
    // Collect outputs and result.
    buff_ptr = Eigen::serialize(buff_ptr, Eigen::get<OutputIndices, typename std::decay<Args>::type...>(args)...);
    buff_ptr = Eigen::serialize(buff_ptr, result);
  }
}

template<typename Kernel, typename... Args>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
void run_serialized(Kernel kernel, uint8_t* buffer, size_t capacity) {
  run_serialized<Kernel, Args...> (make_index_sequence<sizeof...(Args)>{},
                                   extract_output_indices<Args...>{},
                                   kernel, buffer, capacity);
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
void run_serialized_on_gpu_meta_kernel(const Kernel kernel, uint8_t* buffer, size_t capacity) {
  run_serialized<Kernel, Args...>(kernel, buffer, capacity);
}

// Runs kernel(args...) on the GPU via the serialization mechanism. 
//
// Note: this may end up calling the kernel multiple times if the initial output
// buffer is not large enough to hold the outputs.
template<typename Kernel, typename... Args, size_t... Indices, size_t... OutputIndices>
auto run_serialized_on_gpu(index_sequence<Indices...>, 
                           index_sequence<OutputIndices...>,
                           Kernel kernel, Args&&... args) -> decltype(kernel(args...)) {  
  // Compute the required serialization buffer capacity.
  // Round up input size to next power of two to give a little extra room
  // for outputs.
  size_t input_data_size = sizeof(size_t) + Eigen::serialize_size(args...);
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
  auto args_tuple = Eigen::tie(args...);
  EIGEN_UNUSED_VARIABLE(args_tuple)  // Avoid NVCC compile warning.
  host_ptr = Eigen::deserialize(host_ptr, Eigen::get<OutputIndices, Args...>(args_tuple)...);
  
  // Maybe deserialize return value, properly handling void.
  typename void_helper::ReturnType<decltype(kernel(args...))> result;
  host_ptr = Eigen::deserialize(host_ptr, result);
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
  return internal::run_serialized_on_gpu<Kernel, Args...>(
      internal::make_index_sequence<sizeof...(Args)>{},
      internal::extract_output_indices<Args...>{},
      kernel, std::forward<Args>(args)...);
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
 * Runs a kernel on the GPU (if EIGEN_GPUCC), or CPU otherwise.
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
