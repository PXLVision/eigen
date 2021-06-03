// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Contact: <eigen@codeplay.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int64_t
#define EIGEN_USE_SYCL

#include "main.h"
#include "OffByOneScalar.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <stdint.h>
#include <iostream>

template <typename DataType, int DataLayout, typename IndexType>
void test_device_memory(const Eigen::SyclDevice &sycl_device) {

  IndexType sizeDim1 = 10;
  array<IndexType, 1> tensorRange = {{sizeDim1}};
  Tensor<DataType, 1, DataLayout,IndexType> in(tensorRange);
  Tensor<DataType, 1, DataLayout,IndexType> in1(tensorRange);
  DataType* gpu_in_data  = static_cast<DataType*>(sycl_device.allocate(in.size()*sizeof(DataType)));
  
  // memset
  memset(in1.data(), 1, in1.size() * sizeof(DataType));
  
  sycl_device.memset(gpu_in_data, 1, in.size()*sizeof(DataType));
  sycl_device.memcpyDeviceToHost(in.data(), gpu_in_data, in.size()*sizeof(DataType));
  for (IndexType i=0; i<in.size(); i++) {
    VERIFY_IS_EQUAL(in(i), in1(i));
  }
  std::cout << "memset test done" << std::endl;
  
  // fill
  DataType value = DataType(7);
  std::fill_n(in1.data(), in1.size(), value);
  sycl_device.fill(gpu_in_data, gpu_in_data + in.size(), value);
  sycl_device.memcpyDeviceToHost(in.data(), gpu_in_data, in.size()*sizeof(DataType));
  for (IndexType i=0; i<in.size(); i++) {
    if (in(i) != in1(i)) {
      std::cout << i << ": " << double(in(i)) << " vs " << double(in1(i)) << std::endl;
    //   std::cout << std::hex << Eigen::numext::bit_cast<uint32_t>(in(i)) << " vs " << Eigen::numext::bit_cast<uint32_t>(in1(i)) << std::endl;
    }
    VERIFY_IS_EQUAL(in(i), in1(i));
  }
  std::cout << "fill test done" << std::endl;
  
  sycl_device.deallocate(gpu_in_data);
}

template <typename DataType, int DataLayout, typename IndexType>
void test_device_exceptions(const Eigen::SyclDevice &sycl_device) {
  VERIFY(sycl_device.ok());
  IndexType sizeDim1 = 100;
  array<IndexType, 1> tensorDims = {{sizeDim1}};
  DataType* gpu_data = static_cast<DataType*>(sycl_device.allocate(sizeDim1*sizeof(DataType)));
  sycl_device.memset(gpu_data, 1, sizeDim1*sizeof(DataType));

  TensorMap<Tensor<DataType, 1, DataLayout,IndexType>> in(gpu_data, tensorDims);
  TensorMap<Tensor<DataType, 1, DataLayout,IndexType>> out(gpu_data, tensorDims);
  out.device(sycl_device) = in / in.constant(0);

  sycl_device.synchronize();
  VERIFY(!sycl_device.ok());
  sycl_device.deallocate(gpu_data);
}

template<typename DataType> void sycl_device_test_per_device(const cl::sycl::device& d){
  std::cout << "Running on " << d.template get_info<cl::sycl::info::device::name>() << std::endl;
  QueueInterface queueInterface(d);
  auto sycl_device = Eigen::SyclDevice(&queueInterface);
  test_device_memory<DataType, RowMajor, int64_t>(sycl_device);
  test_device_memory<DataType, ColMajor, int64_t>(sycl_device);
  /// this test throw an exception. enable it if you want to see the exception
  //test_device_exceptions<DataType, RowMajor>(sycl_device);
  /// this test throw an exception. enable it if you want to see the exception
  //test_device_exceptions<DataType, ColMajor>(sycl_device);
}

template<typename InitialType, typename FinalType, int NumBytes = 10 * sizeof(InitialType) * sizeof(FinalType)>
void sycl_test_reinterpret(const cl::sycl::device& d){
  constexpr int initial_count = NumBytes / sizeof(InitialType);
  constexpr int final_count = NumBytes / sizeof(FinalType);

  // Fill a byte array.
  Eigen::numext::uint8_t bytes[NumBytes];
  for (int i=0; i<NumBytes; ++i) {
    bytes[i] = i;
  }
  // Copy to initial/final types.
  InitialType initial_values[initial_count];
  FinalType final_values[final_count];
  std::memcpy(initial_values, bytes, NumBytes);
  std::memcpy(final_values, bytes, NumBytes);
    
  // Example modified from
  //   https://www.codeplay.com/portal/blogs/2018/03/09/buffer-reinterpret-viewing-data-from-a-different-perspective.html
  
  // Device
  {
    cl::sycl::buffer<InitialType> host_initial(initial_values, cl::sycl::range<1>(initial_count));

    cl::sycl::buffer<InitialType> initial{cl::sycl::range<1>(initial_count)};
    cl::sycl::queue q {d};
    {
      // setup test data using the initial buffer accessor
      q.submit([&](cl::sycl::handler& cgh) {
        auto initialAcc = initial.template get_access<cl::sycl::access::mode::write,
                                                      cl::sycl::access::target::global_buffer>(cgh);
        auto hostAcc = host_initial.template get_access<cl::sycl::access::mode::read,
                                                        cl::sycl::access::target::global_buffer>(cgh);
        cgh.copy(hostAcc, initialAcc);
      });
      q.wait_and_throw();
    }
    {
      auto initialAcc = initial.template get_access<cl::sycl::access::mode::read>();
      for (size_t i = 0; i < initial_count; ++i) {
        if (initialAcc[i] != initial_values[i] && !(Eigen::numext::isnan(initialAcc[i]) && Eigen::numext::isnan(initial_values[i]))) {
          std::cout << "Error, data change is not visible in the reinterpreted buffer." << std::endl;
          std::cout << "  initialAcc[" << i << "] = " << int(initialAcc[i]) << " vs " << int(initial_values[i]) << std::endl;
        }
      }
    }
    
    auto reint = initial.template reinterpret<FinalType>(cl::sycl::range<1>(final_count));
    cl::sycl::buffer<FinalType> final{cl::sycl::range<1>(final_count)};
    {
      // read data using the reinterpreted buffer and verify them
      q.submit([&](cl::sycl::handler& cgh) {
        auto reintAcc =
            reint.template get_access<cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer>(cgh);
        auto finalAcc = final.template get_access<cl::sycl::access::mode::write,
                                                  cl::sycl::access::target::global_buffer>(cgh);
        cgh.copy(reintAcc, finalAcc);
      });
      q.wait_and_throw();
    }
    {
      auto finalAcc = final.template get_access<cl::sycl::access::mode::read>();
      for (size_t i = 0; i < final_count; ++i) {
        if (finalAcc[i] != final_values[i] && !(Eigen::numext::isnan(finalAcc[i]) && Eigen::numext::isnan(final_values[i]))) {
          std::cout << "Error, data change is not visible in the reinterpreted buffer." << std::endl;
          std::cout << "  reintAcc[" << i << "] = " << int(finalAcc[i]) << " vs " << int(final_values[i]) << std::endl;
        }
      }
    }
  }
}

template<typename InitialType, typename FinalType, int NumBytes = 10 * sizeof(InitialType) * sizeof(FinalType)>
void sycl_test_fill(const cl::sycl::device& d){
  constexpr int initial_count = NumBytes / sizeof(InitialType);
  constexpr int final_count = NumBytes / sizeof(FinalType);

  const FinalType value = Eigen::internal::random<FinalType>();
  int offset = 3;
  int size = ((initial_count - offset) * sizeof(InitialType))/sizeof(FinalType);

  InitialType initial_values[initial_count];
  std::fill_n(initial_values, initial_count, 0);
  FinalType final_values[final_count];
  std::fill_n(final_values, size, value);
  memcpy(initial_values+offset, final_values, size * sizeof(FinalType));

  cl::sycl::buffer<InitialType> buffer {cl::sycl::range<1>(initial_count)};
  {
    auto acc = buffer.template get_access<cl::sycl::access::mode::discard_write>();
    for (int i=0; i<initial_count; ++i) {
      acc[i] = initial_values[i];
    } 
  }
  
  // Device
  {
    cl::sycl::queue q {d};
    {
      // setup test data using the initial buffer accessor
      const int subbuf_size = (size * sizeof(FinalType)) / sizeof(InitialType);
      auto subbuf = cl::sycl::buffer<InitialType>(buffer, cl::sycl::id<1>(offset), cl::sycl::range<1>(subbuf_size));
      std::cout << "buffer size/count: " << buffer.get_size() << ", " << buffer.get_count() << std::endl;
      std::cout << "subbuf size/count: " << subbuf.get_size() << ", " << subbuf.get_count() << std::endl;
      auto reint = subbuf.template reinterpret<FinalType>(cl::sycl::range<1>(size));
      std::cout << "reint size/count:  " << reint.get_size() << ", " << reint.get_count() << std::endl;
      q.submit([&](cl::sycl::handler& cgh) {
        auto acc = reint.template get_access<cl::sycl::access::mode::discard_write,
                                             cl::sycl::access::target::global_buffer>(cgh);
        std::cout << "acc size/count:  " << acc.get_size() << ", " << acc.get_count() << std::endl;
        cgh.fill(acc, value);
      });
      q.wait_and_throw();
    }
    {
      auto acc = buffer.template get_access<cl::sycl::access::mode::read>();
      for (size_t i = 0; i < initial_count; ++i) {
        if (acc[i] != initial_values[i] && !(Eigen::numext::isnan(acc[i]) && Eigen::numext::isnan(initial_values[i]))) {
          std::cout << "Error, data change is not visible in the original buffer." << std::endl;
          std::cout << "  acc[" << i << "] = " << int(acc[i]) << " vs " << int(initial_values[i]) << std::endl;
        }
      }
    }
  }
}

EIGEN_DECLARE_TEST(cxx11_tensor_device_sycl) {
  for (const auto& device : Eigen::get_sycl_supported_devices()) {
    sycl_test_fill<uint8_t, uint32_t>(device);
    // CALL_SUBTEST(sycl_device_test_per_device<int8_t>(device));
    CALL_SUBTEST(sycl_device_test_per_device<uint8_t>(device));
    // CALL_SUBTEST(sycl_device_test_per_device<int8_t>(device));
    // CALL_SUBTEST(sycl_device_test_per_device<OffByOneScalar<int>>(device));
    // sycl_test_reinterpret<float, double>(device);
  }
}
