// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2021 The Eigen Team
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <vector>
#include <Eigen/Core>

struct MyPodType {
  double x;
  int y;
  float z;
};

// Plain-old-data serialization.
void test_pod_type() {
  MyPodType initial = {1.3, 17, 1.9f};
  MyPodType clone = {-1, -1, -1};
  
  Eigen::Serializer<MyPodType> serializer;
  
  // Determine required size.
  size_t buffer_size = serializer.size(initial);
  VERIFY_IS_EQUAL(buffer_size, sizeof(MyPodType));
  
  // Serialize.
  std::vector<uint8_t> buffer(buffer_size);
  uint8_t* dest = serializer.serialize(buffer.data(), initial);
  VERIFY_IS_EQUAL(dest - buffer.data(), buffer_size);
  
  // Deserialize.
  uint8_t* src = serializer.deserialize(buffer.data(), clone);
  VERIFY_IS_EQUAL(src - buffer.data(), buffer_size);
  VERIFY_IS_EQUAL(clone.x, initial.x);
  VERIFY_IS_EQUAL(clone.y, initial.y);
  VERIFY_IS_EQUAL(clone.z, initial.z);
}

// Matrix, Vector, Array
template<typename T>
void test_eigen_type(const T& type) {
  const Index rows = type.rows();
  const Index cols = type.cols();
  
  const T initial = T::Random(rows, cols);
  
  // Serialize.
  Eigen::Serializer<T> serializer;
  size_t buffer_size = serializer.size(initial);
  std::vector<uint8_t> buffer(buffer_size);
  uint8_t* dest = serializer.serialize(buffer.data(), initial);
  VERIFY_IS_EQUAL(dest - buffer.data(), buffer_size);
  
  // Deserialize.
  T clone;
  uint8_t* src = serializer.deserialize(buffer.data(), clone);
  VERIFY_IS_EQUAL(src - buffer.data(), buffer_size);
  VERIFY_IS_CWISE_EQUAL(clone, initial);
}

template<typename T1, typename T2>
void verify_cwise(const T1& actual, const T2& expected) {
  VERIFY_IS_CWISE_EQUAL(actual, expected);
}

// Helper for testing a collection of dense types.
template<size_t... Indices, typename... DenseTypes>
void test_dense_types(Eigen::internal::index_sequence<Indices...>,
                      const DenseTypes&... types) {
  
  // Make random inputs.
  auto inputs = Eigen::make_tuple(types...);
  // Trick to apply function to each tuple element,
  // creates an initializer list of 0s.
  auto randomize = { (Eigen::tuple_get<Indices>(inputs).setRandom(), 0)... };

  // Allocate buffer and serialize.
  size_t buffer_size = Eigen::serialize_size(Eigen::tuple_get<Indices>(inputs)...);
  std::vector<uint8_t> buffer(buffer_size);
  Eigen::serialize(buffer.data(), Eigen::tuple_get<Indices>(inputs)...);
  
  // Clone everything.
  auto clones = Eigen::make_tuple(types...);
  Eigen::deserialize(buffer.data(), Eigen::tuple_get<Indices>(clones)...);
  
  // Verify they equal.
  auto verify = { (verify_cwise(Eigen::tuple_get<Indices>(clones),
                                Eigen::tuple_get<Indices>(inputs)), 0)... };
}

// Check a collection of dense types.
template<typename... DenseTypes>
void test_dense_types(const DenseTypes&... types) {
  test_dense_types(
      Eigen::internal::make_index_sequence<sizeof...(DenseTypes)>{},
      types...);
}

EIGEN_DECLARE_TEST(serializer)
{
  CALL_SUBTEST( test_pod_type() );

  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST( test_eigen_type(Eigen::Array33f()) );
    CALL_SUBTEST( test_eigen_type(Eigen::ArrayXd(10)) );
    CALL_SUBTEST( test_eigen_type(Eigen::Vector3f()) );
    CALL_SUBTEST( test_eigen_type(Eigen::Matrix4d()) );
    CALL_SUBTEST( test_eigen_type(Eigen::MatrixXd(15, 17)) );
    
    CALL_SUBTEST( test_dense_types( Eigen::Array33f(),
                                    Eigen::ArrayXd(10),
                                    Eigen::Vector3f(),
                                    Eigen::Matrix4d(),
                                    Eigen::MatrixXd(15, 17)) );
  }
}
