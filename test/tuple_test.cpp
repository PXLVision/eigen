// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2021 The Eigen Team
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/Core>
#include <Eigen/src/Core/arch/GPU/Tuple.h>

using namespace Eigen::internal;
using Eigen::internal::tuple_impl::tuple;
  
void basic_tuple_test() {  
  // Construction.
  tuple<> tuple0 {};
  tuple<int> tuple1 {1};
  tuple<int, float> tuple2 {3, 5.0f};
  tuple<int, float, double> tuple3 {7, 11.0f, 13.0};
  // Default construction.
  tuple<> tuple0default;
  EIGEN_UNUSED_VARIABLE(tuple0default)
  tuple<int> tuple1default;
  EIGEN_UNUSED_VARIABLE(tuple1default)
  tuple<int, float> tuple2default;
  EIGEN_UNUSED_VARIABLE(tuple2default)
  tuple<int, float, double> tuple3default;
  EIGEN_UNUSED_VARIABLE(tuple3default)
  
  // Assignment.
  tuple<> tuple0b = tuple0;
  EIGEN_UNUSED_VARIABLE(tuple0b)
  decltype(tuple1) tuple1b = tuple1;
  EIGEN_UNUSED_VARIABLE(tuple1b)
  decltype(tuple2) tuple2b = tuple2;
  EIGEN_UNUSED_VARIABLE(tuple2b)
  decltype(tuple3) tuple3b = tuple3;
  EIGEN_UNUSED_VARIABLE(tuple3b)
  
  // get.
  VERIFY_IS_EQUAL(tuple_impl::get<0>(tuple3), 7);
  VERIFY_IS_EQUAL(tuple_impl::get<1>(tuple3), 11.0f);
  VERIFY_IS_EQUAL(tuple_impl::get<2>(tuple3), 13.0);
  
  // tuple_impl::tuple_size.
  VERIFY_IS_EQUAL(tuple_impl::tuple_size<decltype(tuple0)>::value, 0);
  VERIFY_IS_EQUAL(tuple_impl::tuple_size<decltype(tuple1)>::value, 1);
  VERIFY_IS_EQUAL(tuple_impl::tuple_size<decltype(tuple2)>::value, 2);
  VERIFY_IS_EQUAL(tuple_impl::tuple_size<decltype(tuple3)>::value, 3);
  
  // tuple_impl::tuple_cat.
  auto tuple2cat3 = tuple_impl::tuple_cat(tuple2, tuple3);
  VERIFY_IS_EQUAL(tuple_impl::tuple_size<decltype(tuple2cat3)>::value, 5);
  VERIFY_IS_EQUAL(tuple_impl::get<1>(tuple2cat3), 5.0f);
  VERIFY_IS_EQUAL(tuple_impl::get<3>(tuple2cat3), 11.0f);
  auto tuple3cat0 = tuple_impl::tuple_cat(tuple3, tuple0);
  VERIFY_IS_EQUAL(tuple_impl::tuple_size<decltype(tuple3cat0)>::value, 3);
  auto singlecat = tuple_impl::tuple_cat(tuple3);
  VERIFY_IS_EQUAL(tuple_impl::tuple_size<decltype(singlecat)>::value, 3);
  auto emptycat = tuple_impl::tuple_cat();
  VERIFY_IS_EQUAL(tuple_impl::tuple_size<decltype(emptycat)>::value, 0);
  auto tuple0cat1cat2cat3 = tuple_impl::tuple_cat(tuple0, tuple1, tuple2, tuple3);
  VERIFY_IS_EQUAL(tuple_impl::tuple_size<decltype(tuple0cat1cat2cat3)>::value, 6);
  
  // make_tuple.
  double tmp = 20;
  auto tuple_make = tuple_impl::make_tuple(int(10), tmp, float(20.0f), tuple0);
  VERIFY( (std::is_same<decltype(tuple_make), tuple<int, double, float, tuple<> > >::value) );
  VERIFY_IS_EQUAL(tuple_impl::get<1>(tuple_make), tmp);
  
  // forward_as_tuple.
  auto tuple_forward = tuple_impl::forward_as_tuple(int(10), tmp, float(20.0f), tuple0);
  VERIFY( (std::is_same<decltype(tuple_forward), tuple<int, double&, float, tuple<>& > >::value) );
  VERIFY_IS_EQUAL(tuple_impl::get<1>(tuple_forward), tmp);
  
  // tie.
  auto tuple_tie = tuple_impl::tie(tuple0, tuple1, tuple2, tuple3);
  VERIFY( (std::is_same<decltype(tuple_tie), 
                        tuple< decltype(tuple0)&,
                                      decltype(tuple1)&,
                                      decltype(tuple2)&,
                                      decltype(tuple3)&> >::value) );
  VERIFY_IS_EQUAL( (tuple_impl::get<1>(tuple_impl::get<2>(tuple_tie))), 5.0 );
  // Modify value and ensure tuple2 is updated.
  tuple_impl::get<1>(tuple_impl::get<2>(tuple_tie)) = 10.0;
  VERIFY_IS_EQUAL( (tuple_impl::get<1>(tuple2)), 10.0 );
}

void eigen_tuple_test() {
  tuple<Eigen::Matrix3d, Eigen::MatrixXd> tuple;
  tuple_impl::get<0>(tuple).setRandom();
  tuple_impl::get<1>(tuple).setRandom(10, 10);
  
  auto tuple_tie = tuple_impl::tie(tuple_impl::get<0>(tuple), tuple_impl::get<1>(tuple));
  tuple_impl::get<1>(tuple_tie).setIdentity();
  VERIFY(tuple_impl::get<1>(tuple).isIdentity());
}

EIGEN_DECLARE_TEST(tuple)
{
  CALL_SUBTEST(basic_tuple_test());
  CALL_SUBTEST(eigen_tuple_test());
}
