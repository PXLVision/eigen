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

void basic_tuple_test() {
  // Construction.
  Eigen::Tuple<> tuple0 {};
  Eigen::Tuple<int> tuple1 {1};
  Eigen::Tuple<int, float> tuple2 {3, 5.0f};
  Eigen::Tuple<int, float, double> tuple3 {7, 11.0f, 13.0};
  // Default construction.
  Eigen::Tuple<> tuple0default;
  Eigen::Tuple<int> tuple1default;
  Eigen::Tuple<int, float> tuple2default;
  Eigen::Tuple<int, float, double> tuple3default;
  
  // Assignment.
  Eigen::Tuple<> tuple0b = tuple0;
  decltype(tuple1) tuple1b = tuple1;
  decltype(tuple2) tuple2b = tuple2;
  decltype(tuple3) tuple3b = tuple3;
  
  // get.
  VERIFY_IS_EQUAL(Eigen::get<0>(tuple3), 7);
  VERIFY_IS_EQUAL(Eigen::get<1>(tuple3), 11.0f);
  VERIFY_IS_EQUAL(Eigen::get<2>(tuple3), 13.0);
  
  // tuple_size.
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(tuple0)>::value, 0);
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(tuple1)>::value, 1);
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(tuple2)>::value, 2);
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(tuple3)>::value, 3);
  
  // tuple_cat.
  auto tuple2cat3 = Eigen::tuple_cat(tuple2, tuple3);
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(tuple2cat3)>::value, 5);
  VERIFY_IS_EQUAL(Eigen::get<1>(tuple2cat3), 5.0f);
  VERIFY_IS_EQUAL(Eigen::get<3>(tuple2cat3), 11.0f);
  auto tuple3cat0 = Eigen::tuple_cat(tuple3, tuple0);
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(tuple3cat0)>::value, 3);
  auto singlecat = Eigen::tuple_cat(tuple3);
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(singlecat)>::value, 3);
  auto emptycat = Eigen::tuple_cat();
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(emptycat)>::value, 0);
  auto tuple0cat1cat2cat3 = Eigen::tuple_cat(tuple0, tuple1, tuple2, tuple3);
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(tuple0cat1cat2cat3)>::value, 6);
  
  // make_tuple.
  double tmp = 20;
  auto tuple_make = Eigen::make_tuple(int(10), tmp, float(20.0f), tuple0);
  VERIFY( (std::is_same<decltype(tuple_make), Eigen::Tuple<int, double, float, Eigen::Tuple<> > >::value) );
  VERIFY_IS_EQUAL(Eigen::get<1>(tuple_make), tmp);
  
  // forward_as_tuple.
  auto tuple_forward = Eigen::forward_as_tuple(int(10), tmp, float(20.0f), tuple0);
  VERIFY( (std::is_same<decltype(tuple_forward), Eigen::Tuple<int, double&, float, Eigen::Tuple<>& > >::value) );
  VERIFY_IS_EQUAL(Eigen::get<1>(tuple_forward), tmp);
  
  // tie.
  auto tuple_tie = Eigen::tie(tuple0, tuple1, tuple2, tuple3);
  VERIFY( (std::is_same<decltype(tuple_tie), 
                        Eigen::Tuple< decltype(tuple0)&,
                                      decltype(tuple1)&,
                                      decltype(tuple2)&,
                                      decltype(tuple3)&> >::value) );
  VERIFY_IS_EQUAL( (Eigen::get<1>(Eigen::get<2>(tuple_tie))), 5.0 );
  // Modify value and ensure tuple2 is updated.
  Eigen::get<1>(Eigen::get<2>(tuple_tie)) = 10.0;
  VERIFY_IS_EQUAL( (Eigen::get<1>(tuple2)), 10.0 );
}

void eigen_tuple_test() {
  Eigen::Tuple<Eigen::Matrix3d, Eigen::MatrixXd> tuple;
  Eigen::get<0>(tuple).setRandom();
  Eigen::get<1>(tuple).setRandom(10, 10);
  
  auto tuple_tie = Eigen::tie(Eigen::get<0>(tuple), Eigen::get<1>(tuple));
  Eigen::get<1>(tuple_tie).setIdentity();
  VERIFY(Eigen::get<1>(tuple).isIdentity());
}

class NoDefault {
 public:
  NoDefault() = delete;
  NoDefault(double y) : x(y) {}
  double x;
};

EIGEN_DECLARE_TEST(tuple)
{
  CALL_SUBTEST(basic_tuple_test());
  CALL_SUBTEST(eigen_tuple_test());
}
