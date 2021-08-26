#include "main.h"

#include "Tuple.h"

void tuple_test() {
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
  
  // tuple_get.
  VERIFY_IS_EQUAL(Eigen::tuple_get<0>(tuple3), 7);
  VERIFY_IS_EQUAL(Eigen::tuple_get<1>(tuple3), 11.0f);
  VERIFY_IS_EQUAL(Eigen::tuple_get<2>(tuple3), 13.0);
  
  // tuple_size.
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(tuple0)>::value, 0);
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(tuple1)>::value, 1);
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(tuple2)>::value, 2);
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(tuple3)>::value, 3);
  
  // tuple_cat.
  auto tuple2cat3 = Eigen::tuple_cat(tuple2, tuple3);
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(tuple2cat3)>::value, 5);
  VERIFY_IS_EQUAL(Eigen::tuple_get<1>(tuple2cat3), 5.0f);
  VERIFY_IS_EQUAL(Eigen::tuple_get<3>(tuple2cat3), 11.0f);
  
  auto tuple3cat0 = Eigen::tuple_cat(tuple3, tuple0);
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(tuple3cat0)>::value, 3);
  auto singlecat = Eigen::tuple_cat(tuple3);
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(singlecat)>::value, 3);
  auto emptycat = Eigen::tuple_cat();
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(emptycat)>::value, 0);
  auto tuple0cat1cat2cat3 = Eigen::tuple_cat(tuple0, tuple1, tuple2, tuple3);
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(tuple0cat1cat2cat3)>::value, 6);
}

void std_tuple_test() {
  // Construction.
  std::tuple<> tuple0 {};
  std::tuple<int> tuple1 {1};
  std::tuple<int, float> tuple2 {3, 5.0f};
  std::tuple<int, float, double> tuple3 {7, 11.0f, 13.0};
  
  // Assignment.
  std::tuple<> tuple0b = tuple0;
  decltype(tuple1) tuple1b = tuple1;
  decltype(tuple2) tuple2b = tuple2;
  decltype(tuple3) tuple3b = tuple3;
  
  // tuple_get.
  VERIFY_IS_EQUAL(Eigen::tuple_get<0>(tuple3), 7);
  VERIFY_IS_EQUAL(Eigen::tuple_get<1>(tuple3), 11.0f);
  VERIFY_IS_EQUAL(Eigen::tuple_get<2>(tuple3), 13.0);
  
  // tuple_size.
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(tuple0)>::value, 0);
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(tuple1)>::value, 1);
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(tuple2)>::value, 2);
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(tuple3)>::value, 3);
  
  // tuple_cat.
  auto tuple2cat3 = Eigen::tuple_cat(tuple2, tuple3);
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(tuple2cat3)>::value, 5);
  VERIFY_IS_EQUAL(Eigen::tuple_get<1>(tuple2cat3), 5.0f);
  VERIFY_IS_EQUAL(Eigen::tuple_get<3>(tuple2cat3), 11.0f);
  
  auto tuple3cat0 = Eigen::tuple_cat(tuple3, tuple0);
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(tuple3cat0)>::value, 3);
  auto singlecat = Eigen::tuple_cat(tuple3);
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(singlecat)>::value, 3);
  auto emptycat = Eigen::tuple_cat();
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(emptycat)>::value, 0);
  auto tuple0cat1cat2cat3 = Eigen::tuple_cat(tuple0, tuple1, tuple2, tuple3);
  VERIFY_IS_EQUAL(Eigen::tuple_size<decltype(tuple0cat1cat2cat3)>::value, 6);
}

EIGEN_DECLARE_TEST(tuple)
{
  CALL_SUBTEST(tuple_test());
  CALL_SUBTEST(std_tuple_test());
}
