// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>
// Copyright (C) 2019 Rasmus Munk Larsen <rmlarsen@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_GENERIC_TYPE_CASTING_H
#define EIGEN_GENERIC_TYPE_CASTING_H

namespace Eigen {

namespace internal {

template<>
struct scalar_cast_op<float, Eigen::bfloat16> {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_cast_op)
  typedef Eigen::bfloat16 result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::bfloat16 operator() (const float& a) const {
    return Eigen::bfloat16(a);
  }
};

template<>
struct functor_traits<scalar_cast_op<float, Eigen::bfloat16> >
{ enum { Cost = NumTraits<float>::AddCost, PacketAccess = false }; };


template<>
struct scalar_cast_op<int, Eigen::bfloat16> {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_cast_op)
  typedef Eigen::bfloat16 result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::bfloat16 operator() (const int& a) const {
    return Eigen::bfloat16(static_cast<float>(a));
  }
};

template<>
struct functor_traits<scalar_cast_op<int, Eigen::bfloat16> >
{ enum { Cost = NumTraits<float>::AddCost, PacketAccess = false }; };


template<>
struct scalar_cast_op<Eigen::bfloat16, float> {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_cast_op)
  typedef float result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float operator() (const Eigen::bfloat16& a) const {
    return static_cast<float>(a);
  }
};

template<>
struct functor_traits<scalar_cast_op<Eigen::bfloat16, float> >
{ enum { Cost = NumTraits<float>::AddCost, PacketAccess = false }; };


}
}

#endif  // EIGEN_GENERIC_TYPE_CASTING_H
