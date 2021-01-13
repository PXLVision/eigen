// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2021 Everton Constantino (everton.constantino@hotmail.com)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATRIX_PRODUCT_NEON_H
#define EIGEN_MATRIX_PRODUCT_NEON_H
namespace Eigen {

namespace internal {

template<typename ResScalar, typename AccScalar, typename LhsScalar, typename RhsScalar, typename Index, typename DataMapper>
EIGEN_STRONG_INLINE void gemm(const DataMapper& res, const Scalar* blockA, const Scalar* blockB,
          Index rows, Index depth, Index cols, Scalar alpha, Index strideA, Index strideB, Index offsetA, Index offsetB, const int& accRows, const int& accCols)
{
      using AccPacket = packet_traits<AccScalar>::type;
      using LhsPacket = packet_traits<LhsScalar>::type;
      using RhsPacket = packet_traits<RhsScalar>::type;

      const Index remaining_rows = rows % accCols;
      const Index remaining_cols = cols % accRows;

      if( strideA == -1 ) strideA = depth;
      if( strideB == -1 ) strideB = depth;
}

template<typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<float, float, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
{
  void operator()(const DataMapper& res, const float* blockA, const float* blockB,
                  Index rows, Index depth, Index cols, float alpha,
                  Index strideA=-1, Index strideB=-1, Index offsetA=0, Index offsetB=0);
};

template<typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
void gebp_kernel<float, float, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
  ::operator()(const DataMapper& res, const float* blockA, const float* blockB,
               Index rows, Index depth, Index cols, float alpha,
               Index strideA, Index strideB, Index offsetA, Index offsetB)
  {
    gemm<float, float, float, Index, DataMapper, 1, 1>(res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB, accRows, accCols);
  }
} // end namespace internal

} // end namespace Eigen
#endif // EIGEN_MATRIX_PRODUCT_NEON_H