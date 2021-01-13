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
#ifdef __DEBUG__
#include <iostream>
#endif

template<typename Scalar, typename Packet, typename Index, bool IsLhs = true>
class PackMap
{
  const int packetSize = packet_traits<Scalar>::size;
  const Scalar *packed_block;
  const Scalar *residue_block;
  Index packed_stride;
  Index rows, cols;
  Index offset, stride;
  Scalar *cur;
public:
  PackMap(const Scalar *packed_block, const Scalar *residue_block, Index rows, Index cols, Index offset, Index stride) : packed_block(packed_block), residue_block(residue_block), rows(rows), cols(cols), offset(offset), stride(stride)
  {
    if(IsLhs)
      packed_stride = (rows / packetSize) * packetSize;
    else
      packed_stride = (cols / packetSize) * packetSize;
  };

  PackMap(const Scalar *packed_block, Index rows, Index cols, Index offset, Index stride) : packed_block(packed_block), rows(rows), cols(cols)
  {
    if(IsLhs)
    {
      packed_stride = (rows / packetSize) * packetSize;
      residue_block = packed_block + packed_stride*cols*packetSize;
    }
    else {
      packed_stride = (cols / packetSize) * packetSize;
      residue_block = packed_block + packed_stride*rows*packetSize;
    }

  };

  EIGEN_STRONG_INLINE Index get_packed_size()
  {
    return packed_stride;
  };

  EIGEN_STRONG_INLINE const Scalar* get_packed_at(Index at)
  {
    return IsLhs ? packed_block + packed_stride*at : packed_block + packed_stride*at*;
  };

  EIGEN_STRONG_INLINE const Scalar* get_residue_at(Index at)
  {
    return residue_block + stride*at;
  };
};

template<typename ResScalar, typename AccScalar, typename LhsScalar, typename RhsScalar, typename Index, typename DataMapper>
EIGEN_STRONG_INLINE void gemm(const DataMapper& res, const LhsScalar* blockA, const RhsScalar* blockB,
          Index rows, Index depth, Index cols, ResScalar alpha, Index strideA, Index strideB, Index offsetA, Index offsetB)
{
  using AccPacket = typename packet_traits<AccScalar>::type;
  using LhsPacket = typename packet_traits<LhsScalar>::type;
  using RhsPacket = typename packet_traits<RhsScalar>::type;
  using ResPacket = typename packet_traits<ResScalar>::type;

  ResPacket pAlpha = pset1<ResPacket>(alpha);

#ifdef __DEBUG__
  std::cout << "blockA" << std::endl;
  for(auto i = 0; i < rows*depth; i++)
  {
    if(i % 4 == 0 && i > 0)
      std::cout << std::endl;
    std::cout << blockA[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "blockB" << std::endl;
  for(auto i = 0; i < depth*cols; i++)
  {
    if(i % 4 == 0 && i > 0)
      std::cout << std::endl;
    std::cout << blockB[i] << " ";
  }
  std::cout << std::endl;
#endif

  if( strideA == -1 ) strideA = depth;
  if( strideB == -1 ) strideB = depth;

  int accLhsProgress = 4;
  int accRhsProgress = 4;

  PackMap<LhsScalar, LhsPacket, Index> lhsMap(blockA, rows, depth, offsetA, strideA);
  PackMap<RhsScalar, RhsPacket, Index, false> rhsMap(blockB, depth, cols, offsetB, strideB);
  for(auto col = 0; col <= cols; col+=accRhsProgress)
  {
    for(auto k = 0; k <= depth; k++)
    {
      const LhsScalar *lhs_ptr = lhsMap.get_packed_at(col);
      const RhsScalar *rhs_ptr = rhsMap.get_packed_at(col);
      AccPacket acc;
      RhsPacket prhs = pload<RhsPacket>(rhs_ptr);
      PacketBlock<RhsPacket, 4> pbrhs;
      pbrhs.packet[0] = pset1<RhsPacket>(prhs[0]);
      pbrhs.packet[1] = pset1<RhsPacket>(prhs[1]);
      pbrhs.packet[2] = pset1<RhsPacket>(prhs[2]);
      pbrhs.packet[3] = pset1<RhsPacket>(prhs[3]);
      for(auto row = 0; row < lhsMap.get_packed_size(); row+=accLhsProgress)
      {
        LhsPacket plhs = pload<LhsPacket>(lhs_ptr);
        acc = plhs*pbrhs.packet[0]; // C(1,c)C(2,c)C(3,c)C(4,c)
      }
      lhs_ptr = lhsMap.get_residue_at()
    }
  }
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
    gemm<float, float, float, float, Index, DataMapper>(res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB);
  }
} // end namespace internal

} // end namespace Eigen
#endif // EIGEN_MATRIX_PRODUCT_NEON_H