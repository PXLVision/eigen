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
  Index residue_size;
  Index rows, cols;
  Index offset, stride;
  Scalar *cur;
public:
  PackMap(const Scalar *packed_block, const Scalar *residue_block, Index rows, Index cols, Index offset, Index stride) : packed_block(packed_block), residue_block(residue_block), rows(rows), cols(cols), offset(offset), stride(stride)
  {
    if(IsLhs)
    {
      packed_stride = (rows / packetSize) * packetSize;
      residue_size = rows % packetSize;
    }
    else {
      packed_stride = (cols / packetSize) * packetSize;
      residue_size = cols % packetSize;
    }
  };

  PackMap(const Scalar *packed_block, Index rows, Index cols, Index offset, Index stride) : packed_block(packed_block), rows(rows), cols(cols)
  {
    if(IsLhs)
    {
      packed_stride = (rows / packetSize) * packetSize;
      residue_block = packed_block + packed_stride*cols;
      residue_size = rows % packetSize;
    }
    else {
      packed_stride = (cols / packetSize) * packetSize;
      residue_block = packed_block + packed_stride*rows;
      residue_size = cols % packetSize;
    }

  };

  EIGEN_STRONG_INLINE Index get_packed_size()
  {
    return packed_stride;
  };

  EIGEN_STRONG_INLINE Index get_residue_size()
  {
    return residue_size;
  };

  EIGEN_STRONG_INLINE const Scalar* get_packed_at(Index at)
  {
    return IsLhs ? packed_block + at : packed_block + at*packetSize*rows;
  };

  EIGEN_STRONG_INLINE const Scalar* get_residue_at(Index at)
  {
    return residue_block + stride*at;
  };
};

template<typename ResScalar, typename AccScalar, typename LhsScalar, typename RhsScalar, typename Index, typename DataMapper, int BLOCK_SIZE, int N>
struct KernelRowLoop
{
  KernelRowLoop<ResScalar, AccScalar, LhsScalar, RhsScalar, Index, DataMapper, BLOCK_SIZE, N-1> kernelRowLoop;
  using AccPacket = typename packet_traits<AccScalar>::type;
  using LhsPacket = typename packet_traits<LhsScalar>::type;
  using RhsPacket = typename packet_traits<RhsScalar>::type;
  using ResPacket = typename packet_traits<ResScalar>::type;

  const LhsScalar *lhs_ptr;
  PacketBlock<AccPacket, BLOCK_SIZE> acc;
  int _accLhsProgress;
  Index _row,_col;

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void preamble(const DataMapper& res, PackMap<LhsScalar, LhsPacket, Index>& lhsMap, Index row, Index col, int accLhsProgress)
  {
    kernelRowLoop.preamble(res, lhsMap, row, col, accLhsProgress);
    _accLhsProgress = accLhsProgress;
    _row = row;
    _col = col;
    lhs_ptr = lhsMap.get_packed_at(row + N*accLhsProgress);

    acc.packet[0] = pset1<AccPacket>(0);
    acc.packet[1] = pset1<AccPacket>(0);
    acc.packet[2] = pset1<AccPacket>(0);
    acc.packet[3] = pset1<AccPacket>(0);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void operator()(const PacketBlock<RhsPacket, 4>& pbrhs, Index offset)
  {
    kernelRowLoop(pbrhs, offset);

    LhsPacket plhs = pload<LhsPacket>(lhs_ptr);
    acc.packet[0] += plhs*pbrhs.packet[0];
    acc.packet[1] += plhs*pbrhs.packet[1];
    acc.packet[2] += plhs*pbrhs.packet[2];
    acc.packet[3] += plhs*pbrhs.packet[3];

    lhs_ptr += offset;
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void operator()(const RhsPacket& prhs, Index offset)
  {
    kernelRowLoop(prhs, offset);

    LhsPacket plhs = pload<LhsPacket>(lhs_ptr);
    acc.packet[0] += plhs*prhs;

    lhs_ptr += offset;
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void postamble(const DataMapper& res)
  {
    kernelRowLoop.postamble(res);
    // Don't forget to load res and multiply by alpha
    res.template storePacketBlock<ResPacket, BLOCK_SIZE>(_row + N*_accLhsProgress, _col, acc);
  }
};

template<typename ResScalar, typename AccScalar, typename LhsScalar, typename RhsScalar, typename Index, typename DataMapper, int BLOCK_SIZE>
struct KernelRowLoop<ResScalar, AccScalar, LhsScalar, RhsScalar, Index, DataMapper, BLOCK_SIZE, -1>
{
  using LhsPacket = typename packet_traits<LhsScalar>::type;
  using RhsPacket = typename packet_traits<RhsScalar>::type;

  EIGEN_STRONG_INLINE void preamble(const DataMapper&, PackMap<LhsScalar, LhsPacket, Index>&, Index, Index, int) {}
  EIGEN_STRONG_INLINE void operator()(const PacketBlock<RhsPacket, 4>&, Index){}
  EIGEN_STRONG_INLINE void operator()(const RhsPacket&, Index){}
  EIGEN_STRONG_INLINE void postamble(const DataMapper&) {}
};

template<typename ResScalar, typename AccScalar, typename LhsScalar, typename RhsScalar, typename Index, typename DataMapper>
EIGEN_STRONG_INLINE void gemm(const DataMapper& res, const LhsScalar* blockA, const RhsScalar* blockB,
          Index rows, Index depth, Index cols, ResScalar alpha, Index strideA, Index strideB, Index offsetA, Index offsetB)
{
  using AccPacket = typename packet_traits<AccScalar>::type;
  using LhsPacket = typename packet_traits<LhsScalar>::type;
  using RhsPacket = typename packet_traits<RhsScalar>::type;
  using ResPacket = typename packet_traits<ResScalar>::type;
  using LinearMapper = typename DataMapper::LinearMapper;

  if( strideA == -1 ) strideA = depth;
  if( strideB == -1 ) strideB = depth;

  ResPacket pAlpha = pset1<ResPacket>(alpha);

  int accLhsProgress = 4;
  int accRhsProgress = 4;

  PackMap<LhsScalar, LhsPacket, Index> lhsMap(blockA, rows, depth, offsetA, strideA);
  PackMap<RhsScalar, RhsPacket, Index, false> rhsMap(blockB, depth, cols, offsetB, strideB);
  auto col = 0;
  for(; col + 2*accRhsProgress <= rhsMap.get_packed_size(); col+=2*accRhsProgress)
  {
    auto row = 0;
#define ROW_LOOP(K)                                                                               \
    for(; row + K*accLhsProgress <= lhsMap.get_packed_size(); row+=K*accLhsProgress)              \
    {                                                                                             \
      KernelRowLoop<ResScalar, AccScalar, LhsScalar, RhsScalar, Index, DataMapper, 4, (K-1)> kRL; \
      kRL.preamble(res, lhsMap, row, col, accLhsProgress);                                        \
                                                                                                  \
      const RhsScalar *rhs_ptr = rhsMap.get_packed_at(col/accRhsProgress);                        \
                                                                                                  \
      auto k = 0;                                                                                 \
      for(; k < depth; k++)                                                                       \
      {                                                                                           \
        RhsPacket prhs = pload<RhsPacket>(rhs_ptr);                                               \
        PacketBlock<RhsPacket, 4> pbrhs;                                                          \
        pbrhs.packet[0] = pset1<RhsPacket>(prhs[0]);                                              \
        pbrhs.packet[1] = pset1<RhsPacket>(prhs[1]);                                              \
        pbrhs.packet[2] = pset1<RhsPacket>(prhs[2]);                                              \
        pbrhs.packet[3] = pset1<RhsPacket>(prhs[3]);                                              \
                                                                                                  \
        kRL(pbrhs, (rows/accLhsProgress)*accLhsProgress);                                         \
                                                                                                  \
        rhs_ptr += accRhsProgress;                                                                \
      }                                                                                           \
                                                                                                  \
      kRL.postamble(res);                                                                         \
    }

    ROW_LOOP(6);
    ROW_LOOP(5);
    ROW_LOOP(4);
    ROW_LOOP(3);
    ROW_LOOP(2);
    ROW_LOOP(1);

    auto row_residue = 0;
    for(;row < rows; row++)
    {
      const LhsScalar *lhs_ptr = lhsMap.get_residue_at(row_residue);
      const RhsScalar *rhs_ptr = rhsMap.get_packed_at(col/accRhsProgress);
      PacketBlock<AccPacket, 1> acc;
      acc.packet[0] = pset1<AccPacket>(0);

      auto k = 0;
      for(; k < depth; k++)
      {
        RhsPacket prhs = pload<RhsPacket>(rhs_ptr);
        LhsPacket plhs = pset1<LhsPacket>(*lhs_ptr);

        acc.packet[0] += (*lhs_ptr)*prhs;

        lhs_ptr++;
        rhs_ptr += accRhsProgress;
      }

      res(row, col + 0) += acc.packet[0][0];
      res(row, col + 1) += acc.packet[0][1];
      res(row, col + 2) += acc.packet[0][2];
      res(row, col + 3) += acc.packet[0][3];
      row_residue++;
    }
  }
  for(; col + accRhsProgress <= rhsMap.get_packed_size(); col+=accRhsProgress)
  {
    auto row = 0;
#define ROW_LOOP(K)                                                                               \
    for(; row + K*accLhsProgress <= lhsMap.get_packed_size(); row+=K*accLhsProgress)              \
    {                                                                                             \
      KernelRowLoop<ResScalar, AccScalar, LhsScalar, RhsScalar, Index, DataMapper, 4, (K-1)> kRL; \
      kRL.preamble(res, lhsMap, row, col, accLhsProgress);                                        \
                                                                                                  \
      const RhsScalar *rhs_ptr = rhsMap.get_packed_at(col/accRhsProgress);                        \
                                                                                                  \
      auto k = 0;                                                                                 \
      for(; k < depth; k++)                                                                       \
      {                                                                                           \
        RhsPacket prhs = pload<RhsPacket>(rhs_ptr);                                               \
        PacketBlock<RhsPacket, 4> pbrhs;                                                          \
        pbrhs.packet[0] = pset1<RhsPacket>(prhs[0]);                                              \
        pbrhs.packet[1] = pset1<RhsPacket>(prhs[1]);                                              \
        pbrhs.packet[2] = pset1<RhsPacket>(prhs[2]);                                              \
        pbrhs.packet[3] = pset1<RhsPacket>(prhs[3]);                                              \
                                                                                                  \
        kRL(pbrhs, (rows/accLhsProgress)*accLhsProgress);                                         \
                                                                                                  \
        rhs_ptr += accRhsProgress;                                                                \
      }                                                                                           \
                                                                                                  \
      kRL.postamble(res);                                                                         \
    }

    ROW_LOOP(6);
    ROW_LOOP(5);
    ROW_LOOP(4);
    ROW_LOOP(3);
    ROW_LOOP(2);
    ROW_LOOP(1);

    auto row_residue = 0;
    for(;row < rows; row++)
    {
      const LhsScalar *lhs_ptr = lhsMap.get_residue_at(row_residue);
      const RhsScalar *rhs_ptr = rhsMap.get_packed_at(col/accRhsProgress);
      PacketBlock<AccPacket, 1> acc;
      acc.packet[0] = pset1<AccPacket>(0);

      auto k = 0;
      for(; k < depth; k++)
      {
        RhsPacket prhs = pload<RhsPacket>(rhs_ptr);
        LhsPacket plhs = pset1<LhsPacket>(*lhs_ptr);

        acc.packet[0] += (*lhs_ptr)*prhs;

        lhs_ptr++;
        rhs_ptr += accRhsProgress;
      }

      res(row, col + 0) += acc.packet[0][0];
      res(row, col + 1) += acc.packet[0][1];
      res(row, col + 2) += acc.packet[0][2];
      res(row, col + 3) += acc.packet[0][3];
      row_residue++;
    }
  }
  auto col_residue = 0;
  for(; col < cols; col++)
  {
    auto row = 0;
#undef ROW_LOOP
#define ROW_LOOP(K)                                                                               \
    for(; row + K*accLhsProgress <= lhsMap.get_packed_size(); row+=K*accLhsProgress)              \
    {                                                                                             \
      KernelRowLoop<ResScalar, AccScalar, LhsScalar, RhsScalar, Index, DataMapper, 1, (K-1)> kRL; \
      kRL.preamble(res, lhsMap, row, col, accLhsProgress);                                        \
                                                                                                  \
      const RhsScalar *rhs_ptr = rhsMap.get_residue_at(col_residue);                              \
                                                                                                  \
      auto k = 0;                                                                                 \
      for(; k < depth; k++)                                                                       \
      {                                                                                           \
        RhsPacket prhs = pset1<RhsPacket>(*rhs_ptr);                                              \
                                                                                                  \
        kRL(prhs, (rows/accLhsProgress)*accLhsProgress);                                          \
                                                                                                  \
        rhs_ptr++;                                                                                \
      }                                                                                           \
                                                                                                  \
      kRL.postamble(res);                                                                         \
    }

    ROW_LOOP(4);
    ROW_LOOP(3);
    ROW_LOOP(2);
    ROW_LOOP(1);

    auto row_residue = 0;
    for(;row < rows; row++)
    {
      const LhsScalar *lhs_ptr = lhsMap.get_residue_at(row_residue);
      const RhsScalar *rhs_ptr = rhsMap.get_residue_at(col_residue);
      AccScalar acc = 0;
      auto k = 0;
      for(; k < depth; k++)
      {
#ifdef __NDEBUG__
        std::cout << "(" << row << "," << k << "," << col << ")" << std::endl;
        std::cout << "lhs " << plhs[0] << " " << plhs[1] << " " << plhs[2] << " " << plhs[3] << std::endl;
        std::cout << "rhs " << prhs[0] << " " << prhs[1] << " " << prhs[2] << " " << prhs[3] << std::endl;
#endif
        acc += (*lhs_ptr)*(*rhs_ptr);

        lhs_ptr++;
        rhs_ptr++;
      }

      //r0.storePacket(0,r0.template loadPacket<ResPacket>(0) + acc.packet[0]);
      res(row, col) += acc;
      row_residue++;
    }
    col_residue++;
  }
};

template<typename ResScalar, typename AccScalar, typename LhsScalar, typename RhsScalar, typename Index, typename DataMapper>
EIGEN_STRONG_INLINE void gemm_old(const DataMapper& res, const LhsScalar* blockA, const RhsScalar* blockB,
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
  auto col = 0;
  for(; col < rhsMap.get_packed_size(); col+=accRhsProgress)
  {
    for(auto k = 0; k < depth; k++)
    {
      const LhsScalar *lhs_ptr = lhsMap.get_packed_at(k);
      const RhsScalar *rhs_ptr = rhsMap.get_packed_at(col/accRhsProgress) + k*accRhsProgress;
      PacketBlock<AccPacket, 4> acc;
      RhsPacket prhs = pload<RhsPacket>(rhs_ptr);
      PacketBlock<RhsPacket, 4> pbrhs;
      pbrhs.packet[0] = pset1<RhsPacket>(prhs[0]);
      pbrhs.packet[1] = pset1<RhsPacket>(prhs[1]);
      pbrhs.packet[2] = pset1<RhsPacket>(prhs[2]);
      pbrhs.packet[3] = pset1<RhsPacket>(prhs[3]);
      auto row = 0;
      using LinearMapper = typename DataMapper::LinearMapper;
      for(; row < lhsMap.get_packed_size(); row+=accLhsProgress)
      {
        LinearMapper r0 = res.getLinearMapper(row, col + 0);
        LinearMapper r1 = res.getLinearMapper(row, col + 1);
        LinearMapper r2 = res.getLinearMapper(row, col + 2);
        LinearMapper r3 = res.getLinearMapper(row, col + 3);

        LhsPacket plhs = pload<LhsPacket>(lhs_ptr);
#ifdef __NDEBUG__
        std::cout << "(" << row << "," << k << "," << col << ")" << std::endl;
        std::cout << "lhs " << plhs[0] << " " << plhs[1] << " " << plhs[2] << " " << plhs[3] << std::endl;
        std::cout << "rhs " << prhs[0] << " " << prhs[1] << " " << prhs[2] << " " << prhs[3] << std::endl;
#endif
        acc.packet[0] = plhs*pbrhs.packet[0];
        acc.packet[1] = plhs*pbrhs.packet[1];
        acc.packet[2] = plhs*pbrhs.packet[2];
        acc.packet[3] = plhs*pbrhs.packet[3];

        r0.storePacket(0,r0.template loadPacket<ResPacket>(0) + acc.packet[0]);
        r1.storePacket(0,r1.template loadPacket<ResPacket>(0) + acc.packet[1]);
        r2.storePacket(0,r2.template loadPacket<ResPacket>(0) + acc.packet[2]);
        r3.storePacket(0,r3.template loadPacket<ResPacket>(0) + acc.packet[3]);
        lhs_ptr += accLhsProgress;
      }
      auto residue = 0;
      for(;row < rows; row++)
      {
        LhsScalar lhs = *(lhsMap.get_residue_at(residue) + k);
#ifdef __NDEBUG__
        std::cout << "(" << row << "," << k << "," << col << ")" << std::endl;
        std::cout << "lhs " << lhs << " (" << prhs[0] << " " << prhs[1] << " " << prhs[2] << " " << prhs[3] << ")" << std::endl;
#endif
        res(row, col + 0) += lhs*prhs[0];
        res(row, col + 1) += lhs*prhs[1];
        res(row, col + 2) += lhs*prhs[2];
        res(row, col + 3) += lhs*prhs[3];
        residue++;
      }
    }
  }
  auto colResidue = 0;
  for(;col < cols; col++)
  {
    for(auto k = 0; k < depth; k++)
    {
      const LhsScalar *lhs_ptr = lhsMap.get_packed_at(k);
      const RhsScalar *rhs_ptr = rhsMap.get_residue_at(colResidue) + k;
      AccPacket acc;

      RhsPacket prhs = pset1<RhsPacket>(*rhs_ptr);


      auto row = 0;
      using LinearMapper = typename DataMapper::LinearMapper;
      for(; row < lhsMap.get_packed_size(); row+=accLhsProgress)
      {
        LinearMapper r0 = res.getLinearMapper(row, col + 0);

        LhsPacket plhs = pload<LhsPacket>(lhs_ptr);
#ifdef __DEBUG__
        std::cout << "(" << row << "," << k << "," << col << ")" << std::endl;
        std::cout << "lhs " << plhs[0] << " " << plhs[1] << " " << plhs[2] << " " << plhs[3] << std::endl;
        std::cout << "rhs " << prhs[0] << " " << prhs[1] << " " << prhs[2] << " " << prhs[3] << std::endl;
#endif
        acc = plhs*prhs;

        r0.storePacket(0,r0.template loadPacket<ResPacket>(0) + acc);
        lhs_ptr += accLhsProgress;
      }
      auto residue = 0;
      for(;row < rows; row++)
      {
        LhsScalar lhs = *(lhsMap.get_residue_at(residue) + k);
#ifdef __DEBUG__
        std::cout << "(" << row << "," << k << "," << col << ")" << std::endl;
        std::cout << "lhs " << lhs << " (" << prhs[0] << " " << prhs[1] << " " << prhs[2] << " " << prhs[3] << ")" << std::endl;
#endif
        res(row, col + 0) += lhs*prhs[0];
        residue++;
      }
    }
    colResidue++;
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