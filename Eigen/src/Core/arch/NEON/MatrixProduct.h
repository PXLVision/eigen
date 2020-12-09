// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2020 Everton Constantino (everton.constantino@ibm.com)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATRIX_PRODUCT_NEON_H
#define EIGEN_MATRIX_PRODUCT_NEON_H

/**************************************************************************************************
 * TODO                                                                                           *
 * - Check StorageOrder on lhs_pack (the innermost second loop seems unvectorized when it could). *
 * - Check the possibility of transposing as GETREAL and GETIMAG when needed.                     *
 * - Check if change conjugation to xor instead of mul gains any performance.                     *
 * - Remove IsComplex template argument from complex packing.                                     *
 **************************************************************************************************/
namespace Eigen {

namespace internal {

/**************************
 * Constants and typedefs *
 **************************/
const int QuadRegisterCount = 8;

template<typename Scalar>
struct quad_traits
{
  typedef typename packet_traits<Scalar>::type    vectortype;
  typedef PacketBlock<vectortype, 4>                    type;
  typedef vectortype                                 rhstype;
  enum
  {
    vectorsize = packet_traits<Scalar>::size,
    size = 4,
    rows = 4
  };
};

template<>
struct quad_traits<double>
{
  typedef Packet2d                        vectortype;
  typedef PacketBlock<vectortype, 4>            type;
  typedef PacketBlock<Packet2d,2>            rhstype;
  enum
  {
    vectorsize = packet_traits<double>::size,
    size = 2,
    rows = 4
  };
};

// MatrixProduct decomposes real/imaginary vectors into a real vector and an imaginary vector, this turned out
// to be faster than Eigen's usual approach of having real/imaginary pairs on a single vector. This constants then
// are responsible to extract from convert between Eigen's and MatrixProduct approach.
const static Packet4f p4f_CONJUGATE = {-1.0f, -1.0f, -1.0f, -1.0f};

const static Packet2d p2d_CONJUGATE = {-1.0f, -1.0f};

const static Packet16uc p16uc_GETREAL32 = {  0,  1,  2,  3,
                                             8,  9, 10, 11,
                                            16, 17, 18, 19,
                                            24, 25, 26, 27};

const static Packet16uc p16uc_GETIMAG32 = {  4,  5,  6,  7,
                                            12, 13, 14, 15,
                                            20, 21, 22, 23,
                                            28, 29, 30, 31};

const static Packet16uc p16uc_SETCOMPLEX32_FIRST = {  0,  1,  2,  3,
                                                     16, 17, 18, 19,
                                                      4,  5,  6,  7,
                                                     20, 21, 22, 23};

const static Packet16uc p16uc_SETCOMPLEX32_SECOND = {  8,  9, 10, 11,
                                                      24, 25, 26, 27,
                                                      12, 13, 14, 15,
                                                      28, 29, 30, 31};
//[a,ai],[b,bi] = [a,b]
const static Packet16uc p16uc_GETREAL64 = {  0,  1,  2,  3,  4,  5,  6,  7,
                                            16, 17, 18, 19, 20, 21, 22, 23};

//[a,ai],[b,bi] = [ai,bi]
const static Packet16uc p16uc_GETIMAG64 = {  8,  9, 10, 11, 12, 13, 14, 15,
                                            24, 25, 26, 27, 28, 29, 30, 31};

//[a,b],[ai,bi] = [a,ai] - This is equivalent to p16uc_GETREAL64
const static Packet16uc p16uc_SETCOMPLEX64_FIRST = {  0,  1,  2,  3,  4,  5,  6,  7,
                                                     16, 17, 18, 19, 20, 21, 22, 23};

//[a,b],[ai,bi] = [b,bi] - This is equivalent to p16uc_GETIMAG64
const static Packet16uc p16uc_SETCOMPLEX64_SECOND = {  8,  9, 10, 11, 12, 13, 14, 15,
                                                      24, 25, 26, 27, 28, 29, 30, 31};

/*********************************************
 * Single precision real and complex packing *
 * *******************************************/

/**
 * Symm packing is related to packing of symmetric adjoint blocks, as expected the packing leaves
 * the diagonal real, whatever is below it is copied from the respective upper diagonal element and 
 * conjugated. There's no PanelMode available for symm packing.
 *
 * Packing in general is supposed to leave the lhs block and the rhs block easy to be read by gemm using 
 * it's respective rank-update instructions. The float32/64 versions are different because at this moment
 * the size of the accumulator is fixed at 512-bits so you can't have a 4x4 accumulator of 64-bit elements.
 * 
 * As mentioned earlier MatrixProduct breaks complex numbers into a real vector and a complex vector so packing has
 * to take that into account, at the moment, we run pack the real part and then the imaginary part, this is the main
 * reason why packing for complex is broken down into several different parts, also the reason why we endup having a
 * float32/64 and complex float32/64 version.
 **/
template<typename Scalar, typename Index, int StorageOrder>
EIGEN_STRONG_INLINE std::complex<Scalar> getAdjointVal(Index i, Index j, const_blas_data_mapper<std::complex<Scalar>, Index, StorageOrder>& dt)
{
  std::complex<Scalar> v;
  if(i < j)
  {
    v.real(dt(j,i).real());
    v.imag(-dt(j,i).imag());
  } else if(i > j)
  {
    v.real(dt(i,j).real());
    v.imag(dt(i,j).imag());
  } else {
    v.real(dt(i,j).real());
    v.imag((Scalar)0.0f);
  }
  return v;
}

template<typename Scalar, typename Index, int StorageOrder, int N>
EIGEN_STRONG_INLINE void symm_pack_complex_rhs_helper(std::complex<Scalar> *blockB, const std::complex<Scalar>* _rhs, Index rhsStride, Index rows, Index cols, Index k2)
{
  const Index depth = k2 + rows;
  const_blas_data_mapper<std::complex<Scalar>, Index, StorageOrder> rhs(_rhs, rhsStride);
  const int vectorSize = N*quad_traits<Scalar>::vectorsize;
  Scalar* blockBf = reinterpret_cast<Scalar *>(blockB);

  Index ri = 0, j = 0;
  for(; j + vectorSize < cols; j+=vectorSize)
  {
      Index i = k2;
      for(; i < depth; i++)
      {
        for(Index k = 0; k < vectorSize; k++)
        {
          std::complex<Scalar> v = getAdjointVal<Scalar, Index, StorageOrder>(i, j + k, rhs);
          blockBf[ri + k] = v.real();
        }
        ri += vectorSize;
      }

      i = k2;

      for(; i < depth; i++)
      {
        for(Index k = 0; k < vectorSize; k++)
        {
          std::complex<Scalar> v = getAdjointVal<Scalar, Index, StorageOrder>(i, j + k, rhs);
          blockBf[ri + k] = v.imag();
        }
        ri += vectorSize;
      }
  }
  for(Index i = k2; i < depth; i++)
  {
      Index k = j;
      for(; k < cols; k++)
      {
        std::complex<Scalar> v = getAdjointVal<Scalar, Index, StorageOrder>(i, k, rhs);
        blockBf[ri] = v.real();
        ri += 1;
      }
  }
  for(Index i = k2; i < depth; i++)
  {
      Index k = j;
      for(; k < cols; k++)
      {
        std::complex<Scalar> v = getAdjointVal<Scalar, Index, StorageOrder>(i, k, rhs);
        blockBf[ri] = v.imag();
        ri += 1;
      }
  }
}

template<typename Scalar, typename Index, int StorageOrder>
EIGEN_STRONG_INLINE void symm_pack_complex_lhs_helper(std::complex<Scalar> *blockA, const std::complex<Scalar>* _lhs, Index lhsStride, Index cols, Index rows)
{
  const Index depth = cols;
  const_blas_data_mapper<std::complex<Scalar>, Index, StorageOrder> lhs(_lhs, lhsStride);
  const int vectorSize = quad_traits<Scalar>::vectorsize;
  Index ri = 0, j = 0;
  Scalar *blockAf = (Scalar *)(blockA);

  for(; j + vectorSize < rows; j+=vectorSize)
  {
      Index i = 0;

      for(; i < depth; i++)
      {
        for(int k = 0; k < vectorSize; k++)
        {
          std::complex<Scalar> v = getAdjointVal<Scalar, Index, StorageOrder>(j+k, i, lhs);
          blockAf[ri + k] = v.real();
        }
        ri += vectorSize;
      }
      i = 0;
      for(; i < depth; i++)
      {
        for(int k = 0; k < vectorSize; k++)
        {
          std::complex<Scalar> v = getAdjointVal<Scalar, Index, StorageOrder>(j+k, i, lhs);
          blockAf[ri + k] = v.imag();
        }
        ri += vectorSize;
      }
  }

  for(Index i = 0; i < depth; i++)
  {
      Index k = j;
      for(; k < rows; k++)
      {
          std::complex<Scalar> v = getAdjointVal<Scalar, Index, StorageOrder>(k, i, lhs);
          blockAf[ri] = v.real();
          ri += 1;
      }
  }
  for(Index i = 0; i < depth; i++)
  {
      Index k = j;
      for(; k < rows; k++)
      {
          std::complex<Scalar> v = getAdjointVal<Scalar, Index, StorageOrder>(k, i, lhs);
          blockAf[ri] = v.imag();
          ri += 1;
      }
  }
}

template<typename Scalar, typename Index, int StorageOrder, int N>
EIGEN_STRONG_INLINE void symm_pack_rhs_helper(Scalar *blockB, const Scalar* _rhs, Index rhsStride, Index rows, Index cols, Index k2)
{
  const Index depth = k2 + rows;
  const_blas_data_mapper<Scalar, Index, StorageOrder> rhs(_rhs, rhsStride);
  const int vectorSize = quad_traits<Scalar>::vectorsize;

  Index ri = 0, j = 0;
  for(; j + N*vectorSize < cols; j+=N*vectorSize)
  {
      Index i = k2;
      for(; i < depth; i++)
      {
        for(int k = 0; k < N*vectorSize; k++)
        {
          if(i <= j+k)
            blockB[ri + k] = rhs(j+k, i);
          else
            blockB[ri + k] = rhs(i, j+k);
        }
        ri += N*vectorSize;
      }
  }
  for(Index i = k2; i < depth; i++)
  {
      Index k = j;
      for(; k < cols; k++)
      {
        if(k <= i)
          blockB[ri] = rhs(i, k);
        else
          blockB[ri] = rhs(k, i);
        ri += 1;
      }
  }
}

template<typename Scalar, typename Index, int StorageOrder>
EIGEN_STRONG_INLINE void symm_pack_lhs_helper(Scalar *blockA, const Scalar* _lhs, Index lhsStride, Index cols, Index rows)
{
  const Index depth = cols;
  const_blas_data_mapper<Scalar, Index, StorageOrder> lhs(_lhs, lhsStride);
  const int vectorSize = quad_traits<Scalar>::vectorsize;
  Index ri = 0, j = 0;

  for(j = 0; j + vectorSize < rows; j+=vectorSize)
  {
      Index i = 0;

      for(; i < depth; i++)
      {
        for(int k = 0; k < vectorSize; k++)
        {
          if(i <= j+k)
            blockA[ri + k] = lhs(j+k, i);
          else
            blockA[ri + k] = lhs(i, j+k);
        }
        ri += vectorSize;
      }
  }

  for(Index i = 0; i < depth; i++)
  {
      Index k = j;
      for(; k < rows; k++)
      {
          if(i <= k)
            blockA[ri] = lhs(k, i);
          else
            blockA[ri] = lhs(i, k);
          ri += 1;
      }
  }
}

/*
template<typename Index, int nr, int StorageOrder>
struct symm_pack_rhs<std::complex<float>, Index, nr, StorageOrder>
{
  void operator()(std::complex<float>* blockB, const std::complex<float>* _rhs, Index rhsStride, Index rows, Index cols, Index k2)
  {
    symm_pack_complex_rhs_helper<float, Index, StorageOrder, 1>(blockB, _rhs, rhsStride, rows, cols, k2);
  }
};

template<typename Index, int Pack1, int Pack2_dummy, int StorageOrder>
struct symm_pack_lhs<std::complex<float>, Index, Pack1, Pack2_dummy, StorageOrder>
{
  void operator()(std::complex<float>* blockA, const std::complex<float>* _lhs, Index lhsStride, Index cols, Index rows)
  {
   symm_pack_complex_lhs_helper<float, Index, StorageOrder>(blockA, _lhs, lhsStride, cols, rows);
  }
};

// *********** symm_pack std::complex<float64> ***********

template<typename Index, int nr, int StorageOrder>
struct symm_pack_rhs<std::complex<double>, Index, nr, StorageOrder>
{
  void operator()(std::complex<double>* blockB, const std::complex<double>* _rhs, Index rhsStride, Index rows, Index cols, Index k2)
  {
    symm_pack_complex_rhs_helper<double, Index, StorageOrder, 2>(blockB, _rhs, rhsStride, rows, cols, k2);
  }
};

template<typename Index, int Pack1, int Pack2_dummy, int StorageOrder>
struct symm_pack_lhs<std::complex<double>, Index, Pack1, Pack2_dummy, StorageOrder>
{
  void operator()(std::complex<double>* blockA, const std::complex<double>* _lhs, Index lhsStride, Index cols, Index rows)
  {
    symm_pack_complex_lhs_helper<double, Index, StorageOrder>(blockA, _lhs, lhsStride, cols, rows);
  }
};
*/
// *********** symm_pack float32 ***********
template<typename Index, int nr, int StorageOrder>
struct symm_pack_rhs<float, Index, nr, StorageOrder>
{
  void operator()(float* blockB, const float* _rhs, Index rhsStride, Index rows, Index cols, Index k2)
  {
   symm_pack_rhs_helper<float, Index, StorageOrder, 1>(blockB, _rhs, rhsStride, rows, cols, k2);
  }
};

template<typename Index, int Pack1, int Pack2_dummy, int StorageOrder>
struct symm_pack_lhs<float, Index, Pack1, Pack2_dummy, StorageOrder>
{
  void operator()(float* blockA, const float* _lhs, Index lhsStride, Index cols, Index rows)
  {
   symm_pack_lhs_helper<float, Index, StorageOrder>(blockA, _lhs, lhsStride, cols, rows);
  }
};

// *********** symm_pack float64 ***********
template<typename Index, int nr, int StorageOrder>
struct symm_pack_rhs<double, Index, nr, StorageOrder>
{
  void operator()(double* blockB, const double* _rhs, Index rhsStride, Index rows, Index cols, Index k2)
  {
    symm_pack_rhs_helper<double, Index, StorageOrder, 2>(blockB, _rhs, rhsStride, rows, cols, k2);
  }
};

template<typename Index, int Pack1, int Pack2_dummy, int StorageOrder>
struct symm_pack_lhs<double, Index, Pack1, Pack2_dummy, StorageOrder>
{
  void operator()(double* blockA, const double* _lhs, Index lhsStride, Index cols, Index rows)
  {
   symm_pack_lhs_helper<double, Index, StorageOrder>(blockA, _lhs, lhsStride, cols, rows);
  }
};

/**
 * PanelMode
 * Packing might be called several times before being multiplied by gebp_kernel, this happens because 
 * on special occasions it fills part of block with other parts of the matrix. Two variables control
 * how PanelMode should behave: offset and stride. The idea is that those variables represent whatever
 * is going to be the real offset and stride in the future and this is what you should obey. The process
 * is to behave as you would with normal packing but leave the start of each part with the correct offset
 * and the end as well respecting the real stride the block will have. Gebp is aware of both blocks stride
 * and offset and behaves accordingly.
 **/
/*
// General template for lhs complex packing.
template<typename Scalar, bool IsComplex, typename Index, typename DataMapper, typename Packet, typename PacketC, int StorageOrder, bool Conjugate, bool PanelMode>
struct lhs_cpack {
  EIGEN_STRONG_INLINE void operator()(std::complex<Scalar>* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride, Index offset)
  {
    const int vectorSize = quad_traits<Scalar>::vectorsize;
    Index ri = 0, j = 0;
    Scalar *blockAt  = reinterpret_cast<Scalar *>(blockA);
    Packet conj = pset1<Packet>((Scalar)-1.0f);

    for(j = 0; j + vectorSize < rows; j+=vectorSize)
    {
      Index i = 0;

      if(PanelMode) ri += vectorSize*offset;

      for(; i + vectorSize < depth; i+=vectorSize)
      {
        PacketBlock<Packet, 4> block;

        PacketBlock<PacketC, 8> cblock;
        if(StorageOrder == ColMajor)
        {
          cblock.packet[0] = lhs.template loadPacket<PacketC>(j, i + 0);
          cblock.packet[1] = lhs.template loadPacket<PacketC>(j, i + 1);
          cblock.packet[2] = lhs.template loadPacket<PacketC>(j, i + 2);
          cblock.packet[3] = lhs.template loadPacket<PacketC>(j, i + 3);

          cblock.packet[4] = lhs.template loadPacket<PacketC>(j + 2, i + 0);
          cblock.packet[5] = lhs.template loadPacket<PacketC>(j + 2, i + 1);
          cblock.packet[6] = lhs.template loadPacket<PacketC>(j + 2, i + 2);
          cblock.packet[7] = lhs.template loadPacket<PacketC>(j + 2, i + 3);
        } else {
          cblock.packet[0] = lhs.template loadPacket<PacketC>(j + 0, i);
          cblock.packet[1] = lhs.template loadPacket<PacketC>(j + 1, i);
          cblock.packet[2] = lhs.template loadPacket<PacketC>(j + 2, i);
          cblock.packet[3] = lhs.template loadPacket<PacketC>(j + 3, i);

          cblock.packet[4] = lhs.template loadPacket<PacketC>(j + 0, i + 2);
          cblock.packet[5] = lhs.template loadPacket<PacketC>(j + 1, i + 2);
          cblock.packet[6] = lhs.template loadPacket<PacketC>(j + 2, i + 2);
          cblock.packet[7] = lhs.template loadPacket<PacketC>(j + 3, i + 2);
        }

        block.packet[0] = vec_perm(cblock.packet[0].v , cblock.packet[4].v, p16uc_GETREAL32);
        block.packet[1] = vec_perm(cblock.packet[1].v , cblock.packet[5].v, p16uc_GETREAL32);
        block.packet[2] = vec_perm(cblock.packet[2].v , cblock.packet[6].v, p16uc_GETREAL32);
        block.packet[3] = vec_perm(cblock.packet[3].v , cblock.packet[7].v, p16uc_GETREAL32);

        if(StorageOrder == RowMajor) ptranspose(block);

        pstore<Scalar>(blockAt + ri     , block.packet[0]);
        pstore<Scalar>(blockAt + ri +  4, block.packet[1]);
        pstore<Scalar>(blockAt + ri +  8, block.packet[2]);
        pstore<Scalar>(blockAt + ri + 12, block.packet[3]);

        ri += 4*vectorSize;
      }
      for(; i < depth; i++)
      {
        blockAt[ri + 0] = lhs(j + 0, i).real();
        blockAt[ri + 1] = lhs(j + 1, i).real();
        blockAt[ri + 2] = lhs(j + 2, i).real();
        blockAt[ri + 3] = lhs(j + 3, i).real();

        ri += vectorSize;
      }
      if(PanelMode) ri += vectorSize*(stride - offset - depth);
      
      i = 0;

      if(PanelMode) ri += vectorSize*offset;

      for(; i + vectorSize < depth; i+=vectorSize)
      {
        PacketBlock<PacketC, 8> cblock;
        if(StorageOrder == ColMajor)
        {
          cblock.packet[0] = lhs.template loadPacket<PacketC>(j, i + 0);
          cblock.packet[1] = lhs.template loadPacket<PacketC>(j, i + 1);
          cblock.packet[2] = lhs.template loadPacket<PacketC>(j, i + 2);
          cblock.packet[3] = lhs.template loadPacket<PacketC>(j, i + 3);

          cblock.packet[4] = lhs.template loadPacket<PacketC>(j + 2, i + 0);
          cblock.packet[5] = lhs.template loadPacket<PacketC>(j + 2, i + 1);
          cblock.packet[6] = lhs.template loadPacket<PacketC>(j + 2, i + 2);
          cblock.packet[7] = lhs.template loadPacket<PacketC>(j + 2, i + 3);
        } else {
          cblock.packet[0] = lhs.template loadPacket<PacketC>(j + 0, i);
          cblock.packet[1] = lhs.template loadPacket<PacketC>(j + 1, i);
          cblock.packet[2] = lhs.template loadPacket<PacketC>(j + 2, i);
          cblock.packet[3] = lhs.template loadPacket<PacketC>(j + 3, i);

          cblock.packet[4] = lhs.template loadPacket<PacketC>(j + 0, i + 2);
          cblock.packet[5] = lhs.template loadPacket<PacketC>(j + 1, i + 2);
          cblock.packet[6] = lhs.template loadPacket<PacketC>(j + 2, i + 2);
          cblock.packet[7] = lhs.template loadPacket<PacketC>(j + 3, i + 2);
        }

        PacketBlock<Packet, 4> block;
        block.packet[0] = vec_perm(cblock.packet[0].v , cblock.packet[4].v, p16uc_GETIMAG32);
        block.packet[1] = vec_perm(cblock.packet[1].v , cblock.packet[5].v, p16uc_GETIMAG32);
        block.packet[2] = vec_perm(cblock.packet[2].v , cblock.packet[6].v, p16uc_GETIMAG32);
        block.packet[3] = vec_perm(cblock.packet[3].v , cblock.packet[7].v, p16uc_GETIMAG32);

        if(Conjugate)
        {
          block.packet[0] *= conj;
          block.packet[1] *= conj;
          block.packet[2] *= conj;
          block.packet[3] *= conj;
        }

        if(StorageOrder == RowMajor) ptranspose(block);

        pstore<Scalar>(blockAt + ri     , block.packet[0]);
        pstore<Scalar>(blockAt + ri +  4, block.packet[1]);
        pstore<Scalar>(blockAt + ri +  8, block.packet[2]);
        pstore<Scalar>(blockAt + ri + 12, block.packet[3]);

        ri += 4*vectorSize;
      }
      for(; i < depth; i++)
      {
        if(Conjugate)
        {
          blockAt[ri + 0] = -lhs(j + 0, i).imag();
          blockAt[ri + 1] = -lhs(j + 1, i).imag();
          blockAt[ri + 2] = -lhs(j + 2, i).imag();
          blockAt[ri + 3] = -lhs(j + 3, i).imag();
        } else {
          blockAt[ri + 0] = lhs(j + 0, i).imag();
          blockAt[ri + 1] = lhs(j + 1, i).imag();
          blockAt[ri + 2] = lhs(j + 2, i).imag();
          blockAt[ri + 3] = lhs(j + 3, i).imag();
        }

        ri += vectorSize;
      }
      if(PanelMode) ri += vectorSize*(stride - offset - depth);
    }

    if(PanelMode) ri += offset*(rows - j);

    for(Index i = 0; i < depth; i++)
    {
      Index k = j;
      for(; k < rows; k++)
      {
        blockAt[ri] = lhs(k, i).real();
        ri += 1;
      }
    }

    if(PanelMode) ri += (rows - j)*(stride - offset - depth);

    if(PanelMode) ri += offset*(rows - j);

    for(Index i = 0; i < depth; i++)
    {
      Index k = j;
      for(; k < rows; k++)
      {
        if(Conjugate)
          blockAt[ri] = -lhs(k, i).imag();
        else
          blockAt[ri] = lhs(k, i).imag();
        ri += 1;
      }
    }

    if(PanelMode) ri += (rows - j)*(stride - offset - depth);
  }
};
*/
// General template for lhs packing.
template<typename Scalar, typename Index, typename DataMapper, typename Packet, int StorageOrder, bool PanelMode>
struct lhs_pack{
  EIGEN_STRONG_INLINE void operator()(Scalar* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride, Index offset)
  {
    const int vectorSize = quad_traits<Scalar>::vectorsize;
    Index ri = 0, j = 0;

    for(j = 0; j + vectorSize < rows; j+=vectorSize)
    {
      Index i = 0;

      if(PanelMode) ri += vectorSize*offset;

      for(; i + vectorSize < depth; i+=vectorSize)
      {
        PacketBlock<Packet, 4> block;

        if(StorageOrder == RowMajor)
        {
          block.packet[0] = lhs.template loadPacket<Packet>(j + 0, i);
          block.packet[1] = lhs.template loadPacket<Packet>(j + 1, i);
          block.packet[2] = lhs.template loadPacket<Packet>(j + 2, i);
          block.packet[3] = lhs.template loadPacket<Packet>(j + 3, i);

          ptranspose(block);
        } else {
          block.packet[0] = lhs.template loadPacket<Packet>(j, i + 0);
          block.packet[1] = lhs.template loadPacket<Packet>(j, i + 1);
          block.packet[2] = lhs.template loadPacket<Packet>(j, i + 2);
          block.packet[3] = lhs.template loadPacket<Packet>(j, i + 3);
        }

        pstore<Scalar>(blockA + ri     , block.packet[0]);
        pstore<Scalar>(blockA + ri +  4, block.packet[1]);
        pstore<Scalar>(blockA + ri +  8, block.packet[2]);
        pstore<Scalar>(blockA + ri + 12, block.packet[3]);

        ri += 4*vectorSize;
      }
      for(; i < depth; i++)
      {
        if(StorageOrder == RowMajor)
        {
          blockA[ri+0] = lhs(j+0, i);
          blockA[ri+1] = lhs(j+1, i);
          blockA[ri+2] = lhs(j+2, i);
          blockA[ri+3] = lhs(j+3, i);
        } else {
          Packet lhsV = lhs.template loadPacket<Packet>(j, i);
          pstore<Scalar>(blockA + ri, lhsV);
        }

        ri += vectorSize;
      }
      if(PanelMode) ri += vectorSize*(stride - offset - depth);
    }

    if(PanelMode) ri += offset*(rows - j);

    for(Index i = 0; i < depth; i++)
    {
      Index k = j;
      for(; k < rows; k++)
      {
        blockA[ri] = lhs(k, i);
        ri += 1;
      }
    }

    if(PanelMode) ri += (rows - j)*(stride - offset - depth);
  }
};

// General template for rhs complex packing.
/*
template<typename Scalar, typename Index, typename DataMapper, typename Packet, typename PacketC, int StorageOrder, bool Conjugate, bool PanelMode>
struct rhs_cpack
{
  EIGEN_STRONG_INLINE void operator()(std::complex<Scalar>* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset)
  {
    const int vectorSize = quad_traits<Scalar>::vectorsize;
    Scalar *blockBt = reinterpret_cast<Scalar *>(blockB);
    Packet conj = pset1<Packet>((Scalar)-1.0f);

    Index ri = 0, j = 0;
    for(; j + vectorSize < cols; j+=vectorSize)
    {
        Index i = 0;

        if(PanelMode) ri += offset*vectorSize;

        for(; i + vectorSize < depth; i+=vectorSize)
        {
            PacketBlock<PacketC, 8> cblock;
            if(StorageOrder == ColMajor)
            {
              cblock.packet[0] = rhs.template loadPacket<PacketC>(i, j + 0);
              cblock.packet[1] = rhs.template loadPacket<PacketC>(i, j + 1);
              cblock.packet[2] = rhs.template loadPacket<PacketC>(i, j + 2);
              cblock.packet[3] = rhs.template loadPacket<PacketC>(i, j + 3);

              cblock.packet[4] = rhs.template loadPacket<PacketC>(i + 2, j + 0);
              cblock.packet[5] = rhs.template loadPacket<PacketC>(i + 2, j + 1);
              cblock.packet[6] = rhs.template loadPacket<PacketC>(i + 2, j + 2);
              cblock.packet[7] = rhs.template loadPacket<PacketC>(i + 2, j + 3);
            } else {
              cblock.packet[0] = rhs.template loadPacket<PacketC>(i + 0, j);
              cblock.packet[1] = rhs.template loadPacket<PacketC>(i + 1, j);
              cblock.packet[2] = rhs.template loadPacket<PacketC>(i + 2, j);
              cblock.packet[3] = rhs.template loadPacket<PacketC>(i + 3, j);

              cblock.packet[4] = rhs.template loadPacket<PacketC>(i + 0, j + 2);
              cblock.packet[5] = rhs.template loadPacket<PacketC>(i + 1, j + 2);
              cblock.packet[6] = rhs.template loadPacket<PacketC>(i + 2, j + 2);
              cblock.packet[7] = rhs.template loadPacket<PacketC>(i + 3, j + 2);
            }

            PacketBlock<Packet, 4> block;
            block.packet[0] = vec_perm(cblock.packet[0].v , cblock.packet[4].v, p16uc_GETREAL32);
            block.packet[1] = vec_perm(cblock.packet[1].v , cblock.packet[5].v, p16uc_GETREAL32);
            block.packet[2] = vec_perm(cblock.packet[2].v , cblock.packet[6].v, p16uc_GETREAL32);
            block.packet[3] = vec_perm(cblock.packet[3].v , cblock.packet[7].v, p16uc_GETREAL32);

            if(StorageOrder == ColMajor) ptranspose(block);

            pstore<Scalar>(blockBt + ri     , block.packet[0]);
            pstore<Scalar>(blockBt + ri +  4, block.packet[1]);
            pstore<Scalar>(blockBt + ri +  8, block.packet[2]);
            pstore<Scalar>(blockBt + ri + 12, block.packet[3]);

            ri += 4*vectorSize;
        }
        for(; i < depth; i++)
        {
            blockBt[ri+0] = rhs(i, j+0).real();
            blockBt[ri+1] = rhs(i, j+1).real();
            blockBt[ri+2] = rhs(i, j+2).real();
            blockBt[ri+3] = rhs(i, j+3).real();
            ri += vectorSize;
        }

        if(PanelMode) ri += vectorSize*(stride - offset - depth);

        i = 0;

        if(PanelMode) ri += offset*vectorSize;

        for(; i + vectorSize < depth; i+=vectorSize)
        {
          PacketBlock<PacketC, 8> cblock;
          if(StorageOrder == ColMajor)
          {

            cblock.packet[0] = rhs.template loadPacket<PacketC>(i, j + 0);
            cblock.packet[1] = rhs.template loadPacket<PacketC>(i, j + 1);
            cblock.packet[2] = rhs.template loadPacket<PacketC>(i, j + 2);
            cblock.packet[3] = rhs.template loadPacket<PacketC>(i, j + 3);

            cblock.packet[4] = rhs.template loadPacket<PacketC>(i + 2, j + 0);
            cblock.packet[5] = rhs.template loadPacket<PacketC>(i + 2, j + 1);
            cblock.packet[6] = rhs.template loadPacket<PacketC>(i + 2, j + 2);
            cblock.packet[7] = rhs.template loadPacket<PacketC>(i + 2, j + 3);
          } else {
            cblock.packet[0] = rhs.template loadPacket<PacketC>(i + 0, j);
            cblock.packet[1] = rhs.template loadPacket<PacketC>(i + 1, j);
            cblock.packet[2] = rhs.template loadPacket<PacketC>(i + 2, j);
            cblock.packet[3] = rhs.template loadPacket<PacketC>(i + 3, j);

            cblock.packet[4] = rhs.template loadPacket<PacketC>(i + 0, j + 2);
            cblock.packet[5] = rhs.template loadPacket<PacketC>(i + 1, j + 2);
            cblock.packet[6] = rhs.template loadPacket<PacketC>(i + 2, j + 2);
            cblock.packet[7] = rhs.template loadPacket<PacketC>(i + 3, j + 2);
          }

          PacketBlock<Packet, 4> block;
          block.packet[0] = vec_perm(cblock.packet[0].v , cblock.packet[4].v, p16uc_GETIMAG32);
          block.packet[1] = vec_perm(cblock.packet[1].v , cblock.packet[5].v, p16uc_GETIMAG32);
          block.packet[2] = vec_perm(cblock.packet[2].v , cblock.packet[6].v, p16uc_GETIMAG32);
          block.packet[3] = vec_perm(cblock.packet[3].v , cblock.packet[7].v, p16uc_GETIMAG32);

          if(Conjugate)
          {
            block.packet[0] *= conj;
            block.packet[1] *= conj;
            block.packet[2] *= conj;
            block.packet[3] *= conj;
          }

          if(StorageOrder == ColMajor) ptranspose(block);

          pstore<Scalar>(blockBt + ri     , block.packet[0]);
          pstore<Scalar>(blockBt + ri +  4, block.packet[1]);
          pstore<Scalar>(blockBt + ri +  8, block.packet[2]);
          pstore<Scalar>(blockBt + ri + 12, block.packet[3]);

          ri += 4*vectorSize;
        }
        for(; i < depth; i++)
        {
            if(Conjugate)
            {
              blockBt[ri+0] = -rhs(i, j+0).imag();
              blockBt[ri+1] = -rhs(i, j+1).imag();
              blockBt[ri+2] = -rhs(i, j+2).imag();
              blockBt[ri+3] = -rhs(i, j+3).imag();
            } else {
              blockBt[ri+0] = rhs(i, j+0).imag();
              blockBt[ri+1] = rhs(i, j+1).imag();
              blockBt[ri+2] = rhs(i, j+2).imag();
              blockBt[ri+3] = rhs(i, j+3).imag();
            }
            ri += vectorSize;
        }

        if(PanelMode) ri += vectorSize*(stride - offset - depth);
    }

    if(PanelMode) ri += offset*(cols - j);

    for(Index i = 0; i < depth; i++)
    {
        Index k = j;
        for(; k < cols; k++)
        {
            blockBt[ri] = rhs(i, k).real();
            ri += 1;
        }
    }
    if(PanelMode) ri += (cols - j)*(stride - offset - depth);

    if(PanelMode) ri += offset*(cols - j);

    for(Index i = 0; i < depth; i++)
    {
        Index k = j;
        for(; k < cols; k++)
        {
            if(Conjugate)
              blockBt[ri] = -rhs(i, k).imag();
            else
              blockBt[ri] = rhs(i, k).imag();
            ri += 1;
        }
    }
    if(PanelMode) ri += (cols - j)*(stride - offset - depth);
  }
};
*/
// General template for rhs packing.
template<typename Scalar, typename Index, typename DataMapper, typename Packet, int StorageOrder, bool PanelMode>
struct rhs_pack {
  EIGEN_STRONG_INLINE void operator()(Scalar* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset)
  {
    const int vectorSize = quad_traits<Scalar>::vectorsize;
    Index ri = 0, j = 0;
    for(; j + vectorSize < cols; j+=vectorSize)
    {
      Index i = 0;

      if(PanelMode) ri += offset*vectorSize;

      for(; i + vectorSize < depth; i+=vectorSize)
      {
        PacketBlock<Packet, 4> block;
        if(StorageOrder == ColMajor)
        {
          block.packet[0] = rhs.template loadPacket<Packet>(i, j + 0);
          block.packet[1] = rhs.template loadPacket<Packet>(i, j + 1);
          block.packet[2] = rhs.template loadPacket<Packet>(i, j + 2);
          block.packet[3] = rhs.template loadPacket<Packet>(i, j + 3);

          ptranspose(block);
        } else {
          block.packet[0] = rhs.template loadPacket<Packet>(i + 0, j);
          block.packet[1] = rhs.template loadPacket<Packet>(i + 1, j);
          block.packet[2] = rhs.template loadPacket<Packet>(i + 2, j);
          block.packet[3] = rhs.template loadPacket<Packet>(i + 3, j);
        }

        pstore<Scalar>(blockB + ri     , block.packet[0]);
        pstore<Scalar>(blockB + ri +  4, block.packet[1]);
        pstore<Scalar>(blockB + ri +  8, block.packet[2]);
        pstore<Scalar>(blockB + ri + 12, block.packet[3]);

        ri += 4*vectorSize;
      }
      for(; i < depth; i++)
      {
        if(StorageOrder == ColMajor)
        {
          blockB[ri+0] = rhs(i, j+0);
          blockB[ri+1] = rhs(i, j+1);
          blockB[ri+2] = rhs(i, j+2);
          blockB[ri+3] = rhs(i, j+3);
        } else {
          Packet rhsV = rhs.template loadPacket<Packet>(i, j);
          pstore<Scalar>(blockB + ri, rhsV);
        }
        ri += vectorSize;
      }

      if(PanelMode) ri += vectorSize*(stride - offset - depth);
    }

    if(PanelMode) ri += offset*(cols - j);

    for(Index i = 0; i < depth; i++)
    {
      Index k = j;
      for(; k < cols; k++)
      {
        blockB[ri] = rhs(i, k);
        ri += 1;
      }
    }
    if(PanelMode) ri += (cols - j)*(stride - offset - depth);
  }
};

// General template for lhs packing, float64 specialization.
template<typename Index, typename DataMapper, int StorageOrder, bool PanelMode>
struct lhs_pack<double,Index, DataMapper, Packet2d, StorageOrder, PanelMode>
{
  EIGEN_STRONG_INLINE void operator()(double* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride, Index offset)
  {
    const int vectorSize = quad_traits<double>::vectorsize;
    Index ri = 0, j = 0;

    for(j = 0; j + vectorSize < rows; j+=vectorSize)
    {
      Index i = 0;

      if(PanelMode) ri += vectorSize*offset;

      for(; i + vectorSize < depth; i+=vectorSize)
      {
        PacketBlock<Packet2d, 2> block;
        if(StorageOrder == RowMajor)
        {
          block.packet[0] = lhs.template loadPacket<Packet2d>(j + 0, i);
          block.packet[1] = lhs.template loadPacket<Packet2d>(j + 1, i);

          ptranspose(block);
        } else {
          block.packet[0] = lhs.template loadPacket<Packet2d>(j, i + 0);
          block.packet[1] = lhs.template loadPacket<Packet2d>(j, i + 1);
        }

        pstore<double>(blockA + ri    , block.packet[0]);
        pstore<double>(blockA + ri + 2, block.packet[1]);

        ri += 2*vectorSize;
      }
      for(; i < depth; i++)
      {
        if(StorageOrder == RowMajor)
        {
          blockA[ri+0] = lhs(j+0, i);
          blockA[ri+1] = lhs(j+1, i);
        } else {
          Packet2d lhsV = lhs.template loadPacket<Packet2d>(j, i);
          pstore<double>(blockA + ri, lhsV);
        }

        ri += vectorSize;
      }
      if(PanelMode) ri += vectorSize*(stride - offset - depth);
    }

    if(PanelMode) ri += offset*(rows - j);

    for(Index i = 0; i < depth; i++)
    {
      Index k = j;
      for(; k < rows; k++)
      {
        blockA[ri] = lhs(k, i);
        ri += 1;
      }
    }

    if(PanelMode) ri += (rows - j)*(stride - offset - depth);
  }
};

// General template for rhs packing, float64 specialization.
template<typename Index, typename DataMapper, int StorageOrder, bool PanelMode>
struct rhs_pack<double, Index, DataMapper, Packet2d, StorageOrder, PanelMode>
{
  EIGEN_STRONG_INLINE void operator()(double* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset)
  {
    const int vectorSize = quad_traits<double>::vectorsize;
    Index ri = 0, j = 0;
    for(; j + 2*vectorSize < cols; j+=2*vectorSize)
    {
      Index i = 0;

      if(PanelMode) ri += offset*(2*vectorSize);
      for(; i + vectorSize < depth; i+=vectorSize)
      {
        PacketBlock<Packet2d, 4> block;
        if(StorageOrder == ColMajor)
        {
          PacketBlock<Packet2d, 2> block1, block2;
          block1.packet[0] = rhs.template loadPacket<Packet2d>(i, j + 0);
          block1.packet[1] = rhs.template loadPacket<Packet2d>(i, j + 1);
          block2.packet[0] = rhs.template loadPacket<Packet2d>(i, j + 2);
          block2.packet[1] = rhs.template loadPacket<Packet2d>(i, j + 3);

          ptranspose(block1);
          ptranspose(block2);

          pstore<double>(blockB + ri    , block1.packet[0]);
          pstore<double>(blockB + ri + 2, block2.packet[0]);
          pstore<double>(blockB + ri + 4, block1.packet[1]);
          pstore<double>(blockB + ri + 6, block2.packet[1]);
        } else {
          block.packet[0] = rhs.template loadPacket<Packet2d>(i + 0, j + 0); //[a1 a2]
          block.packet[1] = rhs.template loadPacket<Packet2d>(i + 0, j + 2); //[a3 a4]
          block.packet[2] = rhs.template loadPacket<Packet2d>(i + 1, j + 0); //[b1 b2]
          block.packet[3] = rhs.template loadPacket<Packet2d>(i + 1, j + 2); //[b3 b4]

          pstore<double>(blockB + ri    , block.packet[0]);
          pstore<double>(blockB + ri + 2, block.packet[1]);
          pstore<double>(blockB + ri + 4, block.packet[2]);
          pstore<double>(blockB + ri + 6, block.packet[3]);
        }

        ri += 4*vectorSize;
      }
      for(; i < depth; i++)
      {
        if(StorageOrder == ColMajor)
        {
          blockB[ri+0] = rhs(i, j+0);
          blockB[ri+1] = rhs(i, j+1);

          ri += vectorSize;

          blockB[ri+0] = rhs(i, j+2);
          blockB[ri+1] = rhs(i, j+3);
        } else {
          Packet2d rhsV = rhs.template loadPacket<Packet2d>(i, j);
          pstore<double>(blockB + ri, rhsV);

          ri += vectorSize;

          rhsV = rhs.template loadPacket<Packet2d>(i, j + 2);
          pstore<double>(blockB + ri, rhsV);
        }
        ri += vectorSize;
      }

      if(PanelMode) ri += (2*vectorSize)*(stride - offset - depth);
    }

    if(PanelMode) ri += offset*(cols - j);

    for(Index i = 0; i < depth; i++)
    {
      Index k = j;
      for(; k < cols; k++)
      {
        blockB[ri] = rhs(i, k);
        ri += 1;
      }
    }
    if(PanelMode) ri += (cols - j)*(stride - offset - depth);
  }
};

// General template for lhs complex packing, float64 specialization.
/*
template<bool IsComplex, typename Index, typename DataMapper, typename Packet, typename PacketC, int StorageOrder, bool Conjugate, bool PanelMode>
struct lhs_cpack<double, IsComplex, Index, DataMapper, Packet, PacketC, StorageOrder, Conjugate, PanelMode>
{
  EIGEN_STRONG_INLINE void operator()(std::complex<double>* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride, Index offset)
  {
    const int vectorSize = quad_traits<double>::vectorsize;
    Index ri = 0, j = 0;
    double *blockAt  = reinterpret_cast<double *>(blockA);
    Packet conj = pset1<Packet>((double)-1.0f);

    for(j = 0; j + vectorSize < rows; j+=vectorSize)
    {
      Index i = 0;

      if(PanelMode) ri += vectorSize*offset;

      for(; i + vectorSize < depth; i+=vectorSize)
      {
        PacketBlock<Packet, 2> block;

        PacketBlock<PacketC, 4> cblock;
        if(StorageOrder == ColMajor)
        {
          cblock.packet[0] = lhs.template loadPacket<PacketC>(j, i + 0); //[a1 a1i]
          cblock.packet[1] = lhs.template loadPacket<PacketC>(j, i + 1); //[b1 b1i]

          cblock.packet[2] = lhs.template loadPacket<PacketC>(j + 1, i + 0); //[a2 a2i]
          cblock.packet[3] = lhs.template loadPacket<PacketC>(j + 1, i + 1); //[b2 b2i]

          block.packet[0] = vec_perm(cblock.packet[0].v , cblock.packet[2].v, p16uc_GETREAL64); //[a1 a2]
          block.packet[1] = vec_perm(cblock.packet[1].v , cblock.packet[3].v, p16uc_GETREAL64); //[b1 b2]
        } else {
          cblock.packet[0] = lhs.template loadPacket<PacketC>(j + 0, i); //[a1 a1i]
          cblock.packet[1] = lhs.template loadPacket<PacketC>(j + 1, i); //[a2 a2i]

          cblock.packet[2] = lhs.template loadPacket<PacketC>(j + 0, i + 1); //[b1 b1i]
          cblock.packet[3] = lhs.template loadPacket<PacketC>(j + 1, i + 1); //[b2 b2i]

          block.packet[0] = vec_perm(cblock.packet[0].v , cblock.packet[1].v, p16uc_GETREAL64); //[a1 a2]
          block.packet[1] = vec_perm(cblock.packet[2].v , cblock.packet[3].v, p16uc_GETREAL64); //[b1 b2]
        }

        pstore<double>(blockAt + ri     , block.packet[0]);
        pstore<double>(blockAt + ri +  2, block.packet[1]);

        ri += 2*vectorSize;
      }
      for(; i < depth; i++)
      {
        blockAt[ri + 0] = lhs(j + 0, i).real();
        blockAt[ri + 1] = lhs(j + 1, i).real();
        ri += vectorSize;
      }
      if(PanelMode) ri += vectorSize*(stride - offset - depth);
      
      i = 0;

      if(PanelMode) ri += vectorSize*offset;

      for(; i + vectorSize < depth; i+=vectorSize)
      {
        PacketBlock<Packet, 2> block;

        PacketBlock<PacketC, 4> cblock;
        if(StorageOrder == ColMajor)
        {
          cblock.packet[0] = lhs.template loadPacket<PacketC>(j, i + 0);
          cblock.packet[1] = lhs.template loadPacket<PacketC>(j, i + 1);

          cblock.packet[2] = lhs.template loadPacket<PacketC>(j + 1, i + 0);
          cblock.packet[3] = lhs.template loadPacket<PacketC>(j + 1, i + 1);

          block.packet[0] = vec_perm(cblock.packet[0].v , cblock.packet[2].v, p16uc_GETIMAG64);
          block.packet[1] = vec_perm(cblock.packet[1].v , cblock.packet[3].v, p16uc_GETIMAG64);
        } else {
          cblock.packet[0] = lhs.template loadPacket<PacketC>(j + 0, i);
          cblock.packet[1] = lhs.template loadPacket<PacketC>(j + 1, i);

          cblock.packet[2] = lhs.template loadPacket<PacketC>(j + 0, i + 1);
          cblock.packet[3] = lhs.template loadPacket<PacketC>(j + 1, i + 1);

          block.packet[0] = vec_perm(cblock.packet[0].v , cblock.packet[1].v, p16uc_GETIMAG64);
          block.packet[1] = vec_perm(cblock.packet[2].v , cblock.packet[3].v, p16uc_GETIMAG64);
        }

        if(Conjugate)
        {
          block.packet[0] *= conj;
          block.packet[1] *= conj;
        }

        pstore<double>(blockAt + ri     , block.packet[0]);
        pstore<double>(blockAt + ri +  2, block.packet[1]);

        ri += 2*vectorSize;
      }
      for(; i < depth; i++)
      {
        if(Conjugate)
        {
          blockAt[ri + 0] = -lhs(j + 0, i).imag();
          blockAt[ri + 1] = -lhs(j + 1, i).imag();
        } else {
          blockAt[ri + 0] = lhs(j + 0, i).imag();
          blockAt[ri + 1] = lhs(j + 1, i).imag();
        }

        ri += vectorSize;
      }
      if(PanelMode) ri += vectorSize*(stride - offset - depth);
    }

    if(PanelMode) ri += offset*(rows - j);

    for(Index i = 0; i < depth; i++)
    {
      Index k = j;
      for(; k < rows; k++)
      {
        blockAt[ri] = lhs(k, i).real();
        ri += 1;
      }
    }

    if(PanelMode) ri += (rows - j)*(stride - offset - depth);

    if(PanelMode) ri += offset*(rows - j);

    for(Index i = 0; i < depth; i++)
    {
      Index k = j;
      for(; k < rows; k++)
      {
        if(Conjugate)
          blockAt[ri] = -lhs(k, i).imag();
        else
          blockAt[ri] = lhs(k, i).imag();
        ri += 1;
      }
    }

    if(PanelMode) ri += (rows - j)*(stride - offset - depth);
  }
};

// General template for rhs complex packing, float64 specialization.

template<typename Index, typename DataMapper, typename Packet, typename PacketC, int StorageOrder, bool Conjugate, bool PanelMode>
struct rhs_cpack<double, Index, DataMapper, Packet, PacketC, StorageOrder, Conjugate, PanelMode>
{
  EIGEN_STRONG_INLINE void operator()(std::complex<double>* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset)
  {
    const int vectorSize = quad_traits<double>::vectorsize;
    double *blockBt = reinterpret_cast<double *>(blockB);
    Packet conj = pset1<Packet>((double)-1.0f);

    Index ri = 0, j = 0;
    for(; j + 2*vectorSize < cols; j+=2*vectorSize)
    {
      Index i = 0;

      if(PanelMode) ri += offset*(2*vectorSize);

      for(; i < depth; i++)
      {
        PacketBlock<PacketC, 4> cblock;
        PacketBlock<Packet, 2> block;

        cblock.packet[0] = rhs.template loadPacket<PacketC>(i, j + 0);
        cblock.packet[1] = rhs.template loadPacket<PacketC>(i, j + 1);
        cblock.packet[2] = rhs.template loadPacket<PacketC>(i, j + 2);
        cblock.packet[3] = rhs.template loadPacket<PacketC>(i, j + 3);

        block.packet[0] = vec_perm(cblock.packet[0].v , cblock.packet[1].v, p16uc_GETREAL64);
        block.packet[1] = vec_perm(cblock.packet[2].v , cblock.packet[3].v, p16uc_GETREAL64);

        pstore<double>(blockBt + ri    , block.packet[0]);
        pstore<double>(blockBt + ri + 2, block.packet[1]);

        ri += 2*vectorSize;
      }

      if(PanelMode) ri += (2*vectorSize)*(stride - offset - depth);

      i = 0;

      if(PanelMode) ri += offset*(2*vectorSize);

      for(; i < depth; i++)
      {
        PacketBlock<PacketC, 4> cblock;
        PacketBlock<Packet, 2> block;

        cblock.packet[0] = rhs.template loadPacket<PacketC>(i, j + 0); //[a1 a1i]
        cblock.packet[1] = rhs.template loadPacket<PacketC>(i, j + 1); //[b1 b1i]
        cblock.packet[2] = rhs.template loadPacket<PacketC>(i, j + 2); //[c1 c1i]
        cblock.packet[3] = rhs.template loadPacket<PacketC>(i, j + 3); //[d1 d1i]

        block.packet[0] = vec_perm(cblock.packet[0].v , cblock.packet[1].v, p16uc_GETIMAG64);
        block.packet[1] = vec_perm(cblock.packet[2].v , cblock.packet[3].v, p16uc_GETIMAG64);

        if(Conjugate)
        {
          block.packet[0] *= conj;
          block.packet[1] *= conj;
        }

        pstore<double>(blockBt + ri     , block.packet[0]);
        pstore<double>(blockBt + ri +  2, block.packet[1]);

        ri += 2*vectorSize;
      }
      if(PanelMode) ri += (2*vectorSize)*(stride - offset - depth);
    }

    if(PanelMode) ri += offset*(cols - j);

    for(Index i = 0; i < depth; i++)
    {
      Index k = j;
      for(; k < cols; k++)
      {
        blockBt[ri] = rhs(i, k).real();
        ri += 1;
      }
    }
    if(PanelMode) ri += (cols - j)*(stride - offset - depth);

    if(PanelMode) ri += offset*(cols - j);

    for(Index i = 0; i < depth; i++)
    {
      Index k = j;
      for(; k < cols; k++)
      {
        if(Conjugate)
          blockBt[ri] = -rhs(i, k).imag();
        else
          blockBt[ri] = rhs(i, k).imag();
        ri += 1;
      }
    }
    if(PanelMode) ri += (cols - j)*(stride - offset - depth);
  }
};
*/
/**************
 * GEMM utils *
 **************/
/*
// Grab two decouples real/imaginary PacketBlocks and return two coupled (real/imaginary pairs) PacketBlocks.
template<typename Packet, typename Packetc>
EIGEN_STRONG_INLINE void bcouple(PacketBlock<Packet,4>& taccReal, PacketBlock<Packet,4>& taccImag, PacketBlock<Packetc,8>& tRes, PacketBlock<Packetc, 4>& acc1, PacketBlock<Packetc, 4>& acc2)
{
  acc1.packet[0].v = vec_perm(taccReal.packet[0], taccImag.packet[0], p16uc_SETCOMPLEX32_FIRST);
  acc1.packet[1].v = vec_perm(taccReal.packet[1], taccImag.packet[1], p16uc_SETCOMPLEX32_FIRST);
  acc1.packet[2].v = vec_perm(taccReal.packet[2], taccImag.packet[2], p16uc_SETCOMPLEX32_FIRST);
  acc1.packet[3].v = vec_perm(taccReal.packet[3], taccImag.packet[3], p16uc_SETCOMPLEX32_FIRST);

  acc2.packet[0].v = vec_perm(taccReal.packet[0], taccImag.packet[0], p16uc_SETCOMPLEX32_SECOND);
  acc2.packet[1].v = vec_perm(taccReal.packet[1], taccImag.packet[1], p16uc_SETCOMPLEX32_SECOND);
  acc2.packet[2].v = vec_perm(taccReal.packet[2], taccImag.packet[2], p16uc_SETCOMPLEX32_SECOND);
  acc2.packet[3].v = vec_perm(taccReal.packet[3], taccImag.packet[3], p16uc_SETCOMPLEX32_SECOND);

  acc1.packet[0] = padd<Packetc>(tRes.packet[0], acc1.packet[0]);
  acc1.packet[1] = padd<Packetc>(tRes.packet[1], acc1.packet[1]);
  acc1.packet[2] = padd<Packetc>(tRes.packet[2], acc1.packet[2]);
  acc1.packet[3] = padd<Packetc>(tRes.packet[3], acc1.packet[3]);

  acc2.packet[0] = padd<Packetc>(tRes.packet[4], acc2.packet[0]);
  acc2.packet[1] = padd<Packetc>(tRes.packet[5], acc2.packet[1]);
  acc2.packet[2] = padd<Packetc>(tRes.packet[6], acc2.packet[2]);
  acc2.packet[3] = padd<Packetc>(tRes.packet[7], acc2.packet[3]);
}

template<>
EIGEN_STRONG_INLINE void bcouple<Packet2d, Packet1cd>(PacketBlock<Packet2d,4>& taccReal, PacketBlock<Packet2d,4>& taccImag, PacketBlock<Packet1cd,8>& tRes, PacketBlock<Packet1cd, 4>& acc1, PacketBlock<Packet1cd, 4>& acc2)
{
  acc1.packet[0].v = vec_perm(taccReal.packet[0], taccImag.packet[0], p16uc_SETCOMPLEX64_FIRST);
  acc1.packet[1].v = vec_perm(taccReal.packet[1], taccImag.packet[1], p16uc_SETCOMPLEX64_FIRST);
  acc1.packet[2].v = vec_perm(taccReal.packet[2], taccImag.packet[2], p16uc_SETCOMPLEX64_FIRST);
  acc1.packet[3].v = vec_perm(taccReal.packet[3], taccImag.packet[3], p16uc_SETCOMPLEX64_FIRST);

  acc2.packet[0].v = vec_perm(taccReal.packet[0], taccImag.packet[0], p16uc_SETCOMPLEX64_SECOND);
  acc2.packet[1].v = vec_perm(taccReal.packet[1], taccImag.packet[1], p16uc_SETCOMPLEX64_SECOND);
  acc2.packet[2].v = vec_perm(taccReal.packet[2], taccImag.packet[2], p16uc_SETCOMPLEX64_SECOND);
  acc2.packet[3].v = vec_perm(taccReal.packet[3], taccImag.packet[3], p16uc_SETCOMPLEX64_SECOND);

  acc1.packet[0] = padd<Packet1cd>(tRes.packet[0], acc1.packet[0]);
  acc1.packet[1] = padd<Packet1cd>(tRes.packet[1], acc1.packet[1]);
  acc1.packet[2] = padd<Packet1cd>(tRes.packet[2], acc1.packet[2]);
  acc1.packet[3] = padd<Packet1cd>(tRes.packet[3], acc1.packet[3]);

  acc2.packet[0] = padd<Packet1cd>(tRes.packet[4], acc2.packet[0]);
  acc2.packet[1] = padd<Packet1cd>(tRes.packet[5], acc2.packet[1]);
  acc2.packet[2] = padd<Packet1cd>(tRes.packet[6], acc2.packet[2]);
  acc2.packet[3] = padd<Packet1cd>(tRes.packet[7], acc2.packet[3]);
}
*/
// 512-bits rank1-update of acc. It can either positive or negative accumulate (useful for complex gemm).
template<typename Scalar, typename Packet, bool NegativeAccumulate>
EIGEN_STRONG_INLINE void pger(PacketBlock<Packet, 4> *acc, const Scalar* lhs, const Scalar* rhs)
{
  Packet lhsV = *((Packet *) lhs);
  Packet rhsV1 = pset1<Packet>(rhs[0]);
  Packet rhsV2 = pset1<Packet>(rhs[1]);
  Packet rhsV3 = pset1<Packet>(rhs[2]);
  Packet rhsV4 = pset1<Packet>(rhs[3]);

  if(NegativeAccumulate)
  {
    acc->packet[0] -= lhsV*rhsV1;
    acc->packet[1] -= lhsV*rhsV2;
    acc->packet[2] -= lhsV*rhsV3;
    acc->packet[3] -= lhsV*rhsV4;
  } else {
    acc->packet[0] += lhsV*rhsV1;
    acc->packet[1] += lhsV*rhsV2;
    acc->packet[2] += lhsV*rhsV3;
    acc->packet[3] += lhsV*rhsV4;
  }
}

// 512-bits rank1-update of complex acc. It takes decoupled accumulators as entries. It also takes cares of mixed types real * complex and complex * real.
/*
template<typename Scalar, typename Packet, bool ConjugateLhs, bool ConjugateRhs, bool LhsIsReal, bool RhsIsReal>
EIGEN_STRONG_INLINE void pgerc(PacketBlock<Packet, 4>& accReal, PacketBlock<Packet,4>& accImag, const Scalar *rhs_ptr, const Scalar *rhs_ptr_imag, const Scalar *lhs_ptr, const Scalar* lhs_ptr_imag, Packet& conj)
{
  Packet lhsV  = *((Packet *) lhs_ptr);
  Packet rhsV1 = pset1<Packet>(rhs_ptr[0]);
  Packet rhsV2 = pset1<Packet>(rhs_ptr[1]);
  Packet rhsV3 = pset1<Packet>(rhs_ptr[2]);
  Packet rhsV4 = pset1<Packet>(rhs_ptr[3]);

  Packet lhsVi;
  if(!LhsIsReal) lhsVi = *((Packet *) lhs_ptr_imag);
  Packet rhsV1i, rhsV2i, rhsV3i, rhsV4i;
  if(!RhsIsReal)
  {
    rhsV1i = pset1<Packet>(rhs_ptr_imag[0]);
    rhsV2i = pset1<Packet>(rhs_ptr_imag[1]);
    rhsV3i = pset1<Packet>(rhs_ptr_imag[2]);
    rhsV4i = pset1<Packet>(rhs_ptr_imag[3]);
  }

  if(ConjugateLhs && !LhsIsReal) lhsVi = pmul<Packet>(lhsVi,conj);
  if(ConjugateRhs && !RhsIsReal)
  {
    rhsV1i = pmul<Packet>(rhsV1i,conj);
    rhsV2i = pmul<Packet>(rhsV2i,conj);
    rhsV3i = pmul<Packet>(rhsV3i,conj);
    rhsV4i = pmul<Packet>(rhsV4i,conj);
  }

  if(LhsIsReal)
  {
    accReal.packet[0] = pmadd<Packet>(rhsV1, lhsV, accReal.packet[0]);
    accReal.packet[1] = pmadd<Packet>(rhsV2, lhsV, accReal.packet[1]);
    accReal.packet[2] = pmadd<Packet>(rhsV3, lhsV, accReal.packet[2]);
    accReal.packet[3] = pmadd<Packet>(rhsV4, lhsV, accReal.packet[3]);

    accImag.packet[0] = pmadd<Packet>(rhsV1i, lhsV, accImag.packet[0]);
    accImag.packet[1] = pmadd<Packet>(rhsV2i, lhsV, accImag.packet[1]);
    accImag.packet[2] = pmadd<Packet>(rhsV3i, lhsV, accImag.packet[2]);
    accImag.packet[3] = pmadd<Packet>(rhsV4i, lhsV, accImag.packet[3]);
  } else if(RhsIsReal) {
    accReal.packet[0] = pmadd<Packet>(rhsV1, lhsV, accReal.packet[0]);
    accReal.packet[1] = pmadd<Packet>(rhsV2, lhsV, accReal.packet[1]);
    accReal.packet[2] = pmadd<Packet>(rhsV3, lhsV, accReal.packet[2]);
    accReal.packet[3] = pmadd<Packet>(rhsV4, lhsV, accReal.packet[3]);

    accImag.packet[0] = pmadd<Packet>(rhsV1, lhsVi, accImag.packet[0]);
    accImag.packet[1] = pmadd<Packet>(rhsV2, lhsVi, accImag.packet[1]);
    accImag.packet[2] = pmadd<Packet>(rhsV3, lhsVi, accImag.packet[2]);
    accImag.packet[3] = pmadd<Packet>(rhsV4, lhsVi, accImag.packet[3]);
  } else {
    accReal.packet[0] = pmadd<Packet>(rhsV1, lhsV, accReal.packet[0]);
    accReal.packet[1] = pmadd<Packet>(rhsV2, lhsV, accReal.packet[1]);
    accReal.packet[2] = pmadd<Packet>(rhsV3, lhsV, accReal.packet[2]);
    accReal.packet[3] = pmadd<Packet>(rhsV4, lhsV, accReal.packet[3]);

    accImag.packet[0] = pmadd<Packet>(rhsV1i, lhsV, accImag.packet[0]);
    accImag.packet[1] = pmadd<Packet>(rhsV2i, lhsV, accImag.packet[1]);
    accImag.packet[2] = pmadd<Packet>(rhsV3i, lhsV, accImag.packet[2]);
    accImag.packet[3] = pmadd<Packet>(rhsV4i, lhsV, accImag.packet[3]);

    accReal.packet[0] = psub<Packet>(accReal.packet[0], pmul<Packet>(rhsV1i, lhsVi));
    accReal.packet[1] = psub<Packet>(accReal.packet[1], pmul<Packet>(rhsV2i, lhsVi));
    accReal.packet[2] = psub<Packet>(accReal.packet[2], pmul<Packet>(rhsV3i, lhsVi));
    accReal.packet[3] = psub<Packet>(accReal.packet[3], pmul<Packet>(rhsV4i, lhsVi));

    accImag.packet[0] = pmadd<Packet>(rhsV1, lhsVi, accImag.packet[0]);
    accImag.packet[1] = pmadd<Packet>(rhsV2, lhsVi, accImag.packet[1]);
    accImag.packet[2] = pmadd<Packet>(rhsV3, lhsVi, accImag.packet[2]);
    accImag.packet[3] = pmadd<Packet>(rhsV4, lhsVi, accImag.packet[3]);
  }
}
*/

// This is necessary because ploadRhs for double returns a pair of vectors when MMA is enabled.
template<typename Scalar, typename Packet>
EIGEN_STRONG_INLINE Packet ploadRhs(const Scalar *rhs)
{
    return *((Packet *)rhs);
}

template<typename Scalar, typename Packet>
EIGEN_STRONG_INLINE Packet ploadLhs(const Scalar *lhs)
{
    return *((Packet *)lhs);
}

// Zero the accumulator on PacketBlock.
template<typename Scalar, typename Packet>
EIGEN_STRONG_INLINE void bsetzero(PacketBlock<Packet,4>& acc)
{
  acc.packet[0] = pset1<Packet>((Scalar)0);
  acc.packet[1] = pset1<Packet>((Scalar)0);
  acc.packet[2] = pset1<Packet>((Scalar)0);
  acc.packet[3] = pset1<Packet>((Scalar)0);
}

// Scale the PacketBlock vectors by alpha.
template<typename Packet>
EIGEN_STRONG_INLINE void bscale(PacketBlock<Packet,4>& acc, PacketBlock<Packet,4>& accZ, const Packet& pAlpha)
{
  acc.packet[0] = pmadd(pAlpha,accZ.packet[0], acc.packet[0]);
  acc.packet[1] = pmadd(pAlpha,accZ.packet[1], acc.packet[1]);
  acc.packet[2] = pmadd(pAlpha,accZ.packet[2], acc.packet[2]);
  acc.packet[3] = pmadd(pAlpha,accZ.packet[3], acc.packet[3]);
}

// Complex version of PacketBlock scaling.
/*
template<typename Packet>
EIGEN_STRONG_INLINE void bscalec(PacketBlock<Packet,4>& aReal, PacketBlock<Packet,4>& aImag, const Packet& bReal, const Packet& bImag, PacketBlock<Packet,4>& cReal, PacketBlock<Packet,4>& cImag)
{
  cReal.packet[0] = pmul<Packet>(aReal.packet[0], bReal);
  cReal.packet[1] = pmul<Packet>(aReal.packet[1], bReal);
  cReal.packet[2] = pmul<Packet>(aReal.packet[2], bReal);
  cReal.packet[3] = pmul<Packet>(aReal.packet[3], bReal);

  cImag.packet[0] = pmul<Packet>(aImag.packet[0], bReal);
  cImag.packet[1] = pmul<Packet>(aImag.packet[1], bReal);
  cImag.packet[2] = pmul<Packet>(aImag.packet[2], bReal);
  cImag.packet[3] = pmul<Packet>(aImag.packet[3], bReal);

  cReal.packet[0] = psub<Packet>(cReal.packet[0], pmul<Packet>(aImag.packet[0], bImag));
  cReal.packet[1] = psub<Packet>(cReal.packet[1], pmul<Packet>(aImag.packet[1], bImag));
  cReal.packet[2] = psub<Packet>(cReal.packet[2], pmul<Packet>(aImag.packet[2], bImag));
  cReal.packet[3] = psub<Packet>(cReal.packet[3], pmul<Packet>(aImag.packet[3], bImag));

  cImag.packet[0] = pmadd<Packet>(aReal.packet[0], bImag, cImag.packet[0]);
  cImag.packet[1] = pmadd<Packet>(aReal.packet[1], bImag, cImag.packet[1]);
  cImag.packet[2] = pmadd<Packet>(aReal.packet[2], bImag, cImag.packet[2]);
  cImag.packet[3] = pmadd<Packet>(aReal.packet[3], bImag, cImag.packet[3]);
}
*/

// Load a PacketBlock, the N parameters make tunning gemm easier so we can add more accumulators as needed.
template<typename DataMapper, typename Packet, typename Index, int N>
EIGEN_STRONG_INLINE void bload(PacketBlock<Packet,4>& acc, const DataMapper& res, Index row, Index col, Index accCols)
{
  acc.packet[0] = res.template loadPacket<Packet>(row + N*accCols, col + 0);
  acc.packet[1] = res.template loadPacket<Packet>(row + N*accCols, col + 1);
  acc.packet[2] = res.template loadPacket<Packet>(row + N*accCols, col + 2);
  acc.packet[3] = res.template loadPacket<Packet>(row + N*accCols, col + 3);
}

// An overload of bload when you have a PacketBLock with 8 vectors.
template<typename DataMapper, typename Packet, typename Index, int N>
EIGEN_STRONG_INLINE void bload(PacketBlock<Packet,8>& acc, const DataMapper& res, Index row, Index col, Index accCols)
{
  acc.packet[0] = res.template loadPacket<Packet>(row + N*accCols, col + 0);
  acc.packet[1] = res.template loadPacket<Packet>(row + N*accCols, col + 1);
  acc.packet[2] = res.template loadPacket<Packet>(row + N*accCols, col + 2);
  acc.packet[3] = res.template loadPacket<Packet>(row + N*accCols, col + 3);
  acc.packet[4] = res.template loadPacket<Packet>(row + (N+1)*accCols, col + 0);
  acc.packet[5] = res.template loadPacket<Packet>(row + (N+1)*accCols, col + 1);
  acc.packet[6] = res.template loadPacket<Packet>(row + (N+1)*accCols, col + 2);
  acc.packet[7] = res.template loadPacket<Packet>(row + (N+1)*accCols, col + 3);
}

// PEEL loop factor.
#define PEEL 10

/****************
 * GEMM kernels *
 * **************/
template<typename Scalar, typename Index, typename Packet, typename RhsPacket, typename DataMapper>
EIGEN_STRONG_INLINE void gemm(const DataMapper& res, const Scalar* blockA, const Scalar* blockB,
          Index rows, Index depth, Index cols, Scalar alpha, Index strideA, Index strideB, Index offsetA, Index offsetB, const int accRows, const int accCols)
{
      const Index remaining_rows = rows % accCols;
      const Index remaining_cols = cols % accRows;

      if( strideA == -1 ) strideA = depth;
      if( strideB == -1 ) strideB = depth;

      const Packet pAlpha = pset1<Packet>(alpha);
      Index col = 0;
      for(; col + accRows <= cols; col += accRows)
      {
        const Scalar *rhs_base = blockB + ( col/accRows     )*strideB*accRows;
        const Scalar *lhs_base = blockA;

        Index row = 0;
        for(; row + 6*accCols <= rows; row += 6*accCols)
        {
#define MICRO() \
            pger<Scalar, Packet, false>(&accZero1, lhs_ptr1, rhs_ptr); \
            lhs_ptr1 += accCols; \
            pger<Scalar, Packet, false>(&accZero2, lhs_ptr2, rhs_ptr); \
            lhs_ptr2 += accCols; \
            pger<Scalar, Packet, false>(&accZero3, lhs_ptr3, rhs_ptr); \
            lhs_ptr3 += accCols; \
            pger<Scalar, Packet, false>(&accZero4, lhs_ptr4, rhs_ptr); \
            lhs_ptr4 += accCols; \
            pger<Scalar, Packet, false>(&accZero5, lhs_ptr5, rhs_ptr); \
            lhs_ptr5 += accCols; \
            pger<Scalar, Packet, false>(&accZero6, lhs_ptr6, rhs_ptr); \
            lhs_ptr6 += accCols; \
            rhs_ptr += accRows;

          const Scalar *rhs_ptr  = rhs_base;
          const Scalar *lhs_ptr1 = lhs_base + (row/accCols)*strideA*accCols;
          const Scalar *lhs_ptr2 = lhs_base + ((row/accCols) + 1)*strideA*accCols;
          const Scalar *lhs_ptr3 = lhs_base + ((row/accCols) + 2)*strideA*accCols;
          const Scalar *lhs_ptr4 = lhs_base + ((row/accCols) + 3)*strideA*accCols;
          const Scalar *lhs_ptr5 = lhs_base + ((row/accCols) + 4)*strideA*accCols;
          const Scalar *lhs_ptr6 = lhs_base + ((row/accCols) + 5)*strideA*accCols;

          PacketBlock<Packet,4> acc1, accZero1;
          PacketBlock<Packet,4> acc2, accZero2;
          PacketBlock<Packet,4> acc3, accZero3;
          PacketBlock<Packet,4> acc4, accZero4;
          PacketBlock<Packet,4> acc5, accZero5;
          PacketBlock<Packet,4> acc6, accZero6;

          bload<DataMapper, Packet, Index, 0>(acc1, res, row, col, accCols);
          bsetzero<Scalar, Packet>(accZero1);
          bload<DataMapper, Packet, Index, 1>(acc2, res, row, col, accCols);
          bsetzero<Scalar, Packet>(accZero2);
          bload<DataMapper, Packet, Index, 2>(acc3, res, row, col, accCols);
          bsetzero<Scalar, Packet>(accZero3);
          bload<DataMapper, Packet, Index, 3>(acc4, res, row, col, accCols);
          bsetzero<Scalar, Packet>(accZero4);
          bload<DataMapper, Packet, Index, 4>(acc5, res, row, col, accCols);
          bsetzero<Scalar, Packet>(accZero5);
          bload<DataMapper, Packet, Index, 5>(acc6, res, row, col, accCols);
          bsetzero<Scalar, Packet>(accZero6);

          lhs_ptr1 += accCols*offsetA;
          lhs_ptr2 += accCols*offsetA;
          lhs_ptr3 += accCols*offsetA;
          lhs_ptr4 += accCols*offsetA;
          lhs_ptr5 += accCols*offsetA;
          lhs_ptr6 += accCols*offsetA;
          rhs_ptr += accRows*offsetB;

          Index k = 0;
          for(; k + PEEL < depth; k+= PEEL)
          {
            prefetch(rhs_ptr);
            prefetch(lhs_ptr1);
            prefetch(lhs_ptr2);
            prefetch(lhs_ptr3);
            prefetch(lhs_ptr4);
            prefetch(lhs_ptr5);
            prefetch(lhs_ptr6);
            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
#if PEEL > 8
            MICRO();
            MICRO();
#endif
          }
          for(; k < depth; k++)
          {
            MICRO();
          }

          bscale<Packet>(acc1,accZero1, pAlpha);
          bscale<Packet>(acc2,accZero2, pAlpha);
          bscale<Packet>(acc3,accZero3, pAlpha);
          bscale<Packet>(acc4,accZero4, pAlpha);
          bscale<Packet>(acc5,accZero5, pAlpha);
          bscale<Packet>(acc6,accZero6, pAlpha);

          res.template storePacketBlock<Packet, 4>(row + 0*accCols, col, acc1);
          res.template storePacketBlock<Packet, 4>(row + 1*accCols, col, acc2);
          res.template storePacketBlock<Packet, 4>(row + 2*accCols, col, acc3);
          res.template storePacketBlock<Packet, 4>(row + 3*accCols, col, acc4);
          res.template storePacketBlock<Packet, 4>(row + 4*accCols, col, acc5);
          res.template storePacketBlock<Packet, 4>(row + 5*accCols, col, acc6);
#undef MICRO
        }
        for(; row + 5*accCols <= rows; row += 5*accCols)
        {
#define MICRO() \
            pger<Scalar, Packet, false>(&accZero1, lhs_ptr1, rhs_ptr); \
            lhs_ptr1 += accCols; \
            pger<Scalar, Packet, false>(&accZero2, lhs_ptr2, rhs_ptr); \
            lhs_ptr2 += accCols; \
            pger<Scalar, Packet, false>(&accZero3, lhs_ptr3, rhs_ptr); \
            lhs_ptr3 += accCols; \
            pger<Scalar, Packet, false>(&accZero4, lhs_ptr4, rhs_ptr); \
            lhs_ptr4 += accCols; \
            pger<Scalar, Packet, false>(&accZero5, lhs_ptr5, rhs_ptr); \
            lhs_ptr5 += accCols; \
            rhs_ptr += accRows;

          const Scalar *rhs_ptr  = rhs_base;
          const Scalar *lhs_ptr1 = lhs_base + (row/accCols      )*strideA*accCols;
          const Scalar *lhs_ptr2 = lhs_base + ((row/accCols) + 1)*strideA*accCols;
          const Scalar *lhs_ptr3 = lhs_base + ((row/accCols) + 2)*strideA*accCols;
          const Scalar *lhs_ptr4 = lhs_base + ((row/accCols) + 3)*strideA*accCols;
          const Scalar *lhs_ptr5 = lhs_base + ((row/accCols) + 4)*strideA*accCols;

          PacketBlock<Packet,4> acc1, accZero1;
          PacketBlock<Packet,4> acc2, accZero2;
          PacketBlock<Packet,4> acc3, accZero3;
          PacketBlock<Packet,4> acc4, accZero4;
          PacketBlock<Packet,4> acc5, accZero5;

          bload<DataMapper, Packet, Index, 0>(acc1, res, row, col, accCols);
          bsetzero<Scalar, Packet>(accZero1);
          bload<DataMapper, Packet, Index, 1>(acc2, res, row, col, accCols);
          bsetzero<Scalar, Packet>(accZero2);
          bload<DataMapper, Packet, Index, 2>(acc3, res, row, col, accCols);
          bsetzero<Scalar, Packet>(accZero3);
          bload<DataMapper, Packet, Index, 3>(acc4, res, row, col, accCols);
          bsetzero<Scalar, Packet>(accZero4);
          bload<DataMapper, Packet, Index, 4>(acc5, res, row, col, accCols);
          bsetzero<Scalar, Packet>(accZero5);

          lhs_ptr1 += accCols*offsetA;
          lhs_ptr2 += accCols*offsetA;
          lhs_ptr3 += accCols*offsetA;
          lhs_ptr4 += accCols*offsetA;
          lhs_ptr5 += accCols*offsetA;
          rhs_ptr += accRows*offsetB;
          Index k = 0;

          for(; k + PEEL < depth; k+= PEEL)
          {
            prefetch(rhs_ptr);
            prefetch(lhs_ptr1);
            prefetch(lhs_ptr2);
            prefetch(lhs_ptr3);
            prefetch(lhs_ptr4);
            prefetch(lhs_ptr5);

            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
#if PEEL > 8
            MICRO();
            MICRO();
#endif
          }
          for(; k < depth; k++)
          {
            MICRO();
          }

          bscale<Packet>(acc1,accZero1, pAlpha);
          bscale<Packet>(acc2,accZero2, pAlpha);
          bscale<Packet>(acc3,accZero3, pAlpha);
          bscale<Packet>(acc4,accZero4, pAlpha);
          bscale<Packet>(acc5,accZero5, pAlpha);

          res.template storePacketBlock<Packet, 4>(row + 0*accCols, col, acc1);
          res.template storePacketBlock<Packet, 4>(row + 1*accCols, col, acc2);
          res.template storePacketBlock<Packet, 4>(row + 2*accCols, col, acc3);
          res.template storePacketBlock<Packet, 4>(row + 3*accCols, col, acc4);
          res.template storePacketBlock<Packet, 4>(row + 4*accCols, col, acc5);
#undef MICRO
        }
        for(; row + 4*accCols <= rows; row += 4*accCols)
        {
#define MICRO() \
            pger<Scalar, Packet, false>(&accZero1, lhs_ptr1, rhs_ptr); \
            lhs_ptr1 += accCols; \
            pger<Scalar, Packet, false>(&accZero2, lhs_ptr2, rhs_ptr); \
            lhs_ptr2 += accCols; \
            pger<Scalar, Packet, false>(&accZero3, lhs_ptr3, rhs_ptr); \
            lhs_ptr3 += accCols; \
            pger<Scalar, Packet, false>(&accZero4, lhs_ptr4, rhs_ptr); \
            lhs_ptr4 += accCols; \
            rhs_ptr += accRows;

          const Scalar *rhs_ptr  = rhs_base;
          const Scalar *lhs_ptr1 = lhs_base + (row/accCols      )*strideA*accCols;
          const Scalar *lhs_ptr2 = lhs_base + ((row/accCols) + 1)*strideA*accCols;
          const Scalar *lhs_ptr3 = lhs_base + ((row/accCols) + 2)*strideA*accCols;
          const Scalar *lhs_ptr4 = lhs_base + ((row/accCols) + 3)*strideA*accCols;

          PacketBlock<Packet,4> acc1, accZero1;
          PacketBlock<Packet,4> acc2, accZero2;
          PacketBlock<Packet,4> acc3, accZero3;
          PacketBlock<Packet,4> acc4, accZero4;

          bload<DataMapper, Packet, Index, 0>(acc1, res, row, col, accCols);
          bsetzero<Scalar, Packet>(accZero1);
          bload<DataMapper, Packet, Index, 1>(acc2, res, row, col, accCols);
          bsetzero<Scalar, Packet>(accZero2);
          bload<DataMapper, Packet, Index, 2>(acc3, res, row, col, accCols);
          bsetzero<Scalar, Packet>(accZero3);
          bload<DataMapper, Packet, Index, 3>(acc4, res, row, col, accCols);
          bsetzero<Scalar, Packet>(accZero4);

          lhs_ptr1 += accCols*offsetA;
          lhs_ptr2 += accCols*offsetA;
          lhs_ptr3 += accCols*offsetA;
          lhs_ptr4 += accCols*offsetA;
          rhs_ptr += accRows*offsetB;
          Index k = 0;

          for(; k + PEEL < depth; k+= PEEL)
          {
            prefetch(rhs_ptr);
            prefetch(lhs_ptr1);
            prefetch(lhs_ptr2);
            prefetch(lhs_ptr3);
            prefetch(lhs_ptr4);
            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
#if PEEL > 8
            MICRO();
            MICRO();
#endif
          }
          for(; k < depth; k++)
          {
            MICRO();
          }

          bscale<Packet>(acc1,accZero1, pAlpha);
          bscale<Packet>(acc2,accZero2, pAlpha);
          bscale<Packet>(acc3,accZero3, pAlpha);
          bscale<Packet>(acc4,accZero4, pAlpha);

          res.template storePacketBlock<Packet, 4>(row + 0*accCols, col, acc1);
          res.template storePacketBlock<Packet, 4>(row + 1*accCols, col, acc2);
          res.template storePacketBlock<Packet, 4>(row + 2*accCols, col, acc3);
          res.template storePacketBlock<Packet, 4>(row + 3*accCols, col, acc4);
#undef MICRO
        }
        for(; row + 3*accCols <= rows; row += 3*accCols)
        {
#define MICRO() \
            pger<Scalar, Packet, false>(&accZero1, lhs_ptr1, rhs_ptr); \
            lhs_ptr1 += accCols; \
            pger<Scalar, Packet, false>(&accZero2, lhs_ptr2, rhs_ptr); \
            lhs_ptr2 += accCols; \
            pger<Scalar, Packet, false>(&accZero3, lhs_ptr3, rhs_ptr); \
            lhs_ptr3 += accCols; \
            rhs_ptr += accRows;

          const Scalar *rhs_ptr  = rhs_base;
          const Scalar *lhs_ptr1 = lhs_base + (row/accCols      )*strideA*accCols;
          const Scalar *lhs_ptr2 = lhs_base + ((row/accCols) + 1)*strideA*accCols;
          const Scalar *lhs_ptr3 = lhs_base + ((row/accCols) + 2)*strideA*accCols;

          PacketBlock<Packet,4> acc1, accZero1;
          PacketBlock<Packet,4> acc2, accZero2;
          PacketBlock<Packet,4> acc3, accZero3;

          bload<DataMapper, Packet, Index, 0>(acc1, res, row, col, accCols);
          bsetzero<Scalar, Packet>(accZero1);
          bload<DataMapper, Packet, Index, 1>(acc2, res, row, col, accCols);
          bsetzero<Scalar, Packet>(accZero2);
          bload<DataMapper, Packet, Index, 2>(acc3, res, row, col, accCols);
          bsetzero<Scalar, Packet>(accZero3);

          lhs_ptr1 += accCols*offsetA;
          lhs_ptr2 += accCols*offsetA;
          lhs_ptr3 += accCols*offsetA;
          rhs_ptr += accRows*offsetB;
          Index k = 0;
          for(; k + PEEL < depth; k+= PEEL)
          {
            prefetch(rhs_ptr);
            prefetch(lhs_ptr1);
            prefetch(lhs_ptr2);
            prefetch(lhs_ptr3);

            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
#if PEEL > 8
            MICRO();
            MICRO();
#endif
          }
          for(; k < depth; k++)
          {
            MICRO();
          }

          bscale<Packet>(acc1,accZero1, pAlpha);
          bscale<Packet>(acc2,accZero2, pAlpha);
          bscale<Packet>(acc3,accZero3, pAlpha);

          res.template storePacketBlock<Packet, 4>(row + 0*accCols, col, acc1);
          res.template storePacketBlock<Packet, 4>(row + 1*accCols, col, acc2);
          res.template storePacketBlock<Packet, 4>(row + 2*accCols, col, acc3);
#undef MICRO
        }
        for(; row + 2*accCols <= rows; row += 2*accCols)
        {
#define MICRO() \
            pger<Scalar, Packet, false>(&accZero1, lhs_ptr1, rhs_ptr); \
            lhs_ptr1 += accCols; \
            pger<Scalar, Packet, false>(&accZero2, lhs_ptr2, rhs_ptr); \
            lhs_ptr2 += accCols; \
            rhs_ptr += accRows;

          const Scalar *rhs_ptr  = rhs_base;
          const Scalar *lhs_ptr1 = lhs_base + (row/accCols      )*strideA*accCols;
          const Scalar *lhs_ptr2 = lhs_base + ((row/accCols) + 1)*strideA*accCols;
          
          PacketBlock<Packet,4> acc1, accZero1;
          PacketBlock<Packet,4> acc2, accZero2;

          bload<DataMapper, Packet, Index, 0>(acc1, res, row, col, accCols);
          bsetzero<Scalar, Packet>(accZero1);
          bload<DataMapper, Packet, Index, 1>(acc2, res, row, col, accCols);
          bsetzero<Scalar, Packet>(accZero2);

          lhs_ptr1 += accCols*offsetA;
          lhs_ptr2 += accCols*offsetA;
          rhs_ptr += accRows*offsetB;
          Index k = 0;
          for(; k + PEEL < depth; k+= PEEL)
          {
            prefetch(rhs_ptr);
            prefetch(lhs_ptr1);
            prefetch(lhs_ptr2);

            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
#if PEEL > 8
            MICRO();
            MICRO();
#endif
          }
          for(; k < depth; k++)
          {
            MICRO();
          }

          bscale<Packet>(acc1,accZero1, pAlpha);
          bscale<Packet>(acc2,accZero2, pAlpha);

          res.template storePacketBlock<Packet, 4>(row + 0*accCols, col, acc1);
          res.template storePacketBlock<Packet, 4>(row + 1*accCols, col, acc2);
#undef MICRO
        }

        for(; row + accCols <= rows; row += accCols)
        {
#define MICRO() \
            pger<Scalar, Packet, false>(&accZero1, lhs_ptr1, rhs_ptr); \
            lhs_ptr1 += accCols; \
            rhs_ptr += accRows;

          const Scalar *rhs_ptr  = rhs_base;
          const Scalar *lhs_ptr1 = lhs_base + (row/accCols)*strideA*accCols;

          PacketBlock<Packet,4> acc1, accZero1;

          bload<DataMapper, Packet, Index, 0>(acc1, res, row, col, accCols);
          bsetzero<Scalar, Packet>(accZero1);

          lhs_ptr1 += accCols*offsetA;
          rhs_ptr += accRows*offsetB;
          Index k = 0;
          for(; k + PEEL < depth; k+= PEEL)
          {
            prefetch(rhs_ptr);
            prefetch(lhs_ptr1);

            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
#if PEEL > 8
            MICRO();
            MICRO();
#endif
          }
          for(; k < depth; k++)
          {
            MICRO();
          }

          bscale<Packet>(acc1,accZero1, pAlpha);

          res.template storePacketBlock<Packet, 4>(row, col, acc1);
#undef MICRO
        }

        if(remaining_rows > 0)
        {
          const Scalar *rhs_ptr  = rhs_base;
          const Scalar *lhs_ptr = lhs_base + (row/accCols)*strideA*accCols;

          lhs_ptr += remaining_rows*offsetA;
          rhs_ptr += accRows*offsetB;
          for(Index k = 0; k < depth; k++)
          {
              for(Index arow = 0; arow < remaining_rows; arow++)
              {
                  for(Index acol = 0; acol < accRows; acol++ )
                  {
                    res(row + arow, col + acol) += alpha*lhs_ptr[arow]*rhs_ptr[acol];
                  }
              }
              rhs_ptr += accRows;
              lhs_ptr += remaining_rows;
          }
        }
    }

    if(remaining_cols > 0)
    {
      const Scalar *rhs_base = blockB + (col/accRows)*strideB*accRows;
      const Scalar *lhs_base = blockA;

      Index row = 0;
      for(; row + accCols <= rows; row += accCols)
      {
        const Scalar *rhs_ptr = rhs_base;
        const Scalar *lhs_ptr = lhs_base + (row/accCols)*strideA*accCols;

        lhs_ptr += accCols*offsetA;
        rhs_ptr += remaining_cols*offsetB;
        for(Index k = 0; k < depth; k++)
        {
          for(Index arow = 0; arow < accCols; arow++)
          {
            for(Index acol = 0; acol < remaining_cols; acol++ )
            {
              res(row + arow, col + acol) += alpha*lhs_ptr[arow]*rhs_ptr[acol];
            }
          }
          rhs_ptr += remaining_cols;
          lhs_ptr += accCols;
        }
      }
      
      if(remaining_rows > 0 )
      {
        const Scalar *rhs_ptr  = rhs_base;
        const Scalar *lhs_ptr = lhs_base + (row/accCols)*strideA*accCols;

        lhs_ptr += remaining_rows*offsetA;
        rhs_ptr += remaining_cols*offsetB;
        for(Index k = 0; k < depth; k++)
        {
            for(Index arow = 0; arow < remaining_rows; arow++)
            {
                for(Index acol = 0; acol < remaining_cols; acol++ )
                {
                  res(row + arow, col + acol) += alpha*lhs_ptr[arow]*rhs_ptr[acol];
                }
            }
            rhs_ptr += remaining_cols;
            lhs_ptr += remaining_rows;
        }
      }
    }
}

/*
template<typename LhsScalar, typename RhsScalar, typename Scalarc, typename Scalar, typename Index, typename Packet, typename Packetc, typename RhsPacket, typename DataMapper, bool ConjugateLhs, bool ConjugateRhs, bool LhsIsReal, bool RhsIsReal>
EIGEN_STRONG_INLINE void gemm_complex(const DataMapper& res, const LhsScalar* blockAc, const RhsScalar* blockBc,
          Index rows, Index depth, Index cols, Scalarc alpha, Index strideA, Index strideB, Index offsetA, Index offsetB, const int accRows, const int accCols)
{
      const int remaining_rows = rows % accCols;
      const int remaining_cols = cols % accRows;
      const int accColsC = accCols / 2;
      int advanceCols = 2;
      int advanceRows = 2;

      if(LhsIsReal) advanceRows = 1;
      if(RhsIsReal) advanceCols = 1;

      if( strideA == -1 ) strideA = depth;
      if( strideB == -1 ) strideB = depth;

      const Packet pAlphaReal = pset1<Packet>(alpha.real());
      const Packet pAlphaImag = pset1<Packet>(alpha.imag());

      const Scalar *blockA = (Scalar *) blockAc;
      const Scalar *blockB = (Scalar *) blockBc;

      Packet conj = pset1<Packet>((Scalar)-1.0f);

      Index col = 0;
      for(; col + accRows <= cols; col += accRows)
      {
        const Scalar *rhs_base = blockB + ( (advanceCols*col)/accRows     )*strideB*accRows;
        const Scalar *lhs_base = blockA;

        Index row = 0;
        for(; row + accCols <= rows; row += accCols)
        {
#define MICRO() \
            pgerc<Scalar, Packet, ConjugateLhs, ConjugateRhs, LhsIsReal, RhsIsReal>(accReal1, accImag1, rhs_ptr, rhs_ptr_imag, lhs_ptr1, lhs_ptr_imag1, conj); \
            lhs_ptr1 += accCols; \
            rhs_ptr += accRows; \
            if(!LhsIsReal) \
              lhs_ptr_imag1 += accCols; \
            if(!RhsIsReal) \
              rhs_ptr_imag += accRows; 

          const Scalar *rhs_ptr  = rhs_base;
          const Scalar *rhs_ptr_imag = rhs_ptr + accRows*strideB;
          const Scalar *lhs_ptr1 = lhs_base + ((advanceRows*row)/accCols)*strideA*accCols;
          const Scalar *lhs_ptr_imag1 = lhs_ptr1 + accCols*strideA;

          PacketBlock<Packet,4> accReal1, accImag1;
          bsetzero<Scalar, Packet>(accReal1);
          bsetzero<Scalar, Packet>(accImag1);

          lhs_ptr1 += accCols*offsetA;
          if(!LhsIsReal)
            lhs_ptr_imag1 += accCols*offsetA;
          rhs_ptr += accRows*offsetB;
          if(!RhsIsReal)
            rhs_ptr_imag += accRows*offsetB;
          Index k = 0;
          for(; k + PEEL < depth; k+=PEEL)
          {
            prefetch(rhs_ptr);
            prefetch(rhs_ptr_imag);
            prefetch(lhs_ptr1);
            prefetch(lhs_ptr_imag1);
            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
            MICRO();
#if PEEL > 8
            MICRO();
            MICRO();
#endif
          }
          for(; k < depth; k++)
          {
            MICRO();
          }
          PacketBlock<Packet,4> taccReal, taccImag;
          bscalec<Packet>(accReal1, accImag1, pAlphaReal, pAlphaImag, taccReal, taccImag);

          PacketBlock<Packetc, 8> tRes;
          bload<DataMapper, Packetc, Index, 0>(tRes, res, row, col, accColsC);

          PacketBlock<Packetc, 4> acc1, acc2;
          bcouple<Packet, Packetc>(taccReal, taccImag, tRes, acc1, acc2);

          res.template storePacketBlock<Packetc, 4>(row + 0, col, acc1);
          res.template storePacketBlock<Packetc, 4>(row + accColsC, col, acc2);
#undef MICRO
      }

          if(remaining_rows > 0)
          {
            const Scalar *rhs_ptr  = rhs_base;
            const Scalar *rhs_ptr_imag = rhs_ptr + accRows*strideB;
            const Scalar *lhs_ptr = lhs_base + ((advanceRows*row)/accCols)*strideA*accCols;
            const Scalar *lhs_ptr_imag = lhs_ptr + remaining_rows*strideA;

            lhs_ptr += remaining_rows*offsetA;
            if(!LhsIsReal)
              lhs_ptr_imag += remaining_rows*offsetA;
            rhs_ptr += accRows*offsetB;
            if(!RhsIsReal)
              rhs_ptr_imag += accRows*offsetB;
            for(Index k = 0; k < depth; k++)
            {
              for(Index arow = 0; arow < remaining_rows; arow++)
              {
                Scalar lhs_real = lhs_ptr[arow];
                Scalar lhs_imag;
                if(!LhsIsReal) lhs_imag = lhs_ptr_imag[arow];

                Scalarc lhsc;

                lhsc.real(lhs_real);
                if(!LhsIsReal)
                {
                  if(ConjugateLhs) 
                    lhsc.imag(-lhs_imag);
                  else
                    lhsc.imag(lhs_imag);
                } else {
                  //Lazy approach for now
                  lhsc.imag((Scalar)0);
                }

                for(int acol = 0; acol < accRows; acol++ )
                {
                  Scalar rhs_real = rhs_ptr[acol];
                  Scalar rhs_imag;
                  if(!RhsIsReal) rhs_imag = rhs_ptr_imag[acol];
                  Scalarc rhsc;

                  rhsc.real(rhs_real);
                  if(!RhsIsReal)
                  {
                    if(ConjugateRhs)
                      rhsc.imag(-rhs_imag);
                    else
                      rhsc.imag(rhs_imag);
                  } else {
                    //Lazy approach for now
                    rhsc.imag((Scalar)0);
                  }
                  res(row + arow, col + acol) += alpha*lhsc*rhsc;
                }
              }
              rhs_ptr += accRows;
              lhs_ptr += remaining_rows;
              if(!LhsIsReal)
                lhs_ptr_imag += remaining_rows;
              if(!RhsIsReal)
                rhs_ptr_imag += accRows;
            }
          }
      }

      if(remaining_cols > 0)
      {
        const Scalar *rhs_base = blockB + ( (advanceCols*col)/accRows     )*strideB*accRows;
        const Scalar *lhs_base = blockA;
        Index row = 0;

        for(; row + accCols <= rows; row += accCols)
        {
          const Scalar *rhs_ptr  = rhs_base;
          const Scalar *rhs_ptr_imag = rhs_ptr + remaining_cols*strideB;
          const Scalar *lhs_ptr = lhs_base + ((advanceRows*row)/accCols)*strideA*accCols;
          const Scalar *lhs_ptr_imag = lhs_ptr + accCols*strideA;

          lhs_ptr += accCols*offsetA;
          if(!LhsIsReal)
            lhs_ptr_imag += accCols*offsetA;
          rhs_ptr += remaining_cols*offsetB;
          if(!RhsIsReal)
            rhs_ptr_imag += remaining_cols*offsetB;
          Scalarc scalarAcc[4][4];
          for(Index arow = 0; arow < 4; arow++ )
          {
            for(Index acol = 0; acol < 4; acol++ )
            {
              scalarAcc[arow][acol].real((Scalar)0.0f);
              scalarAcc[arow][acol].imag((Scalar)0.0f);
            }
          }
          for(Index k = 0; k < depth; k++)
          {
            for(Index arow = 0; arow < accCols; arow++)
            {
              Scalar lhs_real = lhs_ptr[arow];
              Scalar lhs_imag;
              if(!LhsIsReal) 
              {
                lhs_imag = lhs_ptr_imag[arow];

                if(ConjugateLhs)
                  lhs_imag *= -1;
              } else {
                lhs_imag = (Scalar)0;
              }

              for(int acol = 0; acol < remaining_cols; acol++ )
              {
                Scalar rhs_real = rhs_ptr[acol];
                Scalar rhs_imag;
                if(!RhsIsReal)
                {
                  rhs_imag = rhs_ptr_imag[acol];

                  if(ConjugateRhs)
                    rhs_imag *= -1;
                } else {
                  rhs_imag = (Scalar)0;
                }

                scalarAcc[arow][acol].real(scalarAcc[arow][acol].real() + lhs_real*rhs_real - lhs_imag*rhs_imag);
                scalarAcc[arow][acol].imag(scalarAcc[arow][acol].imag() + lhs_imag*rhs_real + lhs_real*rhs_imag);
              }
            }
            rhs_ptr += remaining_cols;
            lhs_ptr += accCols;
            if(!RhsIsReal)
              rhs_ptr_imag += remaining_cols;
            if(!LhsIsReal)
              lhs_ptr_imag += accCols;
          }
          for(int arow = 0; arow < accCols; arow++ )
          {
            for(int acol = 0; acol < remaining_cols; acol++ )
            {
              Scalar accR = scalarAcc[arow][acol].real();
              Scalar accI = scalarAcc[arow][acol].imag();
              Scalar aR = alpha.real();
              Scalar aI = alpha.imag();
              Scalar resR = res(row + arow, col + acol).real();
              Scalar resI = res(row + arow, col + acol).imag();

              res(row + arow, col + acol).real(resR + accR*aR - accI*aI);
              res(row + arow, col + acol).imag(resI + accR*aI + accI*aR);
            }
          }
        }

        if(remaining_rows > 0)
        {
          const Scalar *rhs_ptr  = rhs_base;
          const Scalar *rhs_ptr_imag = rhs_ptr + remaining_cols*strideB;
          const Scalar *lhs_ptr = lhs_base + ((advanceRows*row)/accCols)*strideA*accCols;
          const Scalar *lhs_ptr_imag = lhs_ptr + remaining_rows*strideA;

          lhs_ptr += remaining_rows*offsetA;
          if(!LhsIsReal)
            lhs_ptr_imag += remaining_rows*offsetA;
          rhs_ptr += remaining_cols*offsetB;
          if(!RhsIsReal)
            rhs_ptr_imag += remaining_cols*offsetB;
          for(Index k = 0; k < depth; k++)
          {
            for(Index arow = 0; arow < remaining_rows; arow++)
            {
              Scalar lhs_real = lhs_ptr[arow];
              Scalar lhs_imag;
              if(!LhsIsReal) lhs_imag = lhs_ptr_imag[arow];
              Scalarc lhsc;

              lhsc.real(lhs_real);
              if(!LhsIsReal)
              {
                if(ConjugateLhs) 
                  lhsc.imag(-lhs_imag);
                else
                  lhsc.imag(lhs_imag);
              } else {
                lhsc.imag((Scalar)0);
              }

              for(Index acol = 0; acol < remaining_cols; acol++ )
              {
                Scalar rhs_real = rhs_ptr[acol];
                Scalar rhs_imag;
                if(!RhsIsReal) rhs_imag = rhs_ptr_imag[acol];
                Scalarc rhsc;

                rhsc.real(rhs_real);
                if(!RhsIsReal)
                {
                  if(ConjugateRhs)
                    rhsc.imag(-rhs_imag);
                  else
                    rhsc.imag(rhs_imag);
                } else {
                  rhsc.imag((Scalar)0);
                }
                res(row + arow, col + acol) += alpha*lhsc*rhsc;
              }
            }
            rhs_ptr += remaining_cols;
            lhs_ptr += remaining_rows;
            if(!LhsIsReal)
              lhs_ptr_imag += remaining_rows;
            if(!RhsIsReal)
              rhs_ptr_imag += remaining_cols;
          }
        }
      }
}
*/
/************************************
 * ppc64le template specializations *
 * **********************************/
template<typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<double, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate, PanelMode>
{
  void operator()(double* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride=0, Index offset=0);
};

template<typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
void gemm_pack_lhs<double, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate, PanelMode>
  ::operator()(double* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride, Index offset)
{
    lhs_pack<double, Index, DataMapper, Packet2d, ColMajor, PanelMode> pack;
    pack(blockA, lhs, depth, rows, stride, offset);
}

template<typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<double, Index, DataMapper, Pack1, Pack2, Packet, RowMajor, Conjugate, PanelMode>
{
  void operator()(double* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride=0, Index offset=0);
};

template<typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
void gemm_pack_lhs<double, Index, DataMapper, Pack1, Pack2, Packet, RowMajor, Conjugate, PanelMode>
  ::operator()(double* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride, Index offset)
{
    lhs_pack<double, Index, DataMapper, Packet2d, RowMajor, PanelMode> pack;
    pack(blockA, lhs, depth, rows, stride, offset);
}

template<typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<double, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode>
{
  void operator()(double* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride=0, Index offset=0);
};

template<typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
void gemm_pack_rhs<double, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode>
  ::operator()(double* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset)
{
  rhs_pack<double, Index, DataMapper, Packet2d, ColMajor, PanelMode> pack;
  pack(blockB, rhs, depth, cols, stride, offset);
}

template<typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<double, Index, DataMapper, nr, RowMajor, Conjugate, PanelMode>
{
  void operator()(double* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride=0, Index offset=0);
};

template<typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
void gemm_pack_rhs<double, Index, DataMapper, nr, RowMajor, Conjugate, PanelMode>
  ::operator()(double* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset)
{
  rhs_pack<double, Index, DataMapper, Packet2d, RowMajor, PanelMode> pack;
  pack(blockB, rhs, depth, cols, stride, offset);
}

template<typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<float, Index, DataMapper, Pack1, Pack2, Packet, RowMajor, Conjugate, PanelMode>
{
  void operator()(float* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride=0, Index offset=0);
};

template<typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
void gemm_pack_lhs<float, Index, DataMapper, Pack1, Pack2, Packet, RowMajor, Conjugate, PanelMode>
  ::operator()(float* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride, Index offset)
{
  lhs_pack<float, Index, DataMapper, Packet4f, RowMajor, PanelMode> pack;
  pack(blockA, lhs, depth, rows, stride, offset);
}

template<typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<float, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate, PanelMode>
{
  void operator()(float* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride=0, Index offset=0);
};

template<typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
void gemm_pack_lhs<float, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate, PanelMode>
  ::operator()(float* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride, Index offset)
{
  lhs_pack<float, Index, DataMapper, Packet4f, ColMajor, PanelMode> pack;
  pack(blockA, lhs, depth, rows, stride, offset);
}
/*
template<typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<std::complex<float>, Index, DataMapper, Pack1, Pack2, Packet, RowMajor, Conjugate, PanelMode>
{
  void operator()(std::complex<float>* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride=0, Index offset=0);
};

template<typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
void gemm_pack_lhs<std::complex<float>, Index, DataMapper, Pack1, Pack2, Packet, RowMajor, Conjugate, PanelMode>
  ::operator()(std::complex<float>* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride, Index offset)
{
  lhs_cpack<float, true, Index, DataMapper, Packet4f, Packet2cf, RowMajor, Conjugate, PanelMode> pack;
  pack(blockA, lhs, depth, rows, stride, offset);
}

template<typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<std::complex<float>, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate, PanelMode>
{
  void operator()(std::complex<float>* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride=0, Index offset=0);
};

template<typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
void gemm_pack_lhs<std::complex<float>, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate, PanelMode>
  ::operator()(std::complex<float>* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride, Index offset)
{
  lhs_cpack<float, true, Index, DataMapper, Packet4f, Packet2cf, ColMajor, Conjugate, PanelMode> pack;
  pack(blockA, lhs, depth, rows, stride, offset);
}
*/

template<typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<float, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode>
{
  void operator()(float* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride=0, Index offset=0);
};

template<typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
void gemm_pack_rhs<float, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode>
  ::operator()(float* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset)
{
  rhs_pack<float, Index, DataMapper, Packet4f, ColMajor, PanelMode> pack;
  pack(blockB, rhs, depth, cols, stride, offset);
}

template<typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<float, Index, DataMapper, nr, RowMajor, Conjugate, PanelMode>
{
  void operator()(float* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride=0, Index offset=0);
};

template<typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
void gemm_pack_rhs<float, Index, DataMapper, nr, RowMajor, Conjugate, PanelMode>
  ::operator()(float* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset)
{
  rhs_pack<float, Index, DataMapper, Packet4f, RowMajor, PanelMode> pack;
  pack(blockB, rhs, depth, cols, stride, offset);
}

/*
template<typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<std::complex<float>, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode>
{
  void operator()(std::complex<float>* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride=0, Index offset=0);
};

template<typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
void gemm_pack_rhs<std::complex<float>, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode>
  ::operator()(std::complex<float>* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset)
{
  rhs_cpack<float, Index, DataMapper, Packet4f, Packet2cf, ColMajor, Conjugate, PanelMode> pack;
  pack(blockB, rhs, depth, cols, stride, offset);
}

template<typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<std::complex<float>, Index, DataMapper, nr, RowMajor, Conjugate, PanelMode>
{
  void operator()(std::complex<float>* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride=0, Index offset=0);
};

template<typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
void gemm_pack_rhs<std::complex<float>, Index, DataMapper, nr, RowMajor, Conjugate, PanelMode>
  ::operator()(std::complex<float>* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset)
{
  rhs_cpack<float, Index, DataMapper, Packet4f, Packet2cf, RowMajor, Conjugate, PanelMode> pack;
  pack(blockB, rhs, depth, cols, stride, offset);
}

template<typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<std::complex<double>, Index, DataMapper, Pack1, Pack2, Packet, RowMajor, Conjugate, PanelMode>
{
  void operator()(std::complex<double>* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride=0, Index offset=0);
};

template<typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
void gemm_pack_lhs<std::complex<double>, Index, DataMapper, Pack1, Pack2, Packet, RowMajor, Conjugate, PanelMode>
  ::operator()(std::complex<double>* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride, Index offset)
{
  lhs_cpack<double, true, Index, DataMapper, Packet2d, Packet1cd, RowMajor, Conjugate, PanelMode> pack;
  pack(blockA, lhs, depth, rows, stride, offset);
}

template<typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<std::complex<double>, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate, PanelMode>
{
  void operator()(std::complex<double>* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride=0, Index offset=0);
};

template<typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
void gemm_pack_lhs<std::complex<double>, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate, PanelMode>
  ::operator()(std::complex<double>* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride, Index offset)
{
  lhs_cpack<double, true, Index, DataMapper, Packet2d, Packet1cd, ColMajor, Conjugate, PanelMode> pack;
  pack(blockA, lhs, depth, rows, stride, offset);
}

template<typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<std::complex<double>, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode>
{
  void operator()(std::complex<double>* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride=0, Index offset=0);
};

template<typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
void gemm_pack_rhs<std::complex<double>, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode>
  ::operator()(std::complex<double>* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset)
{
  rhs_cpack<double, Index, DataMapper, Packet2d, Packet1cd, ColMajor, Conjugate, PanelMode> pack;
  pack(blockB, rhs, depth, cols, stride, offset);
}

template<typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<std::complex<double>, Index, DataMapper, nr, RowMajor, Conjugate, PanelMode>
{
  void operator()(std::complex<double>* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride=0, Index offset=0);
};

template<typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
void gemm_pack_rhs<std::complex<double>, Index, DataMapper, nr, RowMajor, Conjugate, PanelMode>
  ::operator()(std::complex<double>* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset)
{
  rhs_cpack<double, Index, DataMapper, Packet2d, Packet1cd, RowMajor, Conjugate, PanelMode> pack;
  pack(blockB, rhs, depth, cols, stride, offset);
}
*/

// ********* gebp specializations *********
template<typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<float, float, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
{
  typedef typename quad_traits<float>::vectortype   Packet;
  typedef typename quad_traits<float>::rhstype      RhsPacket;

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
    const int accRows = quad_traits<float>::rows;
    const int accCols = quad_traits<float>::size;

    gemm<float, Index, Packet, RhsPacket, DataMapper>(res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB, accRows, accCols);
  }

/*
template<typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<std::complex<float>, std::complex<float>, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
{
  typedef Packet4f   Packet;
  typedef Packet2cf  Packetc;
  typedef Packet4f   RhsPacket;

  void operator()(const DataMapper& res, const std::complex<float>* blockA, const std::complex<float>* blockB,
                  Index rows, Index depth, Index cols, std::complex<float> alpha,
                  Index strideA=-1, Index strideB=-1, Index offsetA=0, Index offsetB=0);
};

template<typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
void gebp_kernel<std::complex<float>, std::complex<float>, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
  ::operator()(const DataMapper& res, const std::complex<float>* blockA, const std::complex<float>* blockB,
               Index rows, Index depth, Index cols, std::complex<float> alpha,
               Index strideA, Index strideB, Index offsetA, Index offsetB)
  {
    const int accRows = quad_traits<float>::rows;
    const int accCols = quad_traits<float>::size;

    gemm_complex<std::complex<float>, std::complex<float>, std::complex<float>, float, Index, Packet, Packetc, RhsPacket, DataMapper, ConjugateLhs, ConjugateRhs, false, false>(res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB, accRows, accCols);
  }

template<typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<float, std::complex<float>, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
{
  typedef Packet4f   Packet;
  typedef Packet2cf  Packetc;
  typedef Packet4f   RhsPacket;

  void operator()(const DataMapper& res, const float* blockA, const std::complex<float>* blockB,
                  Index rows, Index depth, Index cols, std::complex<float> alpha,
                  Index strideA=-1, Index strideB=-1, Index offsetA=0, Index offsetB=0);
};

template<typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
void gebp_kernel<float, std::complex<float>, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
  ::operator()(const DataMapper& res, const float* blockA, const std::complex<float>* blockB,
               Index rows, Index depth, Index cols, std::complex<float> alpha,
               Index strideA, Index strideB, Index offsetA, Index offsetB)
  {
    const int accRows = quad_traits<float>::rows;
    const int accCols = quad_traits<float>::size;

    gemm_complex<float, std::complex<float>, std::complex<float>, float, Index, Packet, Packetc, RhsPacket, DataMapper, ConjugateLhs, ConjugateRhs, true, false>(res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB, accRows, accCols);
  }

template<typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<std::complex<float>, float, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
{
  typedef Packet4f   Packet;
  typedef Packet2cf  Packetc;
  typedef Packet4f   RhsPacket;

  void operator()(const DataMapper& res, const std::complex<float>* blockA, const float* blockB,
                  Index rows, Index depth, Index cols, std::complex<float> alpha,
                  Index strideA=-1, Index strideB=-1, Index offsetA=0, Index offsetB=0);
};

template<typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
void gebp_kernel<std::complex<float>, float, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
  ::operator()(const DataMapper& res, const std::complex<float>* blockA, const float* blockB,
               Index rows, Index depth, Index cols, std::complex<float> alpha,
               Index strideA, Index strideB, Index offsetA, Index offsetB)
  {
    const int accRows = quad_traits<float>::rows;
    const int accCols = quad_traits<float>::size;

    gemm_complex<std::complex<float>, float, std::complex<float>, float, Index, Packet, Packetc, RhsPacket, DataMapper, ConjugateLhs, ConjugateRhs, false, true>(res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB, accRows, accCols);
  }
*/
template<typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<double, double, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
{
  typedef typename quad_traits<double>::vectortype  Packet;
  typedef typename quad_traits<double>::rhstype     RhsPacket;

  void operator()(const DataMapper& res, const double* blockA, const double* blockB,
                  Index rows, Index depth, Index cols, double alpha,
                  Index strideA=-1, Index strideB=-1, Index offsetA=0, Index offsetB=0);
};

template<typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
void gebp_kernel<double, double, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
  ::operator()(const DataMapper& res, const double* blockA, const double* blockB,
               Index rows, Index depth, Index cols, double alpha,
               Index strideA, Index strideB, Index offsetA, Index offsetB)
  {
    const int accRows = quad_traits<double>::rows;
    const int accCols = quad_traits<double>::size;

    gemm<double, Index, Packet, RhsPacket, DataMapper>(res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB, accRows, accCols);
  }
/*
template<typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<std::complex<double>, std::complex<double>, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
{
  typedef quad_traits<double>::vectortype   Packet;
  typedef Packet1cd  Packetc;
  typedef quad_traits<double>::rhstype   RhsPacket;

  void operator()(const DataMapper& res, const std::complex<double>* blockA, const std::complex<double>* blockB,
                  Index rows, Index depth, Index cols, std::complex<double> alpha,
                  Index strideA=-1, Index strideB=-1, Index offsetA=0, Index offsetB=0);
};

template<typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
void gebp_kernel<std::complex<double>, std::complex<double>, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
  ::operator()(const DataMapper& res, const std::complex<double>* blockA, const std::complex<double>* blockB,
               Index rows, Index depth, Index cols, std::complex<double> alpha,
               Index strideA, Index strideB, Index offsetA, Index offsetB)
  {
    const int accRows = quad_traits<double>::rows;
    const int accCols = quad_traits<double>::size;

    gemm_complex<std::complex<double>, std::complex<double>, std::complex<double>, double, Index, Packet, Packetc, RhsPacket, DataMapper, ConjugateLhs, ConjugateRhs, false, false>(res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB, accRows, accCols);
  }

template<typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<std::complex<double>, double, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
{
  typedef quad_traits<double>::vectortype   Packet;
  typedef Packet1cd  Packetc;
  typedef quad_traits<double>::rhstype   RhsPacket;

  void operator()(const DataMapper& res, const std::complex<double>* blockA, const double* blockB,
                  Index rows, Index depth, Index cols, std::complex<double> alpha,
                  Index strideA=-1, Index strideB=-1, Index offsetA=0, Index offsetB=0);
};

template<typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
void gebp_kernel<std::complex<double>, double, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
  ::operator()(const DataMapper& res, const std::complex<double>* blockA, const double* blockB,
               Index rows, Index depth, Index cols, std::complex<double> alpha,
               Index strideA, Index strideB, Index offsetA, Index offsetB)
  {
    const int accRows = quad_traits<double>::rows;
    const int accCols = quad_traits<double>::size;

    gemm_complex<std::complex<double>, double, std::complex<double>, double, Index, Packet, Packetc, RhsPacket, DataMapper, ConjugateLhs, ConjugateRhs, false, true>(res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB, accRows, accCols);
  }

template<typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<double, std::complex<double>, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
{
  typedef quad_traits<double>::vectortype   Packet;
  typedef Packet1cd  Packetc;
  typedef quad_traits<double>::rhstype   RhsPacket;

  void operator()(const DataMapper& res, const double* blockA, const std::complex<double>* blockB,
                  Index rows, Index depth, Index cols, std::complex<double> alpha,
                  Index strideA=-1, Index strideB=-1, Index offsetA=0, Index offsetB=0);
};

template<typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
void gebp_kernel<double, std::complex<double>, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
  ::operator()(const DataMapper& res, const double* blockA, const std::complex<double>* blockB,
               Index rows, Index depth, Index cols, std::complex<double> alpha,
               Index strideA, Index strideB, Index offsetA, Index offsetB)
  {
    const int accRows = quad_traits<double>::rows;
    const int accCols = quad_traits<double>::size;

    gemm_complex<double, std::complex<double>, std::complex<double>, double, Index, Packet, Packetc, RhsPacket, DataMapper, ConjugateLhs, ConjugateRhs, true, false>(res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB, accRows, accCols);
  }
*/
} // end namespace internal

} // end namespace Eigen
#endif // EIGEN_MATRIX_PRODUCT_NEON_H