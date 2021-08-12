// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PARALLELIZER_H
#define EIGEN_PARALLELIZER_H

#if EIGEN_HAS_CXX11_ATOMIC
#include <atomic>
#endif

namespace Eigen {

namespace internal {

/** \internal */
inline void manage_multi_threading(Action action, int* v)
{
  static int m_maxThreads = -1;
  EIGEN_UNUSED_VARIABLE(m_maxThreads)

  if(action==SetAction)
  {
    eigen_internal_assert(v!=0);
    m_maxThreads = *v;
  }
  else if(action==GetAction)
  {
    eigen_internal_assert(v!=0);
    #ifdef EIGEN_HAS_OPENMP
    if(m_maxThreads>0)
      *v = m_maxThreads;
    else
      *v = omp_get_max_threads();
    #else
    *v = 1;
    #endif
  }
  else
  {
    eigen_internal_assert(false);
  }
}

}

/** Must be call first when calling Eigen from multiple threads */
inline void initParallel()
{
  int nbt;
  internal::manage_multi_threading(GetAction, &nbt);
  std::ptrdiff_t l1, l2, l3;
  internal::manage_caching_sizes(GetAction, &l1, &l2, &l3);
}

/** \returns the max number of threads reserved for Eigen
  * \sa setNbThreads */
inline int nbThreads()
{
  int ret;
  internal::manage_multi_threading(GetAction, &ret);
  return ret;
}

/** Sets the max number of threads reserved for Eigen
  * \sa nbThreads */
inline void setNbThreads(int v)
{
  internal::manage_multi_threading(SetAction, &v);
}

namespace internal {

template<
  typename Index,
  typename LhsScalar, int LhsStorageOrder, bool ConjugateLhs,
  typename RhsScalar, int RhsStorageOrder, bool ConjugateRhs,
  int ResInnerStride>
struct SequentialGemmImpl {
  typedef gebp_traits<LhsScalar,RhsScalar> Traits;
  typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType ResScalar;
  
  static void run(Index rows, Index cols, Index depth,
                  const LhsScalar* _lhs, Index lhsStride,
                  const RhsScalar* _rhs, Index rhsStride,
                  ResScalar* _res, Index resIncr, Index resStride,
                  ResScalar alpha, level3_blocking<LhsScalar,RhsScalar>& blocking) {
    typedef const_blas_data_mapper<LhsScalar, Index, LhsStorageOrder> LhsMapper;
    typedef const_blas_data_mapper<RhsScalar, Index, RhsStorageOrder> RhsMapper;
    typedef blas_data_mapper<typename Traits::ResScalar, Index, ColMajor,Unaligned,ResInnerStride> ResMapper;
    LhsMapper lhs(_lhs, lhsStride);
    RhsMapper rhs(_rhs, rhsStride);
    ResMapper res(_res, resStride, resIncr);

    gemm_pack_lhs<LhsScalar, Index, LhsMapper, Traits::mr, Traits::LhsProgress, typename Traits::LhsPacket4Packing, LhsStorageOrder> pack_lhs;
    gemm_pack_rhs<RhsScalar, Index, RhsMapper, Traits::nr, RhsStorageOrder> pack_rhs;
    gebp_kernel<LhsScalar, RhsScalar, Index, ResMapper, Traits::mr, Traits::nr, ConjugateLhs, ConjugateRhs> gebp;
    
    Index kc = blocking.kc();                   // cache block size along the K direction
    Index mc = (std::min)(rows,blocking.mc());  // cache block size along the M direction
    Index nc = (std::min)(cols,blocking.nc());  // cache block size along the N direction
    std::size_t sizeA = kc*mc;
    std::size_t sizeB = kc*nc;

    ei_declare_aligned_stack_constructed_variable(LhsScalar, blockA, sizeA, blocking.blockA());
    ei_declare_aligned_stack_constructed_variable(RhsScalar, blockB, sizeB, blocking.blockB());

    const bool pack_rhs_once = mc!=rows && kc==depth && nc==cols;

    // For each horizontal panel of the rhs, and corresponding panel of the lhs...
    for(Index i2=0; i2<rows; i2+=mc)
    {
      const Index actual_mc = (std::min)(i2+mc,rows)-i2;

      for(Index k2=0; k2<depth; k2+=kc)
      {
        const Index actual_kc = (std::min)(k2+kc,depth)-k2;

        // OK, here we have selected one horizontal panel of rhs and one vertical panel of lhs.
        // => Pack lhs's panel into a sequential chunk of memory (L2/L3 caching)
        // Note that this panel will be read as many times as the number of blocks in the rhs's
        // horizontal panel which is, in practice, a very low number.
        pack_lhs(blockA, lhs.getSubMapper(i2,k2), actual_kc, actual_mc);

        // For each kc x nc block of the rhs's horizontal panel...
        for(Index j2=0; j2<cols; j2+=nc)
        {
          const Index actual_nc = (std::min)(j2+nc,cols)-j2;

          // We pack the rhs's block into a sequential chunk of memory (L2 caching)
          // Note that this block will be read a very high number of times, which is equal to the number of
          // micro horizontal panel of the large rhs's panel (e.g., rows/12 times).
          if((!pack_rhs_once) || i2==0)
            pack_rhs(blockB, rhs.getSubMapper(k2,j2), actual_kc, actual_nc);

          // Everything is packed, we can now call the panel * block kernel:
          gebp(res.getSubMapper(i2, j2), blockA, blockB, actual_mc, actual_kc, actual_nc, alpha);
        }
      }
    }
  }
};

#if defined(EIGEN_HAS_OPENMP)

template<typename Index> struct GemmParallelInfo
{
  GemmParallelInfo() : sync(-1), users(0), lhs_start(0), lhs_length(0) {}

  // volatile is not enough on all architectures (see bug 1572)
  // to guarantee that when thread A says to thread B that it is
  // done with packing a block, then all writes have been really
  // carried out... C++11 memory model+atomic guarantees this.
#if EIGEN_HAS_CXX11_ATOMIC
  std::atomic<Index> sync;
  std::atomic<int> users;
#else
  Index volatile sync;
  int volatile users;
#endif

  Index lhs_start;
  Index lhs_length;
};

template<
  typename Index,
  typename LhsScalar, int LhsStorageOrder, bool ConjugateLhs,
  typename RhsScalar, int RhsStorageOrder, bool ConjugateRhs,
  int ResInnerStride, bool Transpose>
struct OpenMpParallelGemmImpl {
  typedef gebp_traits<LhsScalar,RhsScalar> Traits;
  typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType ResScalar;
  typedef SequentialSequentialGemmImpl<Index, 
      LhsScalar, LhsStorageOrder, 
      ConjugateLhs, RhsScalar, RhsStorageOrder, ConjugateRhs,
      ResInnerStride> SequentialGemm;
      
  static void run(Index rows, Index cols, Index depth,
    const LhsScalar* _lhs, Index lhsStride,
    const RhsScalar* _rhs, Index rhsStride,
    ResScalar* _res, Index resIncr, Index resStride,
    ResScalar alpha,
    level3_blocking<LhsScalar,RhsScalar>& blocking) {
    // TODO when EIGEN_USE_BLAS is defined,
    // we should still enable OMP for other scalar types
    // Without C++11, we have to disable GEMM's parallelization on
    // non x86 architectures because there volatile is not enough for our purpose.
    // See bug 1572.
#if ( defined(EIGEN_USE_BLAS) || ((!EIGEN_HAS_CXX11_ATOMIC) && !(EIGEN_ARCH_i386_OR_x86_64))
    SequentialGemm::run(rows, cols, depth, _lhs, lhsStride, _rhs, rhsStride, alpha, blocking);
#else

    // Dynamically check whether we should enable or disable OpenMP.
    // The conditions are:
    // - the max number of threads we can create is greater than 1
    // - we are not already in a parallel code
    // - the sizes are large enough

    // compute the maximal number of threads from the size of the product:
    // This first heuristic takes into account that the product kernel is fully optimized when working with nr columns at once.
    Index size = Transpose ? rows : cols;
    Index pb_max_threads = std::max<Index>(1,size / Functor::Traits::nr);

    // compute the maximal number of threads from the total amount of work:
    double work = static_cast<double>(rows) * static_cast<double>(cols) *
        static_cast<double>(depth);
    double kMinTaskSize = 50000;  // FIXME improve this heuristic.
    pb_max_threads = std::max<Index>(1, std::min<Index>(pb_max_threads, static_cast<Index>( work / kMinTaskSize ) ));

    // compute the number of threads we are going to use
    Index threads = std::min<Index>(nbThreads(), pb_max_threads);

    // if multi-threading is explicitly disabled, not useful, or if we already are in a parallel session,
    // then abort multi-threading
    if((threads==1) || (omp_get_num_threads()>1))
      return SequentialGemm::run(rows, cols, depth, _lhs, lhsStride, _rhs, rhsStride, alpha, blocking) ;

    Eigen::initParallel();
    func.initParallelSession(threads);

    if(Transpose)
      std::swap(rows,cols);

    ei_declare_aligned_stack_constructed_variable(GemmParallelInfo<Index>,info,threads,0);

    #pragma omp parallel num_threads(threads)
    {
      Index i = omp_get_thread_num();
      // Note that the actual number of threads might be lower than the number of request ones.
      Index actual_threads = omp_get_num_threads();

      Index blockCols = (cols / actual_threads) & ~Index(0x3);
      Index blockRows = (rows / actual_threads);
      blockRows = (blockRows/Functor::Traits::mr)*Functor::Traits::mr;

      Index r0 = i*blockRows;
      Index actualBlockRows = (i+1==actual_threads) ? rows-r0 : blockRows;

      Index c0 = i*blockCols;
      Index actualBlockCols = (i+1==actual_threads) ? cols-c0 : blockCols;

      info[i].lhs_start = r0;
      info[i].lhs_length = actualBlockRows;

      if(Transpose) func(c0, actualBlockCols, 0, rows, info);
      else          func(0, rows, c0, actualBlockCols, info);
    }
  #endif
 }
 
  static void thread_run(Index rows, Index cols, Index depth,
    const LhsScalar* _lhs, Index lhsStride,
    const RhsScalar* _rhs, Index rhsStride,
    ResScalar* _res, Index resIncr, Index resStride,
    ResScalar alpha,
    level3_blocking<LhsScalar,RhsScalar>& blocking,
    GemmParallelInfo<Index>* info = 0)
  {
    typedef const_blas_data_mapper<LhsScalar, Index, LhsStorageOrder> LhsMapper;
    typedef const_blas_data_mapper<RhsScalar, Index, RhsStorageOrder> RhsMapper;
    typedef blas_data_mapper<typename Traits::ResScalar, Index, ColMajor,Unaligned,ResInnerStride> ResMapper;
    LhsMapper lhs(_lhs, lhsStride);
    RhsMapper rhs(_rhs, rhsStride);
    ResMapper res(_res, resStride, resIncr);

    Index kc = blocking.kc();                   // cache block size along the K direction
    Index mc = (std::min)(rows,blocking.mc());  // cache block size along the M direction
    Index nc = (std::min)(cols,blocking.nc());  // cache block size along the N direction

    gemm_pack_lhs<LhsScalar, Index, LhsMapper, Traits::mr, Traits::LhsProgress, typename Traits::LhsPacket4Packing, LhsStorageOrder> pack_lhs;
    gemm_pack_rhs<RhsScalar, Index, RhsMapper, Traits::nr, RhsStorageOrder> pack_rhs;
    gebp_kernel<LhsScalar, RhsScalar, Index, ResMapper, Traits::mr, Traits::nr, ConjugateLhs, ConjugateRhs> gebp;

    if(info)
    {
      // this is the parallel version!
      int tid = omp_get_thread_num();
      int threads = omp_get_num_threads();

      LhsScalar* blockA = blocking.blockA();
      eigen_internal_assert(blockA!=0);

      std::size_t sizeB = kc*nc;
      ei_declare_aligned_stack_constructed_variable(RhsScalar, blockB, sizeB, 0);

      // For each horizontal panel of the rhs, and corresponding vertical panel of the lhs...
      for(Index k=0; k<depth; k+=kc)
      {
        const Index actual_kc = (std::min)(k+kc,depth)-k; // => rows of B', and cols of the A'

        // In order to reduce the chance that a thread has to wait for the other,
        // let's start by packing B'.
        pack_rhs(blockB, rhs.getSubMapper(k,0), actual_kc, nc);

        // Pack A_k to A' in a parallel fashion:
        // each thread packs the sub block A_k,i to A'_i where i is the thread id.

        // However, before copying to A'_i, we have to make sure that no other thread is still using it,
        // i.e., we test that info[tid].users equals 0.
        // Then, we set info[tid].users to the number of threads to mark that all other threads are going to use it.
        while(info[tid].users!=0) {}
        info[tid].users = threads;

        pack_lhs(blockA+info[tid].lhs_start*actual_kc, lhs.getSubMapper(info[tid].lhs_start,k), actual_kc, info[tid].lhs_length);

        // Notify the other threads that the part A'_i is ready to go.
        info[tid].sync = k;

        // Computes C_i += A' * B' per A'_i
        for(int shift=0; shift<threads; ++shift)
        {
          int i = (tid+shift)%threads;

          // At this point we have to make sure that A'_i has been updated by the thread i,
          // we use testAndSetOrdered to mimic a volatile access.
          // However, no need to wait for the B' part which has been updated by the current thread!
          if (shift>0) {
            while(info[i].sync!=k) {
            }
          }

          gebp(res.getSubMapper(info[i].lhs_start, 0), blockA+info[i].lhs_start*actual_kc, blockB, info[i].lhs_length, actual_kc, nc, alpha);
        }

        // Then keep going as usual with the remaining B'
        for(Index j=nc; j<cols; j+=nc)
        {
          const Index actual_nc = (std::min)(j+nc,cols)-j;

          // pack B_k,j to B'
          pack_rhs(blockB, rhs.getSubMapper(k,j), actual_kc, actual_nc);

          // C_j += A' * B'
          gebp(res.getSubMapper(0, j), blockA, blockB, rows, actual_kc, actual_nc, alpha);
        }

        // Release all the sub blocks A'_i of A' for the current thread,
        // i.e., we simply decrement the number of users by 1
        for(Index i=0; i<threads; ++i)
#if !EIGEN_HAS_CXX11_ATOMIC
          #pragma omp atomic
#endif
          info[i].users -= 1;
      }
    }
    else
#endif // EIGEN_HAS_OPENMP
  }

};

#endif // EIGEN_HAS_OPENMP

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_PARALLELIZER_H
