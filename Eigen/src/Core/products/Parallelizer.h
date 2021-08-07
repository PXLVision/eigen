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

#ifndef EIGEN_PARALLEL_GEMM_STEALING
#define EIGEN_PARALLEL_GEMM_STEALING 0
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

template<typename Index> struct GemmParallelInfo
{
  GemmParallelInfo() : 
    shard_index(0), shards(1),
    #if EIGEN_HAS_CXX11_ATOMIC && EIGEN_PARALLEL_GEMM_STEALING
      init_started(false), init_finished(false),
      rhs_depth_next(0), rhs_depth_done(0),
      lhs_depth_next(0),
    #endif
    lhs_depth_ready(-1), lhs_depth_users(0),
    lhs_start(0), lhs_length(0), rhs_start(0), rhs_length(0) {}

  Index shard_index; // Actual shard index.
  Index shards;      // Number of total shards.
  
  // volatile is not enough on all architectures (see bug 1572)
  // to guarantee that when thread A says to thread B that it is
  // done with packing a block, then all writes have been really
  // carried out... C++11 memory model+atomic guarantees this.
#if EIGEN_HAS_CXX11_ATOMIC
  #if EIGEN_PARALLEL_GEMM_STEALING
    std::atomic<bool> init_started;
    std::atomic<bool> init_finished;
    std::atomic<Index> rhs_depth_next;
    std::atomic<Index> rhs_depth_done;
    std::atomic<Index> lhs_depth_next;
  #endif
  std::atomic<Index> lhs_depth_ready;
  std::atomic<int> lhs_depth_users;
#else
  Index volatile lhs_depth_ready;
  int volatile lhs_depth_users;
#endif
  Index lhs_start;
  Index lhs_length;
  Index rhs_start;
  Index rhs_length;
};

template<bool Condition, typename Functor, typename Index>
void parallelize_gemm(const Functor& func, Index rows, Index cols, Index depth, bool transpose)
{
  // TODO when EIGEN_USE_BLAS is defined,
  // we should still enable OMP for other scalar types
  // Without C++11, we have to disable GEMM's parallelization on
  // non x86 architectures because there volatile is not enough for our purpose.
  // See bug 1572.
#if (! defined(EIGEN_HAS_OPENMP)) || defined(EIGEN_USE_BLAS) || ((!EIGEN_HAS_CXX11_ATOMIC) && !(EIGEN_ARCH_i386_OR_x86_64))
  // FIXME the transpose variable is only needed to properly split
  // the matrix product when multithreading is enabled. This is a temporary
  // fix to support row-major destination matrices. This whole
  // parallelizer mechanism has to be redesigned anyway.
  EIGEN_UNUSED_VARIABLE(depth);
  EIGEN_UNUSED_VARIABLE(transpose);
  func(0,rows, 0,cols);
#else

  // Dynamically check whether we should enable or disable OpenMP.
  // The conditions are:
  // - the max number of threads we can create is greater than 1
  // - we are not already in a parallel code
  // - the sizes are large enough

  // compute the maximal number of threads from the size of the product:
  // This first heuristic takes into account that the product kernel is fully optimized when working with nr columns at once.
  Index size = transpose ? rows : cols;
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
  // FIXME omp_get_num_threads()>1 only works for openmp, what if the user does not use openmp?
  if((!Condition) || (threads==1) || (omp_get_num_threads()>1))
    return func(0,rows, 0,cols);

  Eigen::initParallel();
  func.initParallelSession(threads);

  if(transpose)
    std::swap(rows,cols);

  ei_declare_aligned_stack_constructed_variable(GemmParallelInfo<Index>,info,threads,0);

  // threads = 5;
  // Index actual_threads = threads;
  // for (Index i=0; i<actual_threads; ++i) {
  #pragma omp parallel num_threads(threads)
  {
    Index i = omp_get_thread_num();
    // Note that the actual number of threads might be lower than the number of request ones.
    Index actual_threads = omp_get_num_threads();
  
    Index blockCols = (cols / actual_threads) & ~Index(0x3);
    Index blockRows = (rows / actual_threads);
    blockRows = (blockRows/Functor::Traits::mr)*Functor::Traits::mr;

    // Initialize parallel info with true number of threads.
    #if EIGEN_HAS_CXX11_ATOMIC && EIGEN_PARALLEL_GEMM_STEALING
      // Work-stealing approach.
      for (Index shift=0; shift<actual_threads; ++shift) {
        // Cycle around for all shards, starting at the current.
        const Index j = (i + shift) % actual_threads;
        
        bool expected = false;
        if (info[j].init_started.compare_exchange_strong(expected, true)) {
          Index r0 = j*blockRows;
          Index actualBlockRows = (j+1==actual_threads) ? rows-r0 : blockRows;
          Index c0 = j*blockCols;
          Index actualBlockCols = (j+1==actual_threads) ? cols-c0 : blockCols;

          info[j].shard_index = j;
          info[j].shards = actual_threads;
          info[j].lhs_start = r0;
          info[j].lhs_length = actualBlockRows;
          info[j].rhs_start = c0;
          info[j].rhs_length = actualBlockCols;
          
          info[j].init_finished = true;
        }
      }
      
      // Wait for initialization to finish.
      for (Index j=0; j<actual_threads; ++j) {
        while(!info[j].init_finished) {}
      }
    #else
      Index r0 = i*blockRows;
      Index actualBlockRows = (i+1==actual_threads) ? rows-r0 : blockRows;
      Index c0 = i*blockCols;
      Index actualBlockCols = (i+1==actual_threads) ? cols-c0 : blockCols;

      info[i].shard_index = i;
      info[i].shards = actual_threads;
      info[i].lhs_start = r0;
      info[i].lhs_length = actualBlockRows;
      info[i].rhs_start = c0;
      info[i].rhs_length = actualBlockCols;
    #endif   
    
    if(transpose) {
      func(0, cols, 0, rows, info, i);
    } else {
      func(0, rows, 0, cols, info, i);
    }
  }
#endif
}

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_PARALLELIZER_H
