// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_GENERAL_MATRIX_MATRIX_H
#define EIGEN_GENERAL_MATRIX_MATRIX_H

namespace Eigen {

namespace internal {

template<typename LhsScalar_, typename RhsScalar_> class level3_blocking;

/* Specialization for a row-major destination matrix => simple transposition of the product */
template<
  typename Index,
  typename LhsScalar, int LhsStorageOrder, bool ConjugateLhs,
  typename RhsScalar, int RhsStorageOrder, bool ConjugateRhs,
  int ResInnerStride>
struct general_matrix_matrix_product<Index,LhsScalar,LhsStorageOrder,ConjugateLhs,RhsScalar,RhsStorageOrder,ConjugateRhs,RowMajor,ResInnerStride>
{
  typedef gebp_traits<RhsScalar,LhsScalar> Traits;

  typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType ResScalar;
  static EIGEN_STRONG_INLINE void run(
    Index rows, Index cols, Index depth,
    const LhsScalar* lhs, Index lhsStride,
    const RhsScalar* rhs, Index rhsStride,
    ResScalar* res, Index resIncr, Index resStride,
    ResScalar alpha,
    level3_blocking<RhsScalar,LhsScalar>& blocking,
    GemmParallelInfo<Index>* info = 0,
    Index shard_index = 0)
  {
    // transpose the product such that the result is column major
    general_matrix_matrix_product<Index,
      RhsScalar, RhsStorageOrder==RowMajor ? ColMajor : RowMajor, ConjugateRhs,
      LhsScalar, LhsStorageOrder==RowMajor ? ColMajor : RowMajor, ConjugateLhs,
      ColMajor,ResInnerStride>
    ::run(cols,rows,depth,rhs,rhsStride,lhs,lhsStride,res,resIncr,resStride,alpha,blocking,info,shard_index);
  }
};

/*  Specialization for a col-major destination matrix
 *    => Blocking algorithm following Goto's paper */
template<
  typename Index,
  typename LhsScalar, int LhsStorageOrder, bool ConjugateLhs,
  typename RhsScalar, int RhsStorageOrder, bool ConjugateRhs,
  int ResInnerStride>
struct general_matrix_matrix_product<Index,LhsScalar,LhsStorageOrder,ConjugateLhs,RhsScalar,RhsStorageOrder,ConjugateRhs,ColMajor,ResInnerStride>
{

  typedef gebp_traits<LhsScalar,RhsScalar> Traits;
  typedef const_blas_data_mapper<LhsScalar, Index, LhsStorageOrder> LhsMapper;
  typedef const_blas_data_mapper<RhsScalar, Index, RhsStorageOrder> RhsMapper;
  typedef blas_data_mapper<typename Traits::ResScalar, Index, ColMajor,Unaligned,ResInnerStride> ResMapper;
  typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType ResScalar;

  static void run(Index rows, Index cols, Index depth,
    const LhsScalar* _lhs, Index lhsStride,
    const RhsScalar* _rhs, Index rhsStride,
    ResScalar* _res, Index resIncr, Index resStride,
    ResScalar alpha,
    level3_blocking<LhsScalar,RhsScalar>& blocking,
    GemmParallelInfo<Index>* info = 0,
    Index shard_index = 0)
  {
    if (info) {
      #if EIGEN_HAS_CXX11_ATOMIC && EIGEN_PARALLEL_GEMM_STEALING
        run_parallel_stealing(rows, cols, depth, _lhs, lhsStride, _rhs, rhsStride,
                              _res, resIncr, resStride, alpha, blocking, info, shard_index);
      #elif defined(EIGEN_HAS_OPENMP)
        run_parallel(rows, cols, depth, _lhs, lhsStride, _rhs, rhsStride,
                    _res, resIncr, resStride, alpha, blocking, info, shard_index);
      #else
        run_sequential(rows, cols, depth, _lhs, lhsStride, _rhs, rhsStride,
                      _res, resIncr, resStride, alpha, blocking);  
      #endif
    } else {
      run_sequential(rows, cols, depth, _lhs, lhsStride, _rhs, rhsStride,
                    _res, resIncr, resStride, alpha, blocking);  
    }
  }
  
#ifdef EIGEN_HAS_OPENMP
  static void run_parallel(Index rows, Index cols, Index depth,
    const LhsScalar* _lhs, Index lhsStride,
    const RhsScalar* _rhs, Index rhsStride,
    ResScalar* _res, Index resIncr, Index resStride,
    ResScalar alpha,
    level3_blocking<LhsScalar,RhsScalar>& blocking,
    GemmParallelInfo<Index>* info, Index shard_index)
  {
    LhsMapper lhs(_lhs, lhsStride);
    RhsMapper rhs(_rhs, rhsStride);
    ResMapper res(_res, resStride, resIncr);

    // printf("Running basic parallel GEMM.\n");
    // this is the parallel version!
    const int tid = shard_index;
    const int threads = info[tid].shards;
    const Index c0 = info[tid].rhs_start;
    const Index block_cols = info[tid].rhs_length;

    Index kc = blocking.kc();                   // cache block size along the K direction
    // Index mc = (std::min)(rows,blocking.mc());  // cache block size along the M direction
    Index nc = (std::min)(block_cols,blocking.nc());  // cache block size along the N direction

    gemm_pack_lhs<LhsScalar, Index, LhsMapper, Traits::mr, Traits::LhsProgress, typename Traits::LhsPacket4Packing, LhsStorageOrder> pack_lhs;
    gemm_pack_rhs<RhsScalar, Index, RhsMapper, Traits::nr, RhsStorageOrder> pack_rhs;
    gebp_kernel<LhsScalar, RhsScalar, Index, ResMapper, Traits::mr, Traits::nr, ConjugateLhs, ConjugateRhs> gebp;
    // printf("thread %li blocking: %lix%lix%li\n", shard_index, mc, kc, nc);

    LhsScalar* blockA = blocking.blockA();
    eigen_internal_assert(blockA!=0);

    std::size_t sizeB = kc*nc;
    ei_declare_aligned_stack_constructed_variable(RhsScalar, blockB, sizeB, 0);

    // For each horizontal panel of the rhs, and corresponding vertical panel of the lhs...
    for(Index k=0; k<depth; k+=kc)
    {
      const Index actual_kc = (std::min)(k+kc,depth)-k; // => rows of B', and cols of the A'

      // In order to reduce the chance that a thread has to wait for the other,
      // let's start by packing B' = B_k,0.
      // printf("thread %i pack B(%li:%li,%li:%li)\n", tid, k, k+actual_kc, c0, c0+nc);
      pack_rhs(blockB, rhs.getSubMapper(k,c0), actual_kc, nc);

      // Pack A_k to A' in a parallel fashion:
      // each thread packs the sub block A_i,k to A'_i where i is the thread id.

      // However, before copying to A'_i, we have to make sure that no other thread is still using it,
      // i.e., we test that info[tid].users equals 0.
      // Then, we set info[tid].users to the number of threads to mark that all other threads are going to use it.
      while(info[tid].lhs_depth_users!=0) {}
      info[tid].lhs_depth_users = threads;

      // printf("thread %i pack A(%li:%li,%li:%li)\n", tid, info[tid].lhs_start,info[tid].lhs_start+info[tid].lhs_length, k, k+actual_kc);
      pack_lhs(blockA+info[tid].lhs_start*actual_kc, lhs.getSubMapper(info[tid].lhs_start,k), actual_kc, info[tid].lhs_length);

      // Notify the other threads that the part A'_i is ready to go.
      info[tid].lhs_depth_ready = k;

      // Computes C_i += A' * B' per A'_i
      for(int shift=0; shift<threads; ++shift)
      {
        Index i = (tid+shift)%threads;

        // At this point we have to make sure that A'_i has been updated by the thread i,
        // we use testAndSetOrdered to mimic a volatile access.
        // However, no need to wait for the B' part which has been updated by the current thread!
        if (shift>0) {
          while(info[i].lhs_depth_ready!=k) {
          }
        }

        // printf("thread %i C(%li:%li,%li:%li) += A(%li:%li,%li:%li) * B(%li:%li,%li:%li)\n", tid,
        //   info[i].lhs_start, info[i].lhs_start+info[i].lhs_length, info[tid].rhs_start, info[tid].rhs_start + nc,
        //   info[i].lhs_start,info[i].lhs_start+info[i].lhs_length, k, k+actual_kc,
        //   k, k+actual_kc, info[tid].rhs_start, info[tid].rhs_start + nc);
        gebp(res.getSubMapper(info[i].lhs_start, c0), blockA+info[i].lhs_start*actual_kc, blockB, info[i].lhs_length, actual_kc, nc, alpha);
      }

      // Then keep going as usual with the remaining B'
      for(Index j=nc; j<block_cols; j+=nc)
      {
        const Index actual_nc = (std::min)(j+nc,block_cols)-j;

        // pack B_k,j to B'
        pack_rhs(blockB, rhs.getSubMapper(k,c0+j), actual_kc, actual_nc);
        // printf("thread %i pack B(%li:%li,%li:%li)\n", tid, k, k+actual_kc, info[tid].rhs_start + j, info[tid].rhs_start + j + actual_nc);

        // C_j += A' * B'
        // printf("thread %i C(%li:%li,%li:%li) += A(%li:%li,%li:%li) * B(%li:%li,%li:%li)\n", tid,
        //   Index(0), rows, info[tid].rhs_start + j, info[tid].rhs_start + j+actual_nc, Index(0), rows, k, k+actual_kc, k, k+actual_kc,       
        //   info[tid].rhs_start + j, info[tid].rhs_start + j + actual_nc);
        gebp(res.getSubMapper(0,c0+j), blockA, blockB, rows, actual_kc, actual_nc, alpha);
      }

      // Release all the sub blocks A'_i of A' for the current thread,
      // i.e., we simply decrement the number of users by 1
      for(Index i=0; i<threads; ++i)
#if !EIGEN_HAS_CXX11_ATOMIC
        #pragma omp atomic
#endif
        info[i].lhs_depth_users -= 1;
    }
  }
#endif // EIGEN_HAS_OPENMP

#if EIGEN_HAS_CXX11_ATOMIC && EIGEN_PARALLEL_GEMM_STEALING
  static void run_parallel_stealing(Index rows, Index cols, Index depth,
    const LhsScalar* _lhs, Index lhsStride,
    const RhsScalar* _rhs, Index rhsStride,
    ResScalar* _res, Index resIncr, Index resStride,
    ResScalar alpha,
    level3_blocking<LhsScalar,RhsScalar>& blocking,
    GemmParallelInfo<Index>* info, Index shard_index)
  {
    LhsMapper lhs(_lhs, lhsStride);
    RhsMapper rhs(_rhs, rhsStride);
    ResMapper res(_res, resStride, resIncr);

    // this is the parallel version!
    // printf("Running stealing parallel GEMM.\n");
    const int tid = shard_index;
    const int threads = info[tid].shards;

    Index kc = blocking.kc();     // cache block size along the K direction
    // Index mc = blocking.mc();     // cache block size along the M direction
    Index nc = blocking.nc();     // cache block size along the N direction

    gemm_pack_lhs<LhsScalar, Index, LhsMapper, Traits::mr, Traits::LhsProgress, typename Traits::LhsPacket4Packing, LhsStorageOrder> pack_lhs;
    gemm_pack_rhs<RhsScalar, Index, RhsMapper, Traits::nr, RhsStorageOrder> pack_rhs;
    gebp_kernel<LhsScalar, RhsScalar, Index, ResMapper, Traits::mr, Traits::nr, ConjugateLhs, ConjugateRhs> gebp;
    // printf("thread %li blocking: %lix%lix%li\n", shard_index, mc, kc, nc);

    LhsScalar* blockA = blocking.blockA();
    eigen_internal_assert(blockA!=0);

    std::size_t sizeB = kc*nc;
    ei_declare_aligned_stack_constructed_variable(RhsScalar, blockB, sizeB, 0);

    // For each horizontal panel of the rhs, and corresponding vertical panel of the lhs...
    for(Index k=0; k<depth; k+=kc) {
      
      const Index actual_kc = (std::min)(k+kc,depth)-k; // => rows of B', and cols of the A'
      const Index next_k = k + actual_kc;
      
      for (Index rhs_shift=0; rhs_shift<threads; ++rhs_shift) {
        const Index rid = (tid + rhs_shift) % threads;
        
        // Start next column block within `rid` panel if not already started.
        Index current_k = k;
        if (info[rid].rhs_depth_next.compare_exchange_strong(current_k, next_k)) {
          
          // Wait until previous k finished.
          while(info[rid].rhs_depth_done < k) {
            // printf("thread %i waiting for depth[%li] %li\n", tid, rid, k);
          }
          
          const Index c0 = info[rid].rhs_start;
          const Index block_cols = info[rid].rhs_length;
          Index actual_nc = (std::min)(nc,block_cols);
      
          // In order to reduce the chance that a thread has to wait for the other,
          // let's start by packing B' = B_k,0.
          // printf("thread %i pack B(%li:%li,%li:%li)\n", tid, k, k+actual_kc, c0, c0+actual_nc);
          pack_rhs(blockB, rhs.getSubMapper(k,c0), actual_kc, actual_nc);

          // Pack A_k to A' in a parallel fashion:
          for (Index lhs_shift=0; lhs_shift < threads; ++lhs_shift) {
            Index lid = (rid + lhs_shift) % threads;
            const Index r0 = info[lid].lhs_start;
            const Index block_rows = info[lid].lhs_length;
            // printf("r0->block_rows: %li->%li\n", r0, block_rows);
            
            // Check if we need to pack A[lid].
            current_k = k;
            if (info[lid].lhs_depth_next.compare_exchange_strong(current_k, next_k)) {
              // Start packing LHS as soon as there are no users remaining.
              while(info[lid].lhs_depth_users!=0) {
                // printf("thread %i waiting for users[%li]\n", tid, lid);
              }
              // Set info[tid].users to the number of threads to mark that all
              // other threads are going to use it.
              info[lid].lhs_depth_users = threads;
              // printf("thread %i setting users[%li] = %i\n", tid, lid, threads);
              
              if (block_rows > 0) {
                // printf("thread %i pack A(%li:%li,%li:%li)\n", tid, r0, r0+block_rows, k, k+actual_kc);
                pack_lhs(blockA+r0*actual_kc, lhs.getSubMapper(r0,k), actual_kc, block_rows);
                // printf("thread %i packing A done.\n", tid);
              }
              
              // Notify the other threads that the part A'_i is ready to go.
              info[lid].lhs_depth_ready = k;
            } else {
              // Wait until LHS is ready.
              while(info[lid].lhs_depth_ready < k) {
                // printf("thread %i waiting for depth[%li] %li\n", tid, lid, k);
              } 
            }
            
            if (block_rows > 0) {
              // printf("thread %i C(%li:%li,%li:%li) += A(%li:%li,%li:%li) * B(%li:%li,%li:%li)\n", tid,
              //   r0, r0 + info[lid].lhs_length, c0, c0 + actual_nc,
              //   r0, r0 + info[lid].lhs_length, k, k+actual_kc,
              //   k, k+actual_kc, c0, c0 + actual_nc);
              gebp(res.getSubMapper(r0, c0), blockA+r0*actual_kc, blockB, block_rows, actual_kc, actual_nc, alpha);
              // printf("thread %i gebp done.\n", tid);
            }
            
          }

          // Then keep going as usual with the remaining B'. By this point, A'
          // is all-packed and ready to go.
          for(Index j=actual_nc; j<block_cols; j+=nc)
          {
            actual_nc = (std::min)(j+nc,block_cols)-j;

            // pack B_k,j to B'
            pack_rhs(blockB, rhs.getSubMapper(k,c0+j), actual_kc, actual_nc);
            // printf("thread %i pack B(%li:%li,%li:%li)\n", tid, k, k+actual_kc, c0 + j, c0 + j + actual_nc);

            // C_j += A' * B'
            // printf("thread %i C(%li:%li,%li:%li) += A(%li:%li,%li:%li) * B(%li:%li,%li:%li)\n", tid,
            //   Index(0), rows, c0 + j, c0 + j + actual_nc, Index(0), rows, k, k+actual_kc, k, k+actual_kc,       
            //   c0 + j, c0 + j + actual_nc);
            gebp(res.getSubMapper(0,c0+j), blockA, blockB, rows, actual_kc, actual_nc, alpha);
          }

          // Release all the sub blocks A'_i of A' for the current thread,
          // i.e., we simply decrement the number of users by 1
          for(Index i=0; i<threads; ++i) {
            info[i].lhs_depth_users -= 1;
            // int users = info[i].lhs_depth_users;
            // printf("thread %i reducing users[%li] = %i\n", tid, i, users);
          }

          // Notify we are ready for the next depth block.
          info[rid].rhs_depth_done = next_k;
        } // working on block
      } // rhs block
    } // depth k
  }
  #endif

  static void run_sequential(Index rows, Index cols, Index depth,
    const LhsScalar* _lhs, Index lhsStride,
    const RhsScalar* _rhs, Index rhsStride,
    ResScalar* _res, Index resIncr, Index resStride,
    ResScalar alpha,
    level3_blocking<LhsScalar,RhsScalar>& blocking) {

    LhsMapper lhs(_lhs, lhsStride);
    RhsMapper rhs(_rhs, rhsStride);
    ResMapper res(_res, resStride, resIncr);

    Index kc = blocking.kc();                   // cache block size along the K direction
    Index mc = (std::min)(rows,blocking.mc());  // cache block size along the M direction
    Index nc = (std::min)(cols,blocking.nc());  // cache block size along the N direction
    // printf("Size: %li x %li x %li\n", rows, depth, cols);
    // printf("Blocking: %li x %li x %li\n", mc, kc, nc);

    gemm_pack_lhs<LhsScalar, Index, LhsMapper, Traits::mr, Traits::LhsProgress, typename Traits::LhsPacket4Packing, LhsStorageOrder> pack_lhs;
    gemm_pack_rhs<RhsScalar, Index, RhsMapper, Traits::nr, RhsStorageOrder> pack_rhs;
    gebp_kernel<LhsScalar, RhsScalar, Index, ResMapper, Traits::mr, Traits::nr, ConjugateLhs, ConjugateRhs> gebp;

    // this is the sequential version!
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
        // printf("Packing block A(%li:%li,%li:%li)\n", i2, i2+actual_mc, k2, k2+actual_kc);
        pack_lhs(blockA, lhs.getSubMapper(i2,k2), actual_kc, actual_mc);

        // For each kc x nc block of the rhs's horizontal panel...
        for(Index j2=0; j2<cols; j2+=nc)
        {
          const Index actual_nc = (std::min)(j2+nc,cols)-j2;

          // We pack the rhs's block into a sequential chunk of memory (L2 caching)
          // Note that this block will be read a very high number of times, which is equal to the number of
          // micro horizontal panel of the large rhs's panel (e.g., rows/12 times).
          if((!pack_rhs_once) || i2==0) {
            // printf("Packing block B(%li:%li,%li:%li)\n", k2, k2+actual_kc, j2,j2+actual_nc);
            pack_rhs(blockB, rhs.getSubMapper(k2,j2), actual_kc, actual_nc);
          }

          // Everything is packed, we can now call the panel * block kernel:
          // printf("C(%li:%li,%li:%li) += A(%li:%li,%li:%li) * B(%li:%li,%li:%li)\n",
          //   i2, i2+actual_mc, j2, j2+actual_nc,
          //   i2, i2+actual_mc, k2, k2+actual_kc,
          //   k2, k2+actual_kc, j2, j2+actual_nc);
          gebp(res.getSubMapper(i2, j2), blockA, blockB, actual_mc, actual_kc, actual_nc, alpha);
        }
      }
    }
  }

};

/*********************************************************************************
*  Specialization of generic_product_impl for "large" GEMM, i.e.,
*  implementation of the high level wrapper to general_matrix_matrix_product
**********************************************************************************/

template<typename Scalar, typename Index, typename Gemm, typename Lhs, typename Rhs, typename Dest, typename BlockingType>
struct gemm_functor
{
  gemm_functor(const Lhs& lhs, const Rhs& rhs, Dest& dest, const Scalar& actualAlpha, BlockingType& blocking)
    : m_lhs(lhs), m_rhs(rhs), m_dest(dest), m_actualAlpha(actualAlpha), m_blocking(blocking)
  {}

  void initParallelSession(Index num_threads) const
  {
    m_blocking.initParallel(m_lhs.rows(), m_rhs.cols(), m_lhs.cols(), num_threads);
    m_blocking.allocateA();
  }

  void operator() (Index row, Index rows, Index col=0, Index cols=-1, GemmParallelInfo<Index>* info=0, Index shard_index=0) const
  {
    if(cols==-1)
      cols = m_rhs.cols();

    Gemm::run(rows, cols, m_lhs.cols(),
              &m_lhs.coeffRef(row,0), m_lhs.outerStride(),
              &m_rhs.coeffRef(0,col), m_rhs.outerStride(),
              (Scalar*)&(m_dest.coeffRef(row,col)), m_dest.innerStride(), m_dest.outerStride(),
              m_actualAlpha, m_blocking, info, shard_index);
  }

  typedef typename Gemm::Traits Traits;

  protected:
    const Lhs& m_lhs;
    const Rhs& m_rhs;
    Dest& m_dest;
    Scalar m_actualAlpha;
    BlockingType& m_blocking;
};

template<int StorageOrder, typename LhsScalar, typename RhsScalar, int MaxRows, int MaxCols, int MaxDepth, int KcFactor=1,
bool FiniteAtCompileTime = MaxRows!=Dynamic && MaxCols!=Dynamic && MaxDepth != Dynamic> class gemm_blocking_space;

template<typename LhsScalar_, typename RhsScalar_>
class level3_blocking
{
    typedef LhsScalar_ LhsScalar;
    typedef RhsScalar_ RhsScalar;

  protected:
    LhsScalar* m_blockA;
    RhsScalar* m_blockB;

    Index m_mc;
    Index m_nc;
    Index m_kc;

  public:

    level3_blocking()
      : m_blockA(0), m_blockB(0), m_mc(0), m_nc(0), m_kc(0)
    {}

    inline Index mc() const { return m_mc; }
    inline Index nc() const { return m_nc; }
    inline Index kc() const { return m_kc; }

    inline LhsScalar* blockA() { return m_blockA; }
    inline RhsScalar* blockB() { return m_blockB; }
};

template<int StorageOrder, typename LhsScalar_, typename RhsScalar_, int MaxRows, int MaxCols, int MaxDepth, int KcFactor>
class gemm_blocking_space<StorageOrder,LhsScalar_,RhsScalar_,MaxRows, MaxCols, MaxDepth, KcFactor, true /* == FiniteAtCompileTime */>
  : public level3_blocking<
      typename conditional<StorageOrder==RowMajor,RhsScalar_,LhsScalar_>::type,
      typename conditional<StorageOrder==RowMajor,LhsScalar_,RhsScalar_>::type>
{
    enum {
      Transpose = StorageOrder==RowMajor,
      ActualRows = Transpose ? MaxCols : MaxRows,
      ActualCols = Transpose ? MaxRows : MaxCols
    };
    typedef typename conditional<Transpose,RhsScalar_,LhsScalar_>::type LhsScalar;
    typedef typename conditional<Transpose,LhsScalar_,RhsScalar_>::type RhsScalar;
    typedef gebp_traits<LhsScalar,RhsScalar> Traits;
    enum {
      SizeA = ActualRows * MaxDepth,
      SizeB = ActualCols * MaxDepth
    };

#if EIGEN_MAX_STATIC_ALIGN_BYTES >= EIGEN_DEFAULT_ALIGN_BYTES
    EIGEN_ALIGN_MAX LhsScalar m_staticA[SizeA];
    EIGEN_ALIGN_MAX RhsScalar m_staticB[SizeB];
#else
    EIGEN_ALIGN_MAX char m_staticA[SizeA * sizeof(LhsScalar) + EIGEN_DEFAULT_ALIGN_BYTES-1];
    EIGEN_ALIGN_MAX char m_staticB[SizeB * sizeof(RhsScalar) + EIGEN_DEFAULT_ALIGN_BYTES-1];
#endif

  public:

    gemm_blocking_space(Index /*rows*/, Index /*cols*/, Index /*depth*/, Index /*num_threads*/, bool /*full_rows = false*/)
    {
      this->m_mc = ActualRows;
      this->m_nc = ActualCols;
      this->m_kc = MaxDepth;
#if EIGEN_MAX_STATIC_ALIGN_BYTES >= EIGEN_DEFAULT_ALIGN_BYTES
      this->m_blockA = m_staticA;
      this->m_blockB = m_staticB;
#else
      this->m_blockA = reinterpret_cast<LhsScalar*>((internal::UIntPtr(m_staticA) + (EIGEN_DEFAULT_ALIGN_BYTES-1)) & ~std::size_t(EIGEN_DEFAULT_ALIGN_BYTES-1));
      this->m_blockB = reinterpret_cast<RhsScalar*>((internal::UIntPtr(m_staticB) + (EIGEN_DEFAULT_ALIGN_BYTES-1)) & ~std::size_t(EIGEN_DEFAULT_ALIGN_BYTES-1));
#endif
    }

    void initParallel(Index, Index, Index, Index)
    {}

    inline void allocateA() {}
    inline void allocateB() {}
    inline void allocateAll() {}
};

template<int StorageOrder, typename LhsScalar_, typename RhsScalar_, int MaxRows, int MaxCols, int MaxDepth, int KcFactor>
class gemm_blocking_space<StorageOrder,LhsScalar_,RhsScalar_,MaxRows, MaxCols, MaxDepth, KcFactor, false>
  : public level3_blocking<
      typename conditional<StorageOrder==RowMajor,RhsScalar_,LhsScalar_>::type,
      typename conditional<StorageOrder==RowMajor,LhsScalar_,RhsScalar_>::type>
{
    enum {
      Transpose = StorageOrder==RowMajor
    };
    typedef typename conditional<Transpose,RhsScalar_,LhsScalar_>::type LhsScalar;
    typedef typename conditional<Transpose,LhsScalar_,RhsScalar_>::type RhsScalar;
    typedef gebp_traits<LhsScalar,RhsScalar> Traits;

    Index m_sizeA;
    Index m_sizeB;

  public:

    gemm_blocking_space(Index rows, Index cols, Index depth, Index num_threads, bool l3_blocking)
    {
      this->m_mc = Transpose ? cols : rows;
      this->m_nc = Transpose ? rows : cols;
      this->m_kc = depth;

      if(l3_blocking)
      {
        computeProductBlockingSizes<LhsScalar,RhsScalar,KcFactor>(this->m_kc, this->m_mc, this->m_nc, num_threads);
      }
      else  // no l3 blocking
      {
        Index n = this->m_nc;
        computeProductBlockingSizes<LhsScalar,RhsScalar,KcFactor>(this->m_kc, this->m_mc, n, num_threads);
      }

      m_sizeA = this->m_mc * this->m_kc;
      m_sizeB = this->m_kc * this->m_nc;
    }

    void initParallel(Index rows, Index cols, Index depth, Index num_threads)
    {
      this->m_mc = Transpose ? cols : rows;
      this->m_nc = Transpose ? rows : cols;
      this->m_kc = depth;

      eigen_internal_assert(this->m_blockA==0 && this->m_blockB==0);
      Index m = this->m_mc;
      computeProductBlockingSizes<LhsScalar,RhsScalar,KcFactor>(this->m_kc, m, this->m_nc, num_threads);
      m_sizeA = this->m_mc * this->m_kc;
      m_sizeB = this->m_kc * this->m_nc;
    }

    void allocateA()
    {
      if(this->m_blockA==0)
        this->m_blockA = aligned_new<LhsScalar>(m_sizeA);
    }

    void allocateB()
    {
      if(this->m_blockB==0)
        this->m_blockB = aligned_new<RhsScalar>(m_sizeB);
    }

    void allocateAll()
    {
      allocateA();
      allocateB();
    }

    ~gemm_blocking_space()
    {
      aligned_delete(this->m_blockA, m_sizeA);
      aligned_delete(this->m_blockB, m_sizeB);
    }
};

} // end namespace internal

namespace internal {

template<typename Lhs, typename Rhs>
struct generic_product_impl<Lhs,Rhs,DenseShape,DenseShape,GemmProduct>
  : generic_product_impl_base<Lhs,Rhs,generic_product_impl<Lhs,Rhs,DenseShape,DenseShape,GemmProduct> >
{
  typedef typename Product<Lhs,Rhs>::Scalar Scalar;
  typedef typename Lhs::Scalar LhsScalar;
  typedef typename Rhs::Scalar RhsScalar;

  typedef internal::blas_traits<Lhs> LhsBlasTraits;
  typedef typename LhsBlasTraits::DirectLinearAccessType ActualLhsType;
  typedef typename internal::remove_all<ActualLhsType>::type ActualLhsTypeCleaned;

  typedef internal::blas_traits<Rhs> RhsBlasTraits;
  typedef typename RhsBlasTraits::DirectLinearAccessType ActualRhsType;
  typedef typename internal::remove_all<ActualRhsType>::type ActualRhsTypeCleaned;

  enum {
    MaxDepthAtCompileTime = EIGEN_SIZE_MIN_PREFER_FIXED(Lhs::MaxColsAtCompileTime,Rhs::MaxRowsAtCompileTime)
  };

  typedef generic_product_impl<Lhs,Rhs,DenseShape,DenseShape,CoeffBasedProductMode> lazyproduct;

  template<typename Dst>
  static void evalTo(Dst& dst, const Lhs& lhs, const Rhs& rhs)
  {
    // See http://eigen.tuxfamily.org/bz/show_bug.cgi?id=404 for a discussion and helper program
    // to determine the following heuristic.
    // EIGEN_GEMM_TO_COEFFBASED_THRESHOLD is typically defined to 20 in GeneralProduct.h,
    // unless it has been specialized by the user or for a given architecture.
    // Note that the condition rhs.rows()>0 was required because lazy product is (was?) not happy with empty inputs.
    // I'm not sure it is still required.
    if((rhs.rows()+dst.rows()+dst.cols())<EIGEN_GEMM_TO_COEFFBASED_THRESHOLD && rhs.rows()>0)
      lazyproduct::eval_dynamic(dst, lhs, rhs, internal::assign_op<typename Dst::Scalar,Scalar>());
    else
    {
      dst.setZero();
      scaleAndAddTo(dst, lhs, rhs, Scalar(1));
    }
  }

  template<typename Dst>
  static void addTo(Dst& dst, const Lhs& lhs, const Rhs& rhs)
  {
    if((rhs.rows()+dst.rows()+dst.cols())<EIGEN_GEMM_TO_COEFFBASED_THRESHOLD && rhs.rows()>0)
      lazyproduct::eval_dynamic(dst, lhs, rhs, internal::add_assign_op<typename Dst::Scalar,Scalar>());
    else
      scaleAndAddTo(dst,lhs, rhs, Scalar(1));
  }

  template<typename Dst>
  static void subTo(Dst& dst, const Lhs& lhs, const Rhs& rhs)
  {
    if((rhs.rows()+dst.rows()+dst.cols())<EIGEN_GEMM_TO_COEFFBASED_THRESHOLD && rhs.rows()>0)
      lazyproduct::eval_dynamic(dst, lhs, rhs, internal::sub_assign_op<typename Dst::Scalar,Scalar>());
    else
      scaleAndAddTo(dst, lhs, rhs, Scalar(-1));
  }

  template<typename Dest>
  static void scaleAndAddTo(Dest& dst, const Lhs& a_lhs, const Rhs& a_rhs, const Scalar& alpha)
  {
    eigen_assert(dst.rows()==a_lhs.rows() && dst.cols()==a_rhs.cols());
    if(a_lhs.cols()==0 || a_lhs.rows()==0 || a_rhs.cols()==0)
      return;

    if (dst.cols() == 1)
    {
      // Fallback to GEMV if either the lhs or rhs is a runtime vector
      typename Dest::ColXpr dst_vec(dst.col(0));
      return internal::generic_product_impl<Lhs,typename Rhs::ConstColXpr,DenseShape,DenseShape,GemvProduct>
        ::scaleAndAddTo(dst_vec, a_lhs, a_rhs.col(0), alpha);
    }
    else if (dst.rows() == 1)
    {
      // Fallback to GEMV if either the lhs or rhs is a runtime vector
      typename Dest::RowXpr dst_vec(dst.row(0));
      return internal::generic_product_impl<typename Lhs::ConstRowXpr,Rhs,DenseShape,DenseShape,GemvProduct>
        ::scaleAndAddTo(dst_vec, a_lhs.row(0), a_rhs, alpha);
    }

    typename internal::add_const_on_value_type<ActualLhsType>::type lhs = LhsBlasTraits::extract(a_lhs);
    typename internal::add_const_on_value_type<ActualRhsType>::type rhs = RhsBlasTraits::extract(a_rhs);

    Scalar actualAlpha = combine_scalar_factors(alpha, a_lhs, a_rhs);

    typedef internal::gemm_blocking_space<(Dest::Flags&RowMajorBit) ? RowMajor : ColMajor,LhsScalar,RhsScalar,
            Dest::MaxRowsAtCompileTime,Dest::MaxColsAtCompileTime,MaxDepthAtCompileTime> BlockingType;

    typedef internal::gemm_functor<
      Scalar, Index,
      internal::general_matrix_matrix_product<
        Index,
        LhsScalar, (ActualLhsTypeCleaned::Flags&RowMajorBit) ? RowMajor : ColMajor, bool(LhsBlasTraits::NeedToConjugate),
        RhsScalar, (ActualRhsTypeCleaned::Flags&RowMajorBit) ? RowMajor : ColMajor, bool(RhsBlasTraits::NeedToConjugate),
        (Dest::Flags&RowMajorBit) ? RowMajor : ColMajor,
        Dest::InnerStrideAtCompileTime>,
      ActualLhsTypeCleaned, ActualRhsTypeCleaned, Dest, BlockingType> GemmFunctor;

    BlockingType blocking(dst.rows(), dst.cols(), lhs.cols(), 1, true);
    internal::parallelize_gemm<(Dest::MaxRowsAtCompileTime>32 || Dest::MaxRowsAtCompileTime==Dynamic)>
        (GemmFunctor(lhs, rhs, dst, actualAlpha, blocking), a_lhs.rows(), a_rhs.cols(), a_lhs.cols(), Dest::Flags&RowMajorBit);
  }
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_GENERAL_MATRIX_MATRIX_H
