// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2021 The Eigen Authors.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

namespace Eigen {
  
/** 
 * \class Parallizer
 * \ingroup Core_Module
 * 
 * Interface for a general-purpose parallelizer.
 */
template<typename Derived>
class Parallelizer {
 public:
  /** 
   * Parallelize a job.
   * 
   * \param func an input function of two parameters (`i`, `n`).  Internally,
   *        the method will select a total number of shards `n`, and call
   *        `func(i, n)` for `i`=0..`n`-1.  The number of shards \a may be
   *        the number of threads.
   * \param max_shards the maximum number of shards to use. The total number
   *        of actual shards may be fewer.  If zero, this maximum will be
   *        ignored, and the number of shards will be determined without a
   *        bound.
   */
  template<typename Func>
  void EIGEN_DEVICE_FUNC parallelize(const Func& func, Index max_shards = 0) {
    derived().parallelize(func, max_shards);
  }

  /**
   * Parallelize a for-loop.
   * 
   * The loop will be sharded into a set of (mostly) fixed-sized batches.
   * 
   * \param body loop-body with arguments `(i)`, where `i` is the loop
   *             iteration.
   * \param n    total number of iterations.
   * \param batch_size maximum batch size.  If zero, a default will be computed
   *                   based on the total amount of work.
   */
  template<typename Body>
  void EIGEN_DEVICE_FUNC parallel_for(const Body& body, Index n,
      Index batch_size = 0) {
    return derived().parallel_for(body, n, batch_size);
  }
  
  /** \returns a reference to the derived object */
  Derived& derived() { return *static_cast<Derived*>(this); }
  
  /** \returns a const reference to the derived object */
  const Derived& derived() const { return *static_cast<const Derived*>(this); }
  
};

/**
 * \class ParallelizerSelecor
 * \ingroup Core_Module
 * 
 * Selects a parallelizer type by integer ID.
 * 
 * Each parallelizer should be assigned a unique integer ID, and specialize this
 * struct to select that type based on that ID via
 * 
 *     typedef typename ParallizerSelector<id>::type ParallizerType;
 * 
 * Eigen's internals will use the ID `EIGEN_DEFAULT_PARALLELIZER`. This will
 * default to use OpenMP if available, otherwise will use a basic serial
 * implementation.
 */
template<int ParallelizerSelection>
struct ParallelizerSelector;

#ifdef EIGEN_HAS_OPENMP

class OpenMpParallelizer : Parallelizer<OpenMpParallelizer>{
 public:

  template<typename Func>
  void parallelize(const Func& func, Index max_shards = 0) {
    const Index threads = (max_shards == 0) ? nbThreads() : max_shards;
    #pragma omp parallel num_threads(threads)
    {
      Index i = omp_get_thread_num();
      Index n = omp_get_num_threads();
      func(i, n);
    }
  }

  template<typename Body>
  void parallel_for(const Body& body, Index n, Index batch_size = 0) {
    const Index threads = nbThreads();
    // Ceil so we're not left with a small batch at the end.
    const Index max_over_sharding = 4;
    const Index k = (batch_size == 0) ? 
        (n - max_over_sharding * threads + 1) / (max_over_sharding * threads) 
        : batch_size;
    
    #pragma omp parallel for schedule(dynamic, k) num_threads(threads)
    for(Index i=0; i<n; ++i) {
      body(i);
    }
  }

};

// OpenMpParallelizer identifier.
#ifndef EIGEN_OPENMP_PARALLELIZER
#define EIGEN_OPENMP_PARALLELIZER 1
#endif
template<>
struct ParallelizerSelector<EIGEN_OPENMP_PARALLELIZER> {
  typedef OpenMpParallelizer type;
};

// Default to OpenMP if available.
#ifndef EIGEN_DEFAULT_PARALLELIZER
#define EIGEN_DEFAULT_PARALLELIZER EIGEN_OPENMP_PARALLELIZER
#endif

#endif

/**
 * \class SerialParallelizer
 * \ingroup Core_Module
 * 
 * Default non-parallizer, runs everything serially.
 */
class SerialParallelizer : Parallelizer<SerialParallelizer>{
 public:

  template<typename Func>
  void parallelize(const Func& func, Index max_shards = 0) {
    func(/*i=*/0, /*n=*/1);
  }

  template<typename Body>
  void parallel_for(const Body& body, Index n, Index batch_size = 0) {
    for (Index i=0; i<n; ++i) {
      body(i);
    }
  }
};


// SerialParallelizer identifier.
#ifndef EIGEN_SERIAL_PARALLELIZER
#define EIGEN_SERIAL_PARALLELIZER 0
#endif
template<>
struct ParallelizerSelector<EIGEN_SERIAL_PARALLELIZER> {
  typedef SerialParallelizer type;
};

// Default to Serial if not previously configured.
#ifndef EIGEN_DEFAULT_PARALLELIZER
#define EIGEN_DEFAULT_PARALLELIZER EIGEN_SERIAL_PARALLELIZER
#endif

}  // namespace Eigen
