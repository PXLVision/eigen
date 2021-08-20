// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015-2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/Dense>

template<typename T>
struct CoeffwiseKernel {
  EIGEN_DEVICE_FUNC void operator()(typename T::Scalar s, const T& x1,
                                    const T& x2, const T& x3, T& out) const {
    out.array() += (s * x1 + x2).array() * x3.array();
  }
};

template<typename T>
void coeffwise_test(T type) {
  using Scalar = typename T::Scalar;
  const Index rows = type.rows();
  const Index cols = type.cols();
  
  // Initialize random inputs.
  const Scalar s = Eigen::internal::random<Scalar>();
  const T x1 = T::Random(rows, cols);
  const T x2 = T::Random(rows, cols);
  const T x3 = T::Random(rows, cols);
  
  // Intialize outputs.
  T out_cpu = T::Random(rows, cols);
  T out_gpu = out_cpu;
  
  // Call kernel and compare.
  CoeffwiseKernel<T> coeffwise_kernel;
  run_on_cpu(coeffwise_kernel, s, x1, x2, x3, out_cpu);
  run_on_gpu(coeffwise_kernel, s, x1, x2, x3, out_gpu);
  VERIFY_IS_APPROX(out_cpu, out_gpu);
};

template<typename T>
struct SqrtKernel {
  EIGEN_DEVICE_FUNC T operator()(const T& x1) const {
    return x1.cwiseSqrt();
  }
};

template<typename T>
void complex_sqrt_test(const T& type) {
  using Scalar = typename T::Scalar;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  const Index rows = type.rows();
  const Index cols = type.cols();
  
  // Random inputs.
  const T x = T::Random(rows, cols);
  SqrtKernel<T> sqrt_kernel;
  VERIFY_IS_APPROX(run_on_cpu(sqrt_kernel, x),
                   run_on_gpu(sqrt_kernel, x));
  
  // Edge-cases.
  {
    const RealScalar nan = Eigen::NumTraits<RealScalar>::quiet_NaN();
    const RealScalar inf = Eigen::NumTraits<RealScalar>::infinity();
    Eigen::Vector<Scalar, Eigen::Dynamic> edges(18);
    edges.setZero();
  
    int idx = 0;
    edges[idx++] = Scalar(0, 0);
    edges[idx++] = Scalar(-0, 0);
    edges[idx++] = Scalar(0, -0);
    edges[idx++] = Scalar(-0, -0);
    edges[idx++] = Scalar(1.0, inf);
    edges[idx++] = Scalar(nan, inf);
    edges[idx++] = Scalar(1.0, -inf);
    edges[idx++] = Scalar(nan, -inf);
    edges[idx++] = Scalar(-inf, 1.0);
    edges[idx++] = Scalar(inf, 1.0);
    edges[idx++] = Scalar(-inf, -1.0);
    edges[idx++] = Scalar(inf, -1.0);
    edges[idx++] = Scalar(-inf, nan);
    edges[idx++] = Scalar(inf, nan);
    edges[idx++] = Scalar(1.0, nan);
    edges[idx++] = Scalar(nan, 1.0);
    edges[idx++] = Scalar(nan, -1.0);
    edges[idx++] = Scalar(nan, nan);
    
    SqrtKernel<decltype(edges)> sqrt_kernel_edges;                     
    VERIFY_IS_CWISE_APPROX(run_on_cpu(sqrt_kernel_edges, edges),
                           run_on_gpu(sqrt_kernel_edges, edges));
  }
}

template<typename T>
struct ComplexOperatorsKernel {
  using Scalar = typename T::Scalar;
  EIGEN_DEVICE_FUNC Eigen::MatrixX<Scalar> operator()(const T& A, const T& B) {
    // Block the output into chunks.
    const Index num_operators = 19;
    const Index rows = A.rows();
    const Index cols = A.cols();
    Eigen::MatrixX<Scalar> out(num_operators * rows, cols);
    out.setZero();
    
    // Negation.
    Index block = 0;
    out.topRows(rows) = -A;
    
    // Addition.
    block += rows;
    out.middleRows(block, rows) = A + B;
    block += rows;
    out.middleRows(block, rows) = A + B.real();
    block += rows;
    out.middleRows(block, rows) = A.real() + B;
    
    // Subtraction.
    block += rows;
    out.middleRows(block, rows) = A - B;
    block += rows;
    out.middleRows(block, rows) = A - B.real();
    block += rows;
    out.middleRows(block, rows) = A.real() - B;
    
    // Multiplication.
    block += rows;
    out.middleRows(block, rows) = A.array() *  B.array();
    block += rows;
    out.middleRows(block, rows) = A.array() *  B.real().array();
    block += rows;
    out.middleRows(block, rows) = A.real().array() *  B.array();
    
    // Division.
    block += rows;
    out.middleRows(block, rows) = A.array() /  B.array();
    block += rows;
    out.middleRows(block, rows) = A.array() /  B.real().array();
    block += rows;
    out.middleRows(block, rows) = A.real().array() /  B.array();

    // Compound assignments.
    block += rows;
    out.middleRows(block, rows) = A;
    out.middleRows(block, rows) += B;
    block += rows;
    out.middleRows(block, rows) = A;
    out.middleRows(block, rows) -= B;
    block += rows;
    out.middleRows(block, rows) = A;
    out.middleRows(block, rows).array() *= B.array();
    block += rows;
    out.middleRows(block, rows) = A;
    out.middleRows(block, rows).array() /= B.array();

    // Comparisons.
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    const T true_value = T::Constant(rows, cols, Scalar(RealScalar(1), RealScalar(1)));
    const T false_value = T::Constant(rows, cols, Scalar(RealScalar(0), RealScalar(0)));
    block += rows;
    out.middleRows(block, rows) = A == B ? true_value : false_value;
    block += rows;
    out.middleRows(block, rows) = A != B ? true_value : false_value;
   
    return out;
  }
};

template<typename T>
void complex_operators_test(const T& type) {
  const Index rows = type.rows();
  const Index cols = type.cols();
  
  const T x1 = T::Random(rows, cols);
  const T x2 = T::Random(rows, cols);
  
  ComplexOperatorsKernel<T> kernel;
  VERIFY_IS_APPROX(run_on_cpu(kernel, x1, x2),
                   run_on_gpu(kernel, x1, x2));
}

template<typename T>
struct ReplicateKernel {
  EIGEN_DEVICE_FUNC void operator()(typename T::Scalar s, const T& in, T& out1, T& out2, T& out3) const {
    out1 = in.replicate(2, 2);
    out2 = s * in.colwise().replicate(3);
    out3 = s * in.rowwise().replicate(3);
  }
};

template<typename T>
void replicate_test(const T& type) {
  using Scalar = typename T::Scalar;
  const Index rows = type.rows();
  const Index cols = type.cols();
  
  const T x = T::Random(rows, cols);
  const Scalar s = Eigen::internal::random<Scalar>();
  T out_cpu0, out_cpu1, out_cpu2;
  T out_gpu0, out_gpu1, out_gpu2;
  
  ReplicateKernel<T> kernel;
  run_on_cpu(kernel, s, x, out_cpu0, out_cpu1, out_cpu2);
  run_on_gpu(kernel, s, x, out_gpu0, out_gpu1, out_gpu2);
  VERIFY_IS_APPROX(out_cpu0, out_gpu0);
  VERIFY_IS_APPROX(out_cpu1, out_gpu1);
  VERIFY_IS_APPROX(out_cpu2, out_gpu2);
}

// Check that new/delete work on device.
template<typename T>
struct AllocNewDeleteKernel {
  EIGEN_DEVICE_FUNC void operator()(size_t size, typename T::Scalar value) const {
    // Single element.
    T* x = new T();
    x->setConstant(value);
    delete x;
    
    // Array.
    T* y = new T[size];
    for (size_t i=0; i<size; ++i) {
      y[i].setConstant(value);
    }
    delete[] y;
  }
};

template<typename T>
void alloc_new_delete_test(const T&, size_t size) {
  using Scalar = typename T::Scalar;
  const Scalar value = Eigen::internal::random<Scalar>();
  AllocNewDeleteKernel<T> kernel;
  run_on_gpu(kernel, size, value);
}


template<typename T>
struct ReduxKernel {
  using Scalar = typename T::Scalar;
  using ResultType = Eigen::Vector<Scalar, 9>;
  EIGEN_DEVICE_FUNC ResultType operator()(const T& x1) const {
    ResultType out;
    out[0] = x1.minCoeff();
    out[1] = x1.maxCoeff();
    out[2] = x1.sum();
    out[3] = x1.prod();
    out[4] = x1.matrix().squaredNorm();
    out[5] = x1.matrix().norm();
    out[6] = x1.colwise().sum().maxCoeff();
    out[7] = x1.rowwise().maxCoeff().sum();
    out[8] = x1.matrix().colwise().squaredNorm().sum();
    return out;
  }
};

template<typename T>
void redux_test(const T& type) {
  const Index rows = type.rows();
  const Index cols = type.cols();
  const T x = T::Random(rows, cols);
  ReduxKernel<T> kernel;
  VERIFY_IS_APPROX(run_on_cpu(kernel, x),
                   run_on_gpu(kernel, x));
}

template<typename T1, typename T2, typename T3>
struct ProductKernel {
  using Scalar = typename T1::Scalar;
  EIGEN_DEVICE_FUNC void operator()(Scalar alpha, const T1& x1, const T2& x2, T3& x3) {
    x3 += alpha * x1 * x2;
  }
};

template<typename T1, typename T2>
void product_test(const T1& type1, const T2& type2) {
  using Scalar = typename T1::Scalar;
  using T3 = Eigen::Matrix<Scalar, T1::RowsAtCompileTime, T2::ColsAtCompileTime>;
  const Index rows = type1.rows();
  const Index depth = type1.cols();
  const Index cols = type2.cols();
  
  const Scalar alpha = Eigen::internal::random<Scalar>();
  const T1 x1 = T1::Random(rows, depth);
  const T2 x2 = T2::Random(depth, cols);
  T3 out_cpu = T3::Random(rows, cols);
  T3 out_gpu = out_cpu;
  
  ProductKernel<T1, T2, T3> kernel;
  run_on_cpu(kernel, alpha, x1, x2, out_cpu);
  run_on_gpu(kernel, alpha, x1, x2, out_gpu);
  VERIFY_IS_APPROX(out_cpu, out_gpu);
}

template<typename T1, typename T2>
struct DiagonalKernel {  
  EIGEN_DEVICE_FUNC void operator()(const T1& x1, T2& out) const {
    out += x1.diagonal();
  }
};

template<typename T1, typename T2>
void diagonal_test(const T1& type1, const T2&) {
  const Index size = type1.rows();
  const T1 x1 = T1::Random(size, size);
  T2 out_cpu = T2::Random(size);
  T2 out_gpu = out_cpu;
  
  DiagonalKernel<T1, T2> kernel;
  run_on_cpu(kernel, x1, out_cpu);
  run_on_gpu(kernel, x1, out_gpu);
  VERIFY_IS_APPROX(out_cpu, out_gpu);
}

template<typename T>
struct EigenvaluesDirectKernel {
  using ResultType = typename Eigen::SelfAdjointEigenSolver<T>::RealVectorType;
  EIGEN_DEVICE_FUNC ResultType operator()(const T& M) const
  {
    T A = M * M.adjoint();
    Eigen::SelfAdjointEigenSolver<T> eig;
    eig.computeDirect(A);
    return eig.eigenvalues();
  }
};

template<typename T>
void eigenvalues_direct_test(const T& type) {
  const Index rows = type.rows();
  const Index cols = type.cols();
  const T M = T::Random(rows, cols);
  EigenvaluesDirectKernel<T> kernel;
  VERIFY_IS_APPROX(run_on_cpu(kernel, M), run_on_gpu(kernel, M));
}

template<typename T>
struct EigenvaluesKernel {
  using ResultType = typename Eigen::SelfAdjointEigenSolver<T>::RealVectorType;
  EIGEN_DEVICE_FUNC ResultType operator()(const T& M) const
  {
    T A = M * M.adjoint();
    Eigen::SelfAdjointEigenSolver<T> eig;
    eig.compute(A);
    return eig.eigenvalues();
  }
};

template<typename T>
void eigenvalues_test(const T& type) {
  const Index rows = type.rows();
  const Index cols = type.cols();
  const T M = T::Random(rows, cols);
  EigenvaluesKernel<T> kernel;
  VERIFY_IS_APPROX(run_on_cpu(kernel, M), run_on_gpu(kernel, M));
}

template<typename T>
struct MatrixInverseKernel {
  EIGEN_DEVICE_FUNC T operator()(const T& M) const {
    return M.inverse();
  }
};

template<typename T>
void matrix_inverse_test(const T& type) {
  const Index rows = type.rows();
  const Index cols = type.cols();
  const T M = T::Random(rows, cols);
  MatrixInverseKernel<T> kernel;
  VERIFY_IS_APPROX(run_on_cpu(kernel, M), run_on_gpu(kernel, M));
}

template<typename Scalar>
struct NumericLimitsKernel {
  using ReturnType = Eigen::Vector<Scalar, 5>;
  EIGEN_DEVICE_FUNC ReturnType operator()() const
  {
    ReturnType out(5);
    out[0] = numext::numeric_limits<Scalar>::epsilon();
    out[1] = (numext::numeric_limits<Scalar>::max)();
    out[2] = (numext::numeric_limits<Scalar>::min)();
    out[3] = numext::numeric_limits<Scalar>::infinity();
    out[4] = numext::numeric_limits<Scalar>::quiet_NaN();
    return out;
  }
};

template<typename Scalar>
void numeric_limits_test() {
  NumericLimitsKernel<Scalar> kernel;
  // Verify all coefficients are equal or both NaN.
  VERIFY_IS_CWISE_EQUAL(run_on_cpu(kernel), run_on_gpu(kernel));
}

EIGEN_DECLARE_TEST(gpu_basic)
{
  print_gpu_device_info();
  
  for(int i = 0; i < g_repeat; i++) {    
    CALL_SUBTEST( coeffwise_test(Eigen::Vector3f()) );
    CALL_SUBTEST( coeffwise_test(Eigen::Array44f()) );
    CALL_SUBTEST( coeffwise_test(Eigen::ArrayXd(20)) );
    CALL_SUBTEST( coeffwise_test(Eigen::ArrayXXd(13, 17)) );

    #if !defined(EIGEN_USE_HIP)
    // FIXME
    // These subtests result in a compile failure on the HIP platform
    //
    //  eigen-upstream/Eigen/src/Core/Replicate.h:61:65: error:
    //           base class 'internal::dense_xpr_base<Replicate<Array<float, 4, 1, 0, 4, 1>, -1, -1> >::type'
    //           (aka 'ArrayBase<Eigen::Replicate<Eigen::Array<float, 4, 1, 0, 4, 1>, -1, -1> >') has protected default constructor
    CALL_SUBTEST( replicate_test(Eigen::Array33f()) );
    CALL_SUBTEST( replicate_test(Eigen::ArrayXXf(13, 17)) );
    CALL_SUBTEST( replicate_test(Eigen::Array4f()) );
    CALL_SUBTEST( replicate_test(Eigen::MatrixXd(13, 17)) );
    // HIP does not support new/delete on device.
    CALL_SUBTEST( alloc_new_delete_test(Eigen::Vector3f(), 10) );
    CALL_SUBTEST( alloc_new_delete_test(Eigen::Matrix3d(), 7) );
    #endif // EIGEN_USE_HIP

    CALL_SUBTEST( redux_test(Eigen::Array4f()) );
    CALL_SUBTEST( redux_test(Eigen::Matrix3f()) );
    
    CALL_SUBTEST( complex_sqrt_test(Eigen::Vector3cf()) );    
    CALL_SUBTEST( complex_sqrt_test(Eigen::ArrayXX<std::complex<float> >(13, 17)) );    
    CALL_SUBTEST( complex_sqrt_test(Eigen::ArrayXX<std::complex<double> >(13, 17)) );

    CALL_SUBTEST( complex_operators_test(Eigen::Vector3cf()) );    
    CALL_SUBTEST( complex_operators_test(Eigen::MatrixX<std::complex<float> >(13, 17)) );    
    CALL_SUBTEST( complex_operators_test(Eigen::MatrixX<std::complex<double> >(13, 17)) );

    CALL_SUBTEST( product_test(Matrix3f(), Matrix3f()) );
    CALL_SUBTEST( product_test(Matrix4f(), Vector4f()) );
    
    CALL_SUBTEST( diagonal_test(Matrix3f(), Vector3f()) );
    CALL_SUBTEST( diagonal_test(Matrix4f(), Vector4f()) );

    CALL_SUBTEST( matrix_inverse_test(Matrix2f()) );
    CALL_SUBTEST( matrix_inverse_test(Matrix3f()) );
    CALL_SUBTEST( matrix_inverse_test(Matrix4f()) );
      
    CALL_SUBTEST( eigenvalues_direct_test(Matrix3f()) );
    CALL_SUBTEST( eigenvalues_direct_test(Matrix2f()) );
    
    #if !defined(EIGEN_USE_HIP) && !EIGEN_COMP_CLANG
    // FIXME
    // These subtests compiles only with nvcc and fail with HIPCC and clang-cuda
    CALL_SUBTEST( eigenvalues_test(Matrix4f()) );
    CALL_SUBTEST( eigenvalues_test(Matrix<float,6,6>()) );
    #endif

    // numeric_limits
    CALL_SUBTEST( (numeric_limits_test<float>()) );
    CALL_SUBTEST( (numeric_limits_test<double>()) );
  }
}
