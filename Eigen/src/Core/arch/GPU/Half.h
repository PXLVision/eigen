// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


// IEEE 16-bit float type for GPUs if fp16 is available.

#ifndef EIGEN_GPU_HALF_H
#define EIGEN_GPU_HALF_H

#if defined(EIGEN_HAS_GPU_FP16)

// FP16 math available.
#if defined(EIGEN_HIP_DEVICE_COMPILE) || \
  (defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530)
#define EIGEN_GPU_HAS_FP16_ARITHMETIC 1
#endif
#if defined(EIGEN_HIP_DEVICE_COMPILE) || \
  (EIGEN_CUDA_SDK_VER > 80000 && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530)
#define EIGEN_GPU_HAS_FP16_MATH_FUNCTIONS 1
#endif

namespace Eigen {

namespace half_impl {
  
// Simple struct so we can re-use implementation yet still overload operators.
struct HalfFloat : public __half {
  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC HalfFloat() : __half() {}
  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC HalfFloat(const __half& h) : __half(h) {}
  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC HalfFloat(const HalfFloat& h) : __half(h) {}
  using __half::operator =; \
  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC HalfFloat& operator=(const HalfFloat& other) {
    __half::operator=(other); 
    return *this;
  }
     

};

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
numext::uint16_t raw_half_as_uint16(const __half& h) {
#ifdef EIGEN_GPU_COMPILE_PHASE
  return __half_as_ushort(h);
#else
  // This is what CUDA does internally.
  // return *(reinterpret_cast<const numext::uint16_t*>(&(h)));
  // Alternative type-alias-safe approach.
  EIGEN_USING_STD(memcpy)
  numext::uint16_t out;
  memcpy(&out, &h, sizeof(out));
  return out;
#endif
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
__half raw_uint16_as_half(numext::uint16_t bits) {
#ifdef EIGEN_GPU_COMPILE_PHASE
  return __ushort_as_half(bits);
#else
  // This is what CUDA does internally.
  // return *(reinterpret_cast<const numext::uint16_t*>(&(half)));
  // type-alias-safe approach.
  EIGEN_USING_STD(memcpy)
  __half out;
  memcpy(&out, &bits, sizeof(out));
  return out;
#endif
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
float half_to_float(const __half& h) {
  return __half2float(h);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
__half float_to_half(const __half& h) {
  return __float2half(h);
}

}  // namespace half_impl

typedef half_impl::HalfFloat half;

}  // namespace Eigen

namespace std {
template<>
struct numeric_limits<Eigen::half> {
  static const bool is_specialized = true;
  static const bool is_signed = true;
  static const bool is_integer = false;
  static const bool is_exact = false;
  static const bool has_infinity = true;
  static const bool has_quiet_NaN = true;
  static const bool has_signaling_NaN = true;
  static const float_denorm_style has_denorm = denorm_present;
  static const bool has_denorm_loss = false;
  static const std::float_round_style round_style = std::round_to_nearest;
  static const bool is_iec559 = false;
  static const bool is_bounded = false;
  static const bool is_modulo = false;
  static const int digits = 11;
  static const int digits10 = 3;      // according to http://half.sourceforge.net/structstd_1_1numeric__limits_3_01half__float_1_1half_01_4.html
  static const int max_digits10 = 5;  // according to http://half.sourceforge.net/structstd_1_1numeric__limits_3_01half__float_1_1half_01_4.html
  static const int radix = 2;
  static const int min_exponent = -13;
  static const int min_exponent10 = -4;
  static const int max_exponent = 16;
  static const int max_exponent10 = 4;
  static const bool traps = true;
  static const bool tinyness_before = false;

  static Eigen::half (min)() { return Eigen::half_impl::raw_uint16_as_half(0x400); }
  static Eigen::half lowest() { return Eigen::half_impl::raw_uint16_as_half(0xfbff); }
  static Eigen::half (max)() { return Eigen::half_impl::raw_uint16_as_half(0x7bff); }
  static Eigen::half epsilon() { return Eigen::half_impl::raw_uint16_as_half(0x0800); }
  static Eigen::half round_error() { return Eigen::half_impl::float_to_half(0.5); }
  static Eigen::half infinity() { return Eigen::half_impl::raw_uint16_as_half(0x7c00); }
  static Eigen::half quiet_NaN() { return Eigen::half_impl::raw_uint16_as_half(0x7e00); }
  static Eigen::half signaling_NaN() { return Eigen::half_impl::raw_uint16_as_half(0x7d00); }
  static Eigen::half denorm_min() { return Eigen::half_impl::raw_uint16_as_half(0x1); }
};

// If std::numeric_limits<T> is specialized, should also specialize
// std::numeric_limits<const T>, std::numeric_limits<volatile T>, and
// std::numeric_limits<const volatile T>
// https://stackoverflow.com/a/16519653/
template<>
struct numeric_limits<const Eigen::half> : numeric_limits<Eigen::half> {};
template<>
struct numeric_limits<volatile Eigen::half> : numeric_limits<Eigen::half> {};
template<>
struct numeric_limits<const volatile Eigen::half> : numeric_limits<Eigen::half> {};

#if __cplusplus > 199711L
template <>
struct hash<Eigen::half> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::size_t operator()(const Eigen::half& a) const {
    return static_cast<std::size_t>(Eigen::half_impl::raw_half_as_uint16(a));
  }
};
#endif

} // end namespace std


namespace Eigen {
namespace half_impl {
 
// We need operators to be defined for both host and device.

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half operator + (const Eigen::half& a, const Eigen::half& b) {
  #if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
  return __hadd(a, b);
  #else
  return Eigen::half_impl::float_to_half(Eigen::half_impl::half_to_float(a) + Eigen::half_impl::half_to_float(b));
  #endif
}

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half operator * (const Eigen::half& a, const Eigen::half& b) {
  #if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
  return __hmul(a, b);
  #else
  return Eigen::half_impl::float_to_half(Eigen::half_impl::half_to_float(a) * Eigen::half_impl::half_to_float(b));
  #endif
}

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half operator - (const Eigen::half& a, const Eigen::half& b) {
  #if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
  return __hsub(a, b);
  #else
  return Eigen::half_impl::float_to_half(Eigen::half_impl::half_to_float(a) - Eigen::half_impl::half_to_float(b));
  #endif
}

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half operator / (const Eigen::half& a, const Eigen::half& b) {
  #if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC) && (defined(EIGEN_HIPCC) || EIGEN_CUDA_SDK_VER >= 90000)
  return __hdiv(a, b);
  #else
  return Eigen::half_impl::float_to_half(Eigen::half_impl::half_to_float(a) / Eigen::half_impl::half_to_float(b));
  #endif
}

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half operator + (const Eigen::half& a) { return a; }

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half operator - (const Eigen::half& a) {
  #if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
  return __hneg(a);
  #else
  return Eigen::half_impl::raw_uint16_as_half(Eigen::half_impl::raw_half_as_uint16(a) ^ 0x8000);
  #endif
}

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half& operator += (Eigen::half& a, const Eigen::half& b) { a = a + b; return a; }
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half& operator *= (Eigen::half& a, const Eigen::half& b) { a = a * b; return a; }
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half& operator -= (Eigen::half& a, const Eigen::half& b) { a = a - b; return a; }
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half& operator /= (Eigen::half& a, const Eigen::half& b) { a = a / b; return a; }

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator == (const Eigen::half& a, const Eigen::half& b) {
  #if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
  return __heq(a, b);
  #else
  return Eigen::numext::equal_strict(Eigen::half_impl::half_to_float(a), Eigen::half_impl::half_to_float(b));
  #endif
}

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator != (const Eigen::half& a, const Eigen::half& b) {
  #if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
  return __hne(a, b);
  #else
  return Eigen::numext::not_equal_strict(Eigen::half_impl::half_to_float(a), Eigen::half_impl::half_to_float(b));
  #endif
}

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator < (const Eigen::half& a, const Eigen::half& b) {
  #if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
  return __hlt(a, b);
  #else
  return Eigen::half_impl::half_to_float(a) < Eigen::half_impl::half_to_float(b);
  #endif
}

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator <= (const Eigen::half& a, const Eigen::half& b) {
  #if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
  return __hle(a, b);
  #else
  return Eigen::half_impl::half_to_float(a) <= Eigen::half_impl::half_to_float(b);
  #endif
}

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator > (const Eigen::half& a, const Eigen::half& b) {
  #if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
  return __hgt(a, b);
  #else
  return Eigen::half_impl::half_to_float(a) > Eigen::half_impl::half_to_float(b);
  #endif
}

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator >= (const Eigen::half& a, const Eigen::half& b) {
  #if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
  return __hge(a, b);
  #else
  return Eigen::half_impl::half_to_float(a) >= Eigen::half_impl::half_to_float(b);
  #endif
}

// Division by an index. Do it in full float precision to avoid accuracy
// issues in converting the denominator to Eigen::half.
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half operator / (const Eigen::half& a, Eigen::Index b) {
  return Eigen::half_impl::float_to_half(Eigen::half_impl::half_to_float(a) / static_cast<float>(b));
}

} // namespace half_impl
} // namespace Eigen

// We currently don't have ops for log2/log10 that can be specialized, so creating overloads here.
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC __half log10(const __half& a) {
  return Eigen::half_impl::float_to_half(::log10f(Eigen::half_impl::half_to_float(a)));
}

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half log2(const half& a) {
  return Eigen::half_impl::float_to_half(static_cast<float>(EIGEN_LOG2E) * ::logf(Eigen::half_impl::half_to_float(a)));
}

#if !defined(EIGEN_NO_IO)
EIGEN_ALWAYS_INLINE std::ostream& operator << (std::ostream& os, const __half& v) {
  os << Eigen::half_impl::half_to_float(v);
  return os;
}
#endif // !defined(EIGEN_NO_IO)

}  // namespace Eigen

namespace Eigen {

namespace internal {

template<> struct is_arithmetic<half> { enum { value = true }; };

// Cast through float.
template<typename T>
struct cast_impl<half, T> {
  EIGEN_DEVICE_FUNC
  static inline T run(const half& x)
  {
    return internal::cast<float, T>(half_impl::half_to_float(x));
  }
};

// Cast through float.
template<typename T>
struct cast_impl<T, half> {
  EIGEN_DEVICE_FUNC
  static inline half run(const T& x)
  {
    return half_impl::float_to_half(internal::cast<T, float>(x));
  }
};

template<typename T>
struct cast_impl<std::complex<T>, half> {
  EIGEN_DEVICE_FUNC
  static inline half run(const std::complex<T>& c)
  {
    return internal::cast<T, half>(c.real());
  }
};

// Set specific bit pattern.
template<>
struct cast_impl<bool, half> {
  EIGEN_DEVICE_FUNC
  static inline half run(bool b)
  {
    return half_impl::raw_uint16_as_half(b ? 0x3c00 : 0);
  }
};

template<>
struct random_default_impl<Eigen::half, false, false>
{
  static inline Eigen::half run(const Eigen::half& x, const Eigen::half& y)
  {
    return x + (y - x) * half_impl::float_to_half(float(std::rand()) / float(RAND_MAX));
  }
  static inline half run()
  {
    return run(half_impl::float_to_half(-1.f), half_impl::float_to_half(1.f));
  }
};

} // namespace internal

template<> struct NumTraits<Eigen::half>
    : GenericNumTraits<Eigen::half>
{
  enum {
    IsSigned = true,
    IsInteger = false,
    IsComplex = false,
    RequireInitialization = false
  };

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Eigen::half epsilon() {
    return half_impl::raw_uint16_as_half(0x0800);
  }
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Eigen::half dummy_precision() {
    return half_impl::raw_uint16_as_half(0x211f); // Eigen::half(1e-2f);
  }
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Eigen::half highest() {
    return half_impl::raw_uint16_as_half(0x7bff);
  }
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Eigen::half lowest() {
    return half_impl::raw_uint16_as_half(0xfbff);
  }
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Eigen::half infinity() {
    return half_impl::raw_uint16_as_half(0x7c00);
  }
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Eigen::half quiet_NaN() {
    return half_impl::raw_uint16_as_half(0x7e00);
  }
};

// Intrinsics for native fp16 support. Note that on current hardware,
// these are no faster than fp32 arithmetic (you need to use the half2
// versions to get the ALU speed increased), but you do save the
// conversion steps back and forth.

namespace numext {

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool (isinf)(const half& a) {
#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
  return __hisinf(a);
#else
  return (half_impl::raw_half_as_uint16(a) & 0x7fff) == 0x7c00;
#endif
}

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool (isnan)(const half& a) {
#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
  return __hisnan(a);
#else
  return (half_impl::raw_half_as_uint16(a) & 0x7fff) > 0x7c00;
#endif
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool (isfinite)(const half& a) {
  return !(isinf EIGEN_NOT_A_MACRO (a)) && !(isnan EIGEN_NOT_A_MACRO (a));
}

template<>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half abs<half>(const half& a) {
  return half_impl::raw_uint16_as_half(half_impl::raw_half_as_uint16(a) & 0x7fff);
}

template<>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half exp<half>(const half& a) {
#if defined(EIGEN_GPU_HAS_FP16_MATH_FUNCTIONS)
  return ::hexp(a);
#else
   return half_impl::float_to_half(::expf(half_impl::half_to_float(a)));
#endif
}

template<>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half expm1<half>(const half& a) {
  return half_impl::float_to_half(numext::expm1(half_impl::half_to_float(a)));
}

template<>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half log<half>(const half& a) {
#if defined(EIGEN_GPU_HAS_FP16_MATH_FUNCTIONS)
  return ::hlog(a);
#else
  return half_impl::float_to_half(::logf(half_impl::half_to_float(a)));
#endif
}

template<>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half log1p<half>(const half& a) {
  return half_impl::float_to_half(numext::log1p(half_impl::half_to_float(a)));
}

template<>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half sqrt<half>(const half& a) {
#if defined(EIGEN_GPU_HAS_FP16_MATH_FUNCTIONS)
  return ::hsqrt(a);
#else
  return half_impl::float_to_half(::sqrtf(half_impl::half_to_float(a)));
#endif
}

template<>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half pow<half>(const half& a, const half& b) {
  return half_impl::float_to_half(::powf(half_impl::half_to_float(a), half_impl::half_to_float(b)));
}

template<>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half sin<half>(const half& a) {
#if defined(EIGEN_GPU_HAS_FP16_MATH_FUNCTIONS)
  return ::hsin(a);
#else
  return half_impl::float_to_half(::sinf(half_impl::half_to_float(a)));
#endif
}

template<>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half cos<half>(const half& a) {
#if defined(EIGEN_GPU_HAS_FP16_MATH_FUNCTIONS)
  return ::hcos(a);
#else
  return half_impl::float_to_half(::cosf(half_impl::half_to_float(a)));
#endif
}

template<>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half tan<half>(const half& a) {
  return half_impl::float_to_half(::tanf(half_impl::half_to_float(a)));
}

template<>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half tanh<half>(const half& a) {
  return half_impl::float_to_half(::tanhf(half_impl::half_to_float(a)));
}

template<>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half asin<half>(const half& a) {
  return half_impl::float_to_half(::asinf(half_impl::half_to_float(a)));
}

template<>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half acos<half>(const half& a) {
  return half_impl::float_to_half(::acosf(half_impl::half_to_float(a)));
}

template<>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half floor<half>(const half& a) {
#if defined(EIGEN_GPU_HAS_FP16_MATH_FUNCTIONS)
  return ::hfloor(a);
#else
  return half_impl::float_to_half(::floorf(half_impl::half_to_float(a)));
#endif
}

template<>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half rint<half>(const half& a) {
#if defined(EIGEN_GPU_HAS_FP16_MATH_FUNCTIONS)
  return ::hrint(a);
#else
  return half_impl::float_to_half(::rintf(half_impl::half_to_float(a)));
#endif
}

template<>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half ceil<half>(const half& a) {
#if defined(EIGEN_GPU_HAS_FP16_MATH_FUNCTIONS)
  return ::hceil(a);
#else
  return half_impl::float_to_half(::ceilf(half_impl::half_to_float(a)));
#endif
}

template<>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half mini<half>(const half& a, const half& b) {
#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
  return __hlt(b, a) ? b : a;
#else
  const float f1 = half_impl::half_to_float(a);
  const float f2 = half_impl::half_to_float(b);
  return f2 < f1 ? b : a;
#endif
}

template<>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half maxi<half>(const half& a, const half& b) {
#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
  return __hlt(a, b) ? b : a;
#else
  const float f1 = half_impl::half_to_float(a);
  const float f2 = half_impl::half_to_float(b);
  return f1 < f2 ? b : a;
#endif
}

// Bit-cast operations. These should be use instead of half_impl::raw_* externally.
template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half bit_cast<Eigen::half, uint16_t>(const uint16_t& src) {
  return half_impl::raw_uint16_as_half(src);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC uint16_t bit_cast<uint16_t, Eigen::half>(const Eigen::half& src) {
  return half_impl::raw_half_as_uint16(src);
}

}  // namespace numext
}  // namespace Eigen

#undef EIGEN_GPU_HAS_FP16_ARITHMETIC
#undef EIGEN_GPU_HAS_FP16_MATH_FUNCTIONS

#endif // defined(EIGEN_HAS_GPU_FP16)

#endif // EIGEN_GPU_HALF_H
