// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2020 C. Antonio Sanchez <cantonios@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_COMPLEX_GPU_H
#define EIGEN_COMPLEX_GPU_H

namespace Eigen {

namespace internal {

#if defined(EIGEN_GPUCC) && defined(EIGEN_USE_GPU)

//===========================
//          float
//===========================
typedef eigen_packet_wrapper<float4, 1> Packet2cf;

template <>
struct packet_traits<std::complex<float> > : default_packet_traits {
  typedef Packet2cf type;
  typedef Packet2cf half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 2,
    HasHalfPacket = 0,

    HasAdd = 1,
    HasSub = 1,
    HasMul = 1,
    HasDiv = 1,
    HasNegate = 1,
    HasSqrt = 1,
    HasAbs = 0,
    HasAbs2 = 0,
    HasMin = 0,
    HasMax = 0,
    HasSetLinear = 0,
  };
};

template <>
struct unpacket_traits<Packet2cf> {
  typedef std::complex<float> type;
  typedef Packet2cf half;
  typedef float4 as_real;
  enum {
    size = 2,
    alignment = Aligned16,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet2cf
padd<Packet2cf>(const Packet2cf& a, const Packet2cf& b) {
  return padd<float4>(a, b);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet2cf
psub<Packet2cf>(const Packet2cf& a, const Packet2cf& b) {
  return psub<float4>(a, b);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet2cf pnegate(const Packet2cf& a) {
  return pnegate<float4>(a);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet2cf pconj(const Packet2cf& a) {
  return make_float4(a.m_val.x, -a.m_val.y, a.m_val.z, -a.m_val.w);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet2cf
pmul<Packet2cf>(const Packet2cf& a, const Packet2cf& b) {
  return make_float4(a.m_val.x * b.m_val.x - a.m_val.y * b.m_val.y, a.m_val.x * b.m_val.y + a.m_val.y * b.m_val.x,
                     a.m_val.z * b.m_val.z - a.m_val.w * b.m_val.w, a.m_val.z * b.m_val.w + a.m_val.w * b.m_val.z);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet2cf
pdiv<Packet2cf>(const Packet2cf& a, const Packet2cf& b) {
  float d1 = b.m_val.x * b.m_val.x + b.m_val.y * b.m_val.y;
  float d2 = b.m_val.z * b.m_val.z + b.m_val.w * b.m_val.w;
  return make_float4((a.m_val.x * b.m_val.x + a.m_val.y * b.m_val.y) / d1, (a.m_val.y * b.m_val.x - a.m_val.x * b.m_val.y) / d1,
                     (a.m_val.z * b.m_val.z + a.m_val.w * b.m_val.w) / d2,
                     (a.m_val.w * b.m_val.z - a.m_val.z * b.m_val.w) / d2);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet2cf
ptrue<Packet2cf>(const Packet2cf& a) {
  return ptrue<float4>(a);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet2cf
pand<Packet2cf>(const Packet2cf& a, const Packet2cf& b) {
  return pand<float4>(a, b);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet2cf
por<Packet2cf>(const Packet2cf& a, const Packet2cf& b) {
  return por<float4>(a, b);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet2cf
pxor<Packet2cf>(const Packet2cf& a, const Packet2cf& b) {
  return pxor<float4>(a, b);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet2cf
pandnot<Packet2cf>(const Packet2cf& a, const Packet2cf& b) {
  return pandnot<float4>(a, b);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet2cf
pload<Packet2cf>(const std::complex<float>* from) {
  return pload<float4>(&numext::real_ref(*from));
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet2cf
ploadu<Packet2cf>(const std::complex<float>* from) {
  return ploadu<float4>(&numext::real_ref(*from));
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet2cf
pset1<Packet2cf>(const std::complex<float>& from) {
  return make_float4(from.real(), from.imag(), from.real(), from.imag());
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet2cf
ploaddup<Packet2cf>(const std::complex<float>* from) {
  return pset1<Packet2cf>(*from);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC void pstore<std::complex<float> >(
    std::complex<float>* to, const Packet2cf& from) {
  pstore<float4>(&numext::real_ref(*to), from);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC void pstoreu<std::complex<float> >(
    std::complex<float>* to, const Packet2cf& from) {
  pstoreu<float4>(&numext::real_ref(*to), from);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_DEVICE_FUNC inline Packet2cf
pgather<std::complex<float>, Packet2cf>(const std::complex<float>* from,
                                        Index stride) {
  return make_float4(from[0 * stride].real(), from[0 * stride].imag(),
                     from[1 * stride].real(), from[1 * stride].imag());
}

template <>
EIGEN_DEVICE_FUNC EIGEN_DEVICE_FUNC inline void
pscatter<std::complex<float>, Packet2cf>(std::complex<float>* to,
                                         const Packet2cf& from, Index stride) {
  to[stride * 0] = std::complex<float>(from.m_val.x, from.m_val.y);
  to[stride * 1] = std::complex<float>(from.m_val.z, from.m_val.w);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC std::complex<float> pfirst<Packet2cf>(
    const Packet2cf& a) {
  return std::complex<float>(a.m_val.x, a.m_val.y);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet2cf preverse(const Packet2cf& a) {
  return make_float4(a.m_val.z, a.m_val.w, a.m_val.x, a.m_val.y);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC std::complex<float> predux<Packet2cf>(
    const Packet2cf& a) {
  return std::complex<float>(a.m_val.x + a.m_val.z, a.m_val.y + a.m_val.w);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC std::complex<float> predux_mul<Packet2cf>(
    const Packet2cf& a) {
  return std::complex<float>(a.m_val.x * a.m_val.z - a.m_val.y * a.m_val.w, a.m_val.x * a.m_val.w + a.m_val.y * a.m_val.z);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet2cf
pcplxflip<Packet2cf>(const Packet2cf& a) {
  return make_float4(a.m_val.y, a.m_val.x, a.m_val.w, a.m_val.z);
}

EIGEN_DEVICE_FUNC EIGEN_DEVICE_FUNC inline void ptranspose(
    PacketBlock<Packet2cf, 2>& kernel) {
  EIGEN_USING_STD(swap);
  swap(kernel.packet[0].m_val.z, kernel.packet[1].m_val.x);
  swap(kernel.packet[0].m_val.w, kernel.packet[1].m_val.y);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet2cf pcmp_eq(const Packet2cf& a,
                                                        const Packet2cf& b) {
  return pcmp_eq<float4>(a, b);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet2cf
psqrt<Packet2cf>(const Packet2cf& a) {
  return psqrt_complex<Packet2cf>(a);
}

//===========================
//         double
//===========================
typedef eigen_packet_wrapper<double2, 1> Packet1cd;

template <>
struct packet_traits<std::complex<double> > : default_packet_traits {
  typedef Packet1cd type;
  typedef Packet1cd half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 0,
    size = 1,
    HasHalfPacket = 0,

    HasAdd = 1,
    HasSub = 1,
    HasMul = 1,
    HasDiv = 1,
    HasNegate = 1,
    HasSqrt = 1,
    HasAbs = 0,
    HasAbs2 = 0,
    HasMin = 0,
    HasMax = 0,
    HasSetLinear = 0
  };
};

template <>
struct unpacket_traits<Packet1cd> {
  typedef std::complex<double> type;
  typedef Packet1cd half;
  typedef double2 as_real;
  enum {
    size = 1,
    alignment = Aligned16,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet1cd
padd<Packet1cd>(const Packet1cd& a, const Packet1cd& b) {
  return padd<double2>(a, b);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet1cd
psub<Packet1cd>(const Packet1cd& a, const Packet1cd& b) {
  return psub<double2>(a, b);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet1cd pnegate(const Packet1cd& a) {
  return pnegate<double2>(a);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet1cd pconj(const Packet1cd& a) {
  return make_double2(a.m_val.x, -a.m_val.y);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet1cd
pmul<Packet1cd>(const Packet1cd& a, const Packet1cd& b) {
  return make_double2(a.m_val.x * b.m_val.x - a.m_val.y * b.m_val.y, a.m_val.x * b.m_val.y + a.m_val.y * b.m_val.x);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet1cd
pdiv<Packet1cd>(const Packet1cd& a, const Packet1cd& b) {
  double d1 = b.m_val.x * b.m_val.x + b.m_val.y * b.m_val.y;
  return make_double2((a.m_val.x * b.m_val.x + a.m_val.y * b.m_val.y) / d1,
                      (a.m_val.y * b.m_val.x - a.m_val.x * b.m_val.y) / d1);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet1cd
ptrue<Packet1cd>(const Packet1cd& a) {
  return ptrue<double2>(a);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet1cd
pand<Packet1cd>(const Packet1cd& a, const Packet1cd& b) {
  return pand<double2>(a, b);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet1cd
por<Packet1cd>(const Packet1cd& a, const Packet1cd& b) {
  return por<double2>(a, b);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet1cd
pxor<Packet1cd>(const Packet1cd& a, const Packet1cd& b) {
  return pxor<double2>(a, b);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet1cd
pandnot<Packet1cd>(const Packet1cd& a, const Packet1cd& b) {
  return pandnot<double2>(a, b);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet1cd
pload<Packet1cd>(const std::complex<double>* from) {
  return pload<double2>(&numext::real_ref(*from));
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet1cd
ploadu<Packet1cd>(const std::complex<double>* from) {
  return ploadu<double2>(&numext::real_ref(*from));
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet1cd
pset1<Packet1cd>(const std::complex<double>& from) {
  return make_double2(from.real(), from.imag());
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet1cd
ploaddup<Packet1cd>(const std::complex<double>* from) {
  return pset1<Packet1cd>(*from);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC void pstore<std::complex<double> >(
    std::complex<double>* to, const Packet1cd& from) {
  pstore<double2>(&numext::real_ref(*to), from);
}
template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC void pstoreu<std::complex<double> >(
    std::complex<double>* to, const Packet1cd& from) {
  pstoreu<double2>(&numext::real_ref(*to), from);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_DEVICE_FUNC inline Packet1cd
pgather<std::complex<double>, Packet1cd>(const std::complex<double>* from,
                                         Index stride) {
  return pset1<Packet1cd>(*from);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_DEVICE_FUNC inline void
pscatter<std::complex<double>, Packet1cd>(std::complex<double>* to,
                                          const Packet1cd& from, Index stride) {
  to[stride] = std::complex<double>(from.m_val.x, from.m_val.y),
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC std::complex<double> pfirst<Packet1cd>(
    const Packet1cd& a) {
  return std::complex<double>(a.m_val.x, a.m_val.y);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet1cd
pcplxflip<Packet1cd>(const Packet1cd& a) {
  return make_double2(a.m_val.y, a.m_val.x);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet1cd pcmp_eq(const Packet1cd& a,
                                                        const Packet1cd& b) {
  return pcmp_eq<double2>(a, b);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet1cd
psqrt<Packet1cd>(const Packet1cd& a) {
  return psqrt_complex<Packet1cd>(a);
}

#endif // defined(EIGEN_GPUCC) && defined(EIGEN_USE_GPU)

}  // end namespace internal
}  // end namespace Eigen

#endif  // EIGEN_COMPLEX_SSE_H
