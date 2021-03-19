#include <iostream>
#include <cstdio>
#include <complex>
#include <Eigen/Core>

namespace Eigen {

// Complex number implementation that can be used on device.
template<typename T>
class complex {
 public:

  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR
  complex(const T& re = T(), const T& im = T()) : re_(re), im_(im) {}

  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR
  complex(const complex<T>& v) : re_(v.real_ref()), im_(v.imag_ref()) {}

  template<typename X>
  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR
  complex(const complex<X>& v) : 
    re_(static_cast<T>(v.real_ref())), 
    im_(static_cast<T>(v.imag_ref())) {}

  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR
  complex(const std::complex<T>& v) : 
    re_(reinterpret_cast<const T(&)[2]>(v)[0]), 
    im_(reinterpret_cast<const T(&)[2]>(v)[1]) {}

  template<typename X>
  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR
  complex(const std::complex<X>& v) : 
    re_(static_cast<T>(reinterpret_cast<const X(&)[2]>(v)[0])), 
    im_(static_cast<T>(reinterpret_cast<const X(&)[2]>(v)[1])) {}

  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  complex& operator=(const T& re) {
    real(re);
    imag(T(0));
    return *this;
  }

  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  complex& operator=(const complex<T>& v) {
    real(v.real_ref());
    imag(v.imag_ref());
    return *this;
  }

  template<typename X>
  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  complex& operator=(const complex<X>& v) {
    real(static_cast<T>(v.real_ref()));
    imag(static_cast<T>(v.imag_ref()));
    return *this;
  }
  
  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  complex& operator=(const std::complex<T>& v) {
    real(reinterpret_cast<const T(&)[2]>(v)[0]);
    imag(reinterpret_cast<const T(&)[2]>(v)[1]);
    return *this;
  }

  template<typename X>
  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  complex& operator=(const std::complex<X>& v) {
    real(static_cast<T>(reinterpret_cast<const X(&)[2]>(v)[0]));
    imag(static_cast<T>(reinterpret_cast<const X(&)[2]>(v)[1]));
    return *this;
  }

  // Reference directly as a std::complex<T>.
  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  std::complex<T>& std() {
    return *reinterpret_cast<std::complex<T>*>(&re_);
  }

  // Reference directly as a std::complex<T>.
  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  const std::complex<T>& std() const {
    return *reinterpret_cast<const std::complex<T>*>(&re_);
  }

  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  T real() const {
    return real_ref();
  }
  
  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  void real(T value) {
    real_ref() = value;
  }
  
  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  T imag() const {
    return imag_ref();
  }
  
  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  void imag(T value) {
    imag_ref() = value;
  }

  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  T& real_ref() {
    return re_;
  }

  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  const T& real_ref() const {
    return re_;
  }
  
  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  T& imag_ref() {
    return im_;
  }

  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  const T& imag_ref() const {
    return im_;
  }

  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  complex& operator+=(const complex& other) {
    real_ref() += other.real_ref();
    imag_ref() += other.imag_ref();
    return *this;
  }

  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  complex& operator+=(const T& other) {
    real_ref() += other;
    return *this;
  }

  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  complex& operator-=(const complex& other) {
    real_ref() -= other.real_ref();
    imag_ref() -= other.imag_ref();
    return *this;
  }

  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  complex& operator-=(const T& other) {
    real_ref() -= other;
    return *this;
  }

  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  complex& operator*=(const complex& other) {
    *this = *this * other;
    return *this;
  }

  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  complex& operator*=(const T& other) {
    real_ref() *= other;
    imag_ref() *= other;
    return *this;
  }

  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  complex& operator/=(const complex& other) {
    *this = *this / other;
    return *this;
  }

  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
  complex& operator/=(const T& other) {
    real_ref() /= other;
    imag_ref() /= other;
    return *this;
  }

 private:
  T re_;
  T im_;
};

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> operator+(const complex<T>& a) {
    return a;
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> operator-(const complex<T>& b) {
    return complex<T>(-b.real_ref(), -b.imag_ref());
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> operator+(const complex<T>& a, const complex<T>& b) {
    return complex<T>(a.real_ref() + b.real_ref(), a.imag_ref() + b.imag_ref());
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> operator-(const complex<T>& a, const complex<T>& b) {
    return complex<T>(a.real_ref() - b.real_ref(), a.imag_ref() - b.imag_ref());
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> operator*(const complex<T>& a, const complex<T>& b) {
    return complex<T>(a.real_ref() * b.real_ref() - a.imag_ref() * b.imag_ref(),
                      a.imag_ref() * b.real_ref() + a.real_ref() * b.imag_ref());
}

template<typename T>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC
complex<T> operator/(const complex<T>& a, const complex<T>& b) {
#if EIGEN_FAST_MATH
    // Fast divide, sensitive to small norms.
  const T norm = T(1) / (b.real_ref() * b.real_ref() + b.imag_ref() * b.imag_ref());
  return complex<T>((a.real_ref() * b.real_ref() + a.imag_ref() * b.imag_ref()) * norm,
                    (a.imag_ref() * b.real_ref() - a.real_ref() * b.imag_ref()) * norm);
#else
  // Stable divide, guard against over/under-flow.
  const T scale = T(1) / (numext::abs(b.real_ref()) + numext::abs(b.imag_ref()));
  const T a_real_scaled = a.real_ref() * scale;
  const T a_imag_scaled = a.imag_ref() * scale;
  const T b_real_scaled = b.real_ref() * scale;
  const T b_imag_scaled = b.imag_ref() * scale;
  const T b_norm2_scaled = b_real_scaled * b_real_scaled + b_imag_scaled * b_imag_scaled;
  return std::complex<T>(
      (a_real_scaled * b_real_scaled + a_imag_scaled * b_imag_scaled) / b_norm2_scaled,
      (a_imag_scaled * b_real_scaled - a_real_scaled * b_imag_scaled) / b_norm2_scaled);
#endif
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> operator+(const complex<T>& a, const T& b) {
    return complex<T>(a.real_ref() + b, a.imag_ref());
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> operator-(const complex<T>& a, const T& b) {
    return complex<T>(a.real_ref() - b, a.imag_ref());
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> operator*(const complex<T>& a, const T& b) {
    return complex<T>(a.real_ref() * b, a.imag_ref() * b);
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> operator/(const complex<T>& a, const T& b) {
  return complex<T>(a.real_ref() / b,  a.imag_ref() / b);
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> operator+(const T& a, const complex<T>& b) {
    return complex<T>(a + b.real_ref(), b.imag_ref());
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> operator-(const T& a, const complex<T>& b) {
    return complex<T>(a - b.real_ref(), -b.imag_ref());
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> operator*(const T& a, const complex<T>& b) {
    return complex<T>(a * b.real_ref(), a * b.imag_ref());
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> operator/(const T& a, const complex<T>& b) {
  const T b_norm2 = b.real_ref() * b.real_ref() + b.imag_ref() * b.imag_ref();
  return complex<T>( (a * b.real_ref()) / b_norm2, -(a* b.imag_ref()) / b_norm2);
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
bool operator==(const complex<T>& a, const complex<T>& b) {
    return a.real_ref() == b.real_ref() && a.imag_ref() == b.imag_ref();
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
bool operator==(const complex<T>& a, const T& b) {
    return a.real_ref() == b && a.imag_ref() == T(0);
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
bool operator==(const T& a, const complex<T>& b) {
    return a == b.real_ref() && T(0) == b.imag_ref();
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
bool operator!=(const complex<T>& a, const complex<T>& b) {
    return !(a == b);
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
bool operator!=(const complex<T>& a, const T& b) {
    return !(a == b);
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
bool operator!=(const T& a, const complex<T>& b) {
    return !(a == b);
}

#if !defined(EIGEN_NO_IO) && !defined(EIGEN_GPU_COMPILE_PHASE)

template <class T, class CharT, class Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os, const complex<T>& x) {
  return os << x.std();
}

template <class T, class CharT, class Traits>
std::basic_istream<CharT, Traits>& operator>>(std::basic_istream<CharT, Traits>& is,  std::complex<T>& x) {
  std::complex<T> tmp;
  is >> tmp;
  x = tmp;
  return is;
}

#endif // !EIGEN_NO_IO && !EIGEN_GPU_COMPILE_PHASE

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
T real(const complex<T>& z) {
    return z.real();
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
T imag(const complex<T>& z) {
    return z.imag();
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
T abs(const complex<T>& z) {
    return numext::hypot(z.real(), z.imag());
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
T arg(const complex<T>& z) {
    EIGEN_USING_STD(atan2)
    return atan2(z.imag(), z.real());
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
T norm(const complex<T>& z) {
    return z.real_ref()*z.real_ref() + z.imag_ref()*z.imag_ref();
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> conj(const complex<T>& z) {
    return complex<T>(z.real_ref(), -z.imag_ref());
}

template<typename T>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC
complex<T> proj(const complex<T>& z) {
    if ((numext::isinf)(z.real_ref()) || (numext::isinf)(z.imag_ref())) {
        if (z.imag_ref() < T(0)) {
            return complex<T>(NumTraits<T>::infinity(), -T(0));
        }
        return complex<T>(NumTraits<T>::infinity(), T(0));
    }
    return z;
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> polar(const T& r, const T& theta = T()) {
  return complex<T>(r * numext::cos(theta), r * numext::sin(theta));
}

template<typename T>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC
complex<T> exp(const complex<T>& z) {
    // TODO: edge cases
    T r = numext::exp(z.real_ref());
    return complex<T>(r * numext::cos(z.imag_ref()), r * numext::sin(z.imag_ref()));
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> log(const complex<T>& z) {
    return complex<T>(numext::log(abs(z)), arg(z));
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> log10(const complex<T>& z) {
    return log(z)/numext::log(T(10));
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> pow(const complex<T>& x, const complex<T>& y) {
    return exp(y * log(x));
}

template<typename T>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC
complex<T> sqrt(const complex<T>& z) {
  const T x = z.real();
  const T y = z.imag();
  const T zero = T(0);
  const T w = numext::sqrt(T(0.5) * (numext::abs(x) + numext::hypot(x, y)));
  return
    (numext::isinf)(y) ? complex<T>(NumTraits<T>::infinity(), y)
      : x == zero ? complex<T>(w, y < zero ? -w : w)
      : x > zero ? complex<T>(w, y / (2 * w))
      : complex<T>(numext::abs(y) / (2 * w), y < zero ? -w : w );
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> square(const complex<T>& z) {
  return complex<T>((z.real_ref() - z.imag_ref())*(z.real_ref() + z.imag_ref()),
                    T(2)*z.real_ref()*z.imag_ref());
}

template<typename T>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC
complex<T> asinh(const complex<T>& z) {
  // TODO edge cases
  return log(z + sqrt(square(z) + T(1)));
}

template<typename T>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC
complex<T> acosh(const complex<T>& z) {
  // TODO edge cases
  return log(z + sqrt(square(z) - T(1)));
}

template<typename T>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC
complex<T> atanh(const complex<T>& z) {
  // TODO edge cases
  return log((z + T(1)/(z + T(1)))) / T(2);
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> sinh(const complex<T>& z) {
  // TODO edge cases
  return complex<T>(numext::sinh(z.real_ref()) * numext::cos(z.imag_ref()), 
                    numext::cosh(z.real_ref()) * numext::sin(z.imag_ref()));
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> cosh(const complex<T>& z) {
  // TODO edge cases
  return complex<T>(numext::cosh(z.real_ref()) * numext::cos(z.imag_ref()), 
                    numext::sinh(z.real_ref()) * numext::sin(z.imag_ref()));
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> tanh(const complex<T>& z) {
  // TODO
  return z;
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> asin(const complex<T>& z) {
  const complex<T> x = asinh(complex<T>(-z.imag_ref(), z.real_ref()));
  return complex<T>(x.imag_ref(), -x.real_ref());
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> acos(const complex<T>& z) {
  // TODO
  return z;
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> atan(const complex<T>& z) {
  const complex<T> x = atanh(complex<T>(-z.imag_ref(), z.real_ref()));
  return complex<T>(x.imag_ref(), -x.real_ref());
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> sin(const complex<T>& z) {
  const complex<T> x = sinh(complex<T>(-z.imag_ref(), z.real_ref()));
  return complex<T>(x.imag_ref(), -x.real_ref());
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> cos(const complex<T>& z) {
  return cosh(complex<T>(-z.imag_ref(), z.real_ref()));
}

template<typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
complex<T> tan(const complex<T>& z) {
  const complex<T> x = tanh(complex<T>(-z.imag_ref(), z.real_ref()));
  return complex<T>(x.imag_ref(), -x.real_ref());
}

}

__global__ void on_device() {

    double r = 5;

    Eigen::complex<double> a(1);
    Eigen::complex<double> b(3, 4);
    Eigen::complex<double> c(a);
    Eigen::complex<double> d(a.std());
    Eigen::complex<double> e(Eigen::complex<float>(1.f, 2.f));
    Eigen::complex<double> f(Eigen::complex<float>(1.f, 2.f).std());

    c = a;
    c = r;
    c = a.std();
    c = Eigen::complex<float>(1.f, 2.f);
    c = Eigen::complex<float>(1.f, 2.f).std();

    r = c.real();
    c.real(r);
    r = c.imag();
    c.imag(r);

    c += a;
    c += r;
    c -= a;
    c -= r;
    c *= a;
    c *= r;
    c /= a;
    c /= r;

    c = +a;
    c = -a;
    c = a + b;
    c = a - b;
    c = a * b;
    c = a / b;
    c = a + r;
    c = a - r;
    c = a * r;
    c = a / r;
    c = r + b;
    c = r - b;
    c = r * b;
    c = r / b;


    bool g = false;
    g |= (a == b);
    g |= (a == r);
    g |= (r == b);
    g |= (a != b);
    g |= (a != r);
    g |= (r != b);

    r = real(a);
    r = imag(a);
    r = abs(a);
    r = arg(a);
    r = norm(a);
    c = conj(a);
    c = proj(a);
    c = Eigen::polar<double>(r, r);

    c = exp(a);
    c = log(a);
    c = log10(a);
    c = pow(a, b);
    c = sqrt(a);

    c = sin(a);
    c = cos(a);
    c = tan(a);
    c = asin(a);
    c = acos(a);
    c = atan(a);
    c = sinh(a);
    c = cosh(a);
    c = tanh(a);
    c = asinh(a);
    c = acosh(a);
    c = atanh(a);
    
    printf("%f %f\n", c.real(), c.imag());
}

int main() {

    Eigen::complex<double> a(1, 2);
    Eigen::complex<double> b(3, 4);
    Eigen::complex<double> c = a + b;
    std::cout << c.std() << std::endl;
   
    on_device<<<32,1>>>();
    cudaDeviceSynchronize();

    return 0;
}
