#ifndef TEST_SIMPLE_TUPLE_H
#define TEST_SIMPLE_TUPLE_H

#include <tuple>
#include <type_traits>

namespace Eigen {
  
namespace internal {

// Internal Tuple implementation.
template<size_t N, typename... Types>
class TupleImpl;

// Generic recursive tuple.
template<size_t N, typename T1, typename... Ts>
class TupleImpl<N, T1, Ts...> {
 public:
 
  // Default constructor, enable if all types are default-constructible.
  template<typename EnableIf = typename std::enable_if<
    reduce_all<
      std::is_default_constructible<T1>::value,
      std::is_default_constructible<Ts>::value...>::value>::type >
  EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC
  TupleImpl() : head_{}, tail_{} {}
 
  // Element constructor.
  template<typename U1, typename... Us, 
           // Only enable if...
           typename EnableIf = typename std::enable_if<
              // the number of input arguments match, and ...
              sizeof...(Us) == sizeof...(Ts) && (
                // this does not look like a copy/move constructor.
                N > 1 || std::is_convertible<U1, T1>::value)
           >::type>
  EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC
  TupleImpl(U1&& arg1, Us&&... args) 
    : head_(std::forward<U1>(arg1)), tail_(std::forward<Us>(args)...) {}
 
  // The first stored value. 
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  T1& head() {
    return head_;
  }
  
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  const T1& head() const {
    return head_;
  }
  
  // The tail values.
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  TupleImpl<N-1, Ts...>& tail() {
    return tail_;
  }
  
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  const TupleImpl<N-1, Ts...>& tail() const {
    return tail_;
  }
  
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  void swap(TupleImpl& other) {
    using numext::swap;
    swap(head_, other.head_);
    swap(tail_, other.tail_);
  }
  
 private:
  T1 head_;
  TupleImpl<N-1, Ts...> tail_;
};

// Empty tuple specialization.
template<>
class TupleImpl<size_t(0)> {};

template<typename TupleType>
struct is_tuple : std::false_type {};

template<typename... Types>
struct is_tuple< TupleImpl<sizeof...(Types), Types...> > : std::true_type {};

// Gets an element from a Tuple.
template<size_t Idx, typename T1, typename... Ts>
struct tuple_get_impl {
  using TupleType = TupleImpl<sizeof...(Ts) + 1, T1, Ts...>;
  using ReturnType = typename tuple_get_impl<Idx - 1, Ts...>::ReturnType;
  
  static EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  ReturnType& run(TupleType& tuple) {
    return tuple_get_impl<Idx-1, Ts...>::run(tuple.tail());
  }

  static EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  const ReturnType& run(const TupleType& tuple) {
    return tuple_get_impl<Idx-1, Ts...>::run(tuple.tail());
  }
};

// Base case, getting the head element.
template<typename T1, typename... Ts>
struct tuple_get_impl<0, T1, Ts...> {
  using TupleType = TupleImpl<sizeof...(Ts) + 1, T1, Ts...>;
  using ReturnType = T1;

  static EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  T1& run(TupleType& tuple) {
    return tuple.head();
  }

  static EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  const T1& run(const TupleType& tuple) {
    return tuple.head();
  }
};

// Concatenates N Tuples.
template<size_t NTuples, typename... Tuples>
struct tuple_cat_impl;

template<size_t NTuples, size_t N1, typename... Args1, size_t N2, typename... Args2, typename... Tuples>
struct tuple_cat_impl<NTuples, TupleImpl<N1, Args1...>, TupleImpl<N2, Args2...>, Tuples...> {
  using TupleType1 = TupleImpl<N1, Args1...>;
  using TupleType2 = TupleImpl<N2, Args2...>;
  using MergedTupleType = TupleImpl<N1 + N2, Args1..., Args2...>;
  
  using ReturnType = typename tuple_cat_impl<NTuples-1, MergedTupleType, Tuples...>::ReturnType;
  
  // Uses the index sequences to extract and merge elements from tuple1 and tuple2,
  // then recursively calls again.
  template<typename Tuple1, size_t... I1s, typename Tuple2, size_t... I2s, typename... MoreTuples>
  static EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  ReturnType run(Tuple1&& tuple1, index_sequence<I1s...>,
                 Tuple2&& tuple2, index_sequence<I2s...>,
                 MoreTuples&&... tuples) {
    return tuple_cat_impl<NTuples-1, MergedTupleType, Tuples...>::run(
        MergedTupleType(tuple_get_impl<I1s, Args1...>::run(std::forward<Tuple1>(tuple1))...,
                        tuple_get_impl<I2s, Args2...>::run(std::forward<Tuple2>(tuple2))...),
        std::forward<MoreTuples>(tuples)...);
  }
  
  // Concatenates the first two tuples.
  template<typename Tuple1, typename Tuple2, typename... MoreTuples>
  static EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  ReturnType run(Tuple1&& tuple1, Tuple2&& tuple2, MoreTuples&&... tuples) {
    return run(std::forward<Tuple1>(tuple1), make_index_sequence<N1>{},
               std::forward<Tuple2>(tuple2), make_index_sequence<N2>{},
               std::forward<MoreTuples>(tuples)...);
  }
};

// Base case with a single tuple.
template<size_t N, typename... Args>
struct tuple_cat_impl<1, TupleImpl<N, Args...> > { 
  using ReturnType = TupleImpl<N, Args...>;
  
  template<typename Tuple1>
  static EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  ReturnType run(Tuple1&& tuple1) {
    return tuple1;
  }
};

// Special case of no tuples.
template<>
struct tuple_cat_impl<0> { 
  using ReturnType = TupleImpl<0>;
  static EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  ReturnType run() {return ReturnType{}; }
};

}  // namespace internal

/**
 * Basic Tuple class, similar to std::tuple, that can be used on device.
 */
template<typename... Types>
using Tuple = internal::TupleImpl<sizeof...(Types), Types...>;

/**
 * Utility for determining a Tuple's size.
 */
template<typename Tuple>
struct tuple_size;

template<typename... Types >
struct tuple_size< Tuple<Types...> > : std::integral_constant<size_t, sizeof...(Types)> {};

template<typename... Types >
struct tuple_size< std::tuple<Types...> > : std::tuple_size<std::tuple<Types...> > {};

/**
 * Gets an element of a tuple.
 * \tparam Idx index of the element.
 * \tparam Types ... tuple element types.
 * \param tuple the tuple.
 * \return a reference to the desired element.
 */
template<size_t Idx, typename... Types>
EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
const typename internal::tuple_get_impl<Idx, Types...>::ReturnType&
tuple_get(const Tuple<Types...>& tuple) {
  return internal::tuple_get_impl<Idx, Types...>::run(tuple);
}

template<size_t Idx, typename... Types>
EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
typename internal::tuple_get_impl<Idx, Types...>::ReturnType&
tuple_get(Tuple<Types...>& tuple) {
  return internal::tuple_get_impl<Idx, Types...>::run(tuple);
}

template<size_t Idx, typename... Types>
auto tuple_get(const std::tuple<Types...>& tuple) -> decltype(std::get<Idx>(tuple)) {
  return std::get<Idx>(tuple);
}

template<size_t Idx, typename... Types>
auto tuple_get(std::tuple<Types...>& tuple) -> decltype(std::get<Idx>(tuple)) {
  return std::get<Idx>(tuple);
}

/**
 * Concatenate multiple tuples.
 * \param tuples ... list of tuples.
 * \return concatenated tuple.
 */
template<typename... Tuples,
          typename EnableIf = typename std::enable_if<
            internal::reduce_all<
              internal::is_tuple<typename std::decay<Tuples>::type>::value...>::value>::type
        >
EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
typename internal::tuple_cat_impl<sizeof...(Tuples), typename std::decay<Tuples>::type...>::ReturnType
tuple_cat(Tuples&&... tuples) {
  return internal::tuple_cat_impl<sizeof...(Tuples), typename std::decay<Tuples>::type...>::run(std::forward<Tuples>(tuples)...);
}

// std::tuple specialization.
template<typename... Tuples,
          typename EnableIf = typename std::enable_if<
            !internal::reduce_all<
              internal::is_tuple<typename std::decay<Tuples>::type>::value...>::value>::type
        >
EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
auto tuple_cat(Tuples&&... tuples) -> decltype(std::tuple_cat(std::forward<Tuples>(tuples)...)){
  return std::tuple_cat(std::forward<Tuples>(tuples)...);
}

}  // namespace Eigen

#endif
