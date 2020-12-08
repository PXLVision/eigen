// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
// Copyright (C) 2020 Matthias Peschke <mpeschke@physnet.uni-hamburg.de>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_IO_H
#define EIGEN_CXX11_TENSOR_TENSOR_IO_H

namespace Eigen {

template<std::size_t rank> struct TensorIOFormat;

namespace internal {
template<typename Tensor, std::size_t rank> struct TensorPrinter;
}

template<std::size_t rank>
struct TensorIOFormat
{
        TensorIOFormat(const std::array<std::string,rank>& _separator, const std::array<std::string,rank>& _prefix, const std::array<std::string,rank>& _suffix,
                       int _precision = StreamPrecision, int _flags = 0, const std::string& _tenPrefix="", const std::string& _tenSuffix="", const char _fill=' ')
                : tenPrefix(_tenPrefix), tenSuffix(_tenSuffix), prefix(_prefix), suffix(_suffix), separator(_separator),
                  fill(_fill), precision(_precision), flags(_flags) {
                init_spacer();
        }

        TensorIOFormat(int _precision = StreamPrecision, int _flags = 0, const std::string& _tenPrefix="", const std::string& _tenSuffix="", const char _fill=' ')
                : tenPrefix(_tenPrefix), tenSuffix(_tenSuffix), fill(_fill), precision(_precision), flags(_flags)
        {
                if (rank == 0) {return;}

                // default values of prefix, suffix and separator
                prefix[rank-1] = ""; for (std::size_t k=0; k<rank-1; k++) {prefix[k] = "[";}
                suffix[rank-1] = ""; for (std::size_t k=0; k<rank-1; k++) {suffix[k] = "]";}
                separator[rank-1] = ", "; if (rank >=2) {separator[rank-2] = "\n"; for (std::size_t k=0; k<rank-2; k++) {separator[k] = "";}}

                init_spacer();
        }

        void init_spacer() {
                if((flags & DontAlignCols))
                        return;
                spacer[rank-1] = "";
                int i = int(tenPrefix.length())-1;
                while (i>=0 && tenPrefix[i]!='\n') {
                        spacer[rank-1] += ' ';
                        i--;
                }
                for (std::size_t k=0; k<rank-1; k++) {
                        int i = int(prefix[k].length())-1;
                        while (i>=0 && prefix[k][i]!='\n') {
                                spacer[k] += ' ';
                                i--;
                        }
                }
        }

        std::string tenPrefix;
        std::string tenSuffix;
        std::array<std::string,rank> prefix;
        std::array<std::string,rank> suffix;
        std::array<std::string,rank> separator;
        char fill;
        int precision;
        int flags;
        std::array<std::string, rank> spacer{};
};

namespace internal {
        template<std::size_t rank>
        TensorIOFormat<rank> gen_numpy_format() {
                TensorIOFormat<rank> NumpyFormat(StreamPrecision, 0, "array([" , "])");
                return NumpyFormat;
        }

        template<std::size_t rank>
        TensorIOFormat<rank> gen_short_format() {
                std::array<std::string, rank> separator{}; separator.back() = ", ";
                std::array<std::string, rank> suffix{};
                std::array<std::string, rank> prefix{};

                TensorIOFormat<rank> ShortFormat(separator, prefix, suffix, StreamPrecision, 1, " << " , ";");
                return ShortFormat;
        }

        template<std::size_t rank>
        TensorIOFormat<rank> gen_eigen_format() {
                std::array<std::string, rank> separator{}; separator[rank-1] = " "; separator[rank-2] = "\n"; separator[rank-3] = "\n";
                std::array<std::string, rank> suffix{};
                std::array<std::string, rank> prefix{};

                TensorIOFormat<rank> EigenFormat(separator, prefix, suffix, StreamPrecision, 0, "" , "", ' ');
                return EigenFormat;
        }

        template<>
        TensorIOFormat<0> gen_eigen_format() {
                std::array<std::string, 0> separator{};
                std::array<std::string, 0> suffix{};
                std::array<std::string, 0> prefix{};

                TensorIOFormat<0> EigenFormat(separator, prefix, suffix, StreamPrecision, 0, "" , "",' ');
                return EigenFormat;
        }

        template<>
        TensorIOFormat<1> gen_eigen_format() {
                std::array<std::string, 1> separator{}; separator[0] = " ";
                std::array<std::string, 1> suffix{};
                std::array<std::string, 1> prefix{};

                TensorIOFormat<1> EigenFormat(separator, prefix, suffix, StreamPrecision, 0, "" , "",' ');
                return EigenFormat;
        }

        template<>
        TensorIOFormat<2> gen_eigen_format() {
                std::array<std::string, 2> separator{}; separator[1] = " "; separator[0] = "\n";
                std::array<std::string, 2> suffix{};
                std::array<std::string, 2> prefix{};

                TensorIOFormat<2> EigenFormat(separator, prefix, suffix, StreamPrecision, 0, "" , "",' ');
                return EigenFormat;
        }
}

namespace IOFormats {
        template<std::size_t rank>
        const TensorIOFormat<rank> Numpy = internal::gen_numpy_format<rank>();

        template<std::size_t rank>
        const TensorIOFormat<rank> Short = internal::gen_short_format<rank>();

        template<std::size_t rank>
        const TensorIOFormat<rank> Native = internal::gen_eigen_format<rank>();
}

template<typename T, int Layout> class TensorWithFormat;

template<typename T>
class TensorWithFormat<T,RowMajor>
{
public:

        TensorWithFormat(const T& tensor, const TensorIOFormat<T::NumDimensions>& format)
                : t_tensor(tensor), t_format(format)
        {}

        friend std::ostream & operator << (std::ostream & os, const TensorWithFormat<T, RowMajor>& wf)
        {
                // Evaluate the expression if needed
                typedef TensorEvaluator<const TensorForcedEvalOp<const T>, DefaultDevice> Evaluator;
                TensorForcedEvalOp<const T> eval = wf.t_tensor.eval();
                Evaluator tensor(eval, DefaultDevice());
                tensor.evalSubExprsIfNeeded(NULL);
                static const std::size_t rank = internal::array_size<typename Evaluator::Dimensions>::value;
                internal::TensorPrinter<Evaluator, rank>::run(os, tensor, wf.t_format);
                // Cleanup.
                tensor.cleanup();
                return os;
        }

protected:
        T t_tensor;
        TensorIOFormat<T::NumDimensions> t_format;
};

template<typename T, int Layout> class TensorWithFormat;

template<typename T>
class TensorWithFormat<T,ColMajor>
{
  public:
        TensorWithFormat(const T& tensor, const TensorIOFormat<T::NumDimensions>& format)
      : t_tensor(tensor), t_format(format)
    {}

    friend std::ostream & operator << (std::ostream & os, const TensorWithFormat<T, ColMajor>& wf)
    {
            // Switch to RowMajor storage and print afterwards
            static const std::size_t rank = T::NumDimensions;
            typedef typename T::Index Index;
            std::array<Index, rank> shuffle;
            std::array<Index, rank> id; std::iota(id.begin(), id.end(), Index(0));
            std::copy(id.begin(), id.end(), shuffle.rbegin());
            auto tensor_row_major = wf.t_tensor.swap_layout().shuffle(shuffle);

            // Evaluate the expression if needed
            typedef TensorEvaluator<const TensorForcedEvalOp<const decltype(tensor_row_major)>, DefaultDevice> Evaluator;
            TensorForcedEvalOp<const decltype(tensor_row_major)> eval = tensor_row_major.eval();
            Evaluator tensor(eval, DefaultDevice());
            tensor.evalSubExprsIfNeeded(NULL);
            internal::TensorPrinter<Evaluator, rank>::run(os, tensor, wf.t_format);
            // Cleanup.
            tensor.cleanup();
            return os;
    }

  protected:
    T t_tensor;
    TensorIOFormat<T::NumDimensions> t_format;
};

namespace internal {
template <typename Tensor, std::size_t rank>
struct TensorPrinter {
        static void run (std::ostream& s, const Tensor& _t, const TensorIOFormat<rank>& fmt) {
                typedef typename Tensor::Scalar Scalar;
                typedef typename Tensor::Index Index;
                static const int layout = Tensor::Layout;
                assert(layout == RowMajor);

                typedef typename
                        conditional<
                                is_same<Scalar, char>::value ||
                        is_same<Scalar, unsigned char>::value ||
                        is_same<Scalar, numext::int8_t>::value ||
                        is_same<Scalar, numext::uint8_t>::value,
                        int,
                        typename conditional<
                                is_same<Scalar, std::complex<char> >::value ||
                                is_same<Scalar, std::complex<unsigned char> >::value ||
                                is_same<Scalar, std::complex<numext::int8_t> >::value ||
                                is_same<Scalar, std::complex<numext::uint8_t> >::value,
                                std::complex<int>,
                                const Scalar&
                                >::type
                        >::type PrintType;

                const Index total_size = array_prod(_t.dimensions());

                std::streamsize explicit_precision;
                if(fmt.precision == StreamPrecision) {
                        explicit_precision = 0;
                }
                else if(fmt.precision == FullPrecision) {
                        if (NumTraits<Scalar>::IsInteger) {
                                explicit_precision = 0;
                        }
                        else {
                                explicit_precision = significant_decimals_impl<Scalar>::run();
                        }
                }
                else {
                        explicit_precision = fmt.precision;
                }

                std::streamsize old_precision = 0;
                if(explicit_precision) old_precision = s.precision(explicit_precision);

                Index width = 0;

                bool align_cols = !(fmt.flags & DontAlignCols);
                if(align_cols) {
                        // compute the largest width
                        for (Index i=0; i<total_size; i++) {
                                std::stringstream sstr;
                                sstr.copyfmt(s);
                                sstr << static_cast<PrintType>(_t.data()[i]);
                                width = std::max<Index>(width, Index(sstr.str().length()));
                        }
                }
                std::streamsize old_width = s.width();
                char old_fill_character = s.fill();

                s << fmt.tenPrefix;
                for (Index i = 0; i<total_size; i++) {
                        std::array<bool, rank> IS_AT_END{};
                        std::array<bool, rank> IS_AT_BEGIN{};

                        //is the ith element the end of an coeff (always true), of a row, of a matrix, ...?
                        for (std::size_t k=0; k<rank; k++) {
                                if ((i+1) % (std::accumulate(_t.dimensions().rbegin(), _t.dimensions().rbegin()+k, 1, std::multiplies<Index>())) == 0) {
                                        IS_AT_END[rank-1-k] = true;
                                }
                        }

                        //is the ith element the begin of an coeff (always true), of a row, of a matrix, ...?
                        for (std::size_t k=0; k<rank; k++) {
                                if (i % (std::accumulate(_t.dimensions().rbegin(), _t.dimensions().rbegin()+k, 1, std::multiplies<Index>())) == 0) {
                                        IS_AT_BEGIN[rank-1-k] = true;
                                }
                        }

                        //do we have a line break?
                        bool IS_AT_BEGIN_AFTER_NEWLINE=false;
                        for (std::size_t k=0; k<rank; k++) {
                                if (IS_AT_BEGIN[k]) {
                                        if(fmt.separator[k].find('\n') != std::string::npos) {IS_AT_BEGIN_AFTER_NEWLINE=true;}
                                }
                        }

                        bool IS_AT_END_BEFORE_NEWLINE=false;
                        for (std::size_t k=0; k<rank; k++) {
                                if (IS_AT_END[k]) {
                                        if(fmt.separator[k].find('\n') != std::string::npos) {IS_AT_END_BEFORE_NEWLINE=true;}
                                }
                        }

                        std::stringstream suffix, prefix, separator;
                        for (int k=rank-1; k>=0; k--) {
                                if (IS_AT_END[k]) {suffix << fmt.suffix[k];}
                        }
                        for (int k=rank-1; k>=0; k--) {
                                if (IS_AT_END[k] and (!IS_AT_END_BEFORE_NEWLINE or fmt.separator[k].find('\n') != std::string::npos)) {separator << fmt.separator[k]; }
                        }
                        for (int k=rank-1; k>=0; k--) {
                                if (i!=0 and IS_AT_BEGIN_AFTER_NEWLINE and (!IS_AT_BEGIN[k] or k==rank-1)) {prefix << fmt.spacer[k];}
                        }
                        for (int k=0; k<static_cast<int>(rank); k++) {
                                if (IS_AT_BEGIN[k]) {prefix << fmt.prefix[k];}
                        }

                        s << prefix.str();
                        if(width) {
                                s.fill(fmt.fill);
                                s.width(width);
                                s << std::left;
                        }
                        s << _t.data()[i];
                        s << suffix.str();
                        if (i < total_size -1) {s << separator.str();}
                }
                s << fmt.tenSuffix;
                if(explicit_precision) s.precision(old_precision);
                if(width) {
                        s.fill(old_fill_character);
                        s.width(old_width);
                }
        }
};

template <typename Tensor>
struct TensorPrinter<Tensor, 0> {
        static void run (std::ostream& s, const Tensor& _t, const TensorIOFormat<0>& fmt) {
                typedef typename Tensor::Scalar Scalar;
                typedef typename Tensor::Index Index;

                const Index total_size = array_prod(_t.dimensions());

                std::streamsize explicit_precision;
                if(fmt.precision == StreamPrecision) {
                        explicit_precision = 0;
                }
                else if(fmt.precision == FullPrecision) {
                        if (NumTraits<Scalar>::IsInteger) {
                                explicit_precision = 0;
                        }
                        else {
                                explicit_precision = significant_decimals_impl<Scalar>::run();
                        }
                }
                else {
                        explicit_precision = fmt.precision;
                }

                std::streamsize old_precision = 0;
                if(explicit_precision) old_precision = s.precision(explicit_precision);

                s << fmt.tenPrefix << _t.coeff(0) << fmt.tenSuffix;
                if(explicit_precision) s.precision(old_precision);
        }
};

} // end namespace internal
template<typename T>
std::ostream & operator <<(std::ostream & s, const TensorBase<T,ReadOnlyAccessors> & t)
{
        s << t.format(IOFormats::Native<T::NumDimensions>);
        return s;
}
} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_IO_H
