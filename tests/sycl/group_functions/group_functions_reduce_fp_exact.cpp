/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full test details.
 */
// SPDX-License-Identifier: BSD-2-Clause

// Regression tests for the numerical exactness of floating point
// reduce_over_group(work_group, ...).
//
// Background: the SSCP work-group reduction (hipsycl::libkernel::sscp::wg_reduce,
// sscp/builtins/detail/reduction.hpp) broadcasts the final result by bit-casting
// it to its integer representation and storing it into shared memory typed as
// the floating point type. Assigning the raw bit pattern of a double/float to a
// __acpp_f64/__acpp_f64 storage element goes through an implicit intN ->
// float/double NUMERIC conversion, destroying all selected bits that do not fit
// the destination mantissa. As a result the reduction output can be distorted
// by several hundred ULPs (observed: ~206 ULP for double, ~25 ULP for float)
// even though every summation order of the inputs would produce the same bits.
//
// These tests use input data where every possible summation order (linear
// folds, binary trees - verified exhaustively) yields bit-identical results,
// so the implementation may accumulate in any order or even in extended
// precision and still yield an exactly predictable result. Distorting the
// final bits therefore produces an unmistakable test failure, while correctly
// rounded implementations pass.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "../sycl_test_suite.hpp"
#include "group_functions.hpp"

#ifdef HIPSYCL_ENABLE_GROUP_ALGORITHM_TESTS

BOOST_FIXTURE_TEST_SUITE(group_functions_tests, reset_device_fixture)

namespace fp_exact_detail {

template <int Line, typename T>
class fp_exact_reduce_kernel;

template <typename T>
std::uint64_t bit_pattern(T x) {
  static_assert(sizeof(T) == 4 || sizeof(T) == 8);
  if constexpr (sizeof(T) == 4) {
    std::uint32_t i;
    std::memcpy(&i, &x, sizeof(i));
    return i;
  } else {
    std::uint64_t i;
    std::memcpy(&i, &x, sizeof(i));
    return i;
  }
}

// Bit-exact equality (implemented on the bit patterns so that Boost.Test's
// tolerance-based float comparison logic cannot kick in).
template <typename T>
bool bit_equal(T a, T b) {
  return bit_pattern(a) == bit_pattern(b);
}

// Number of ULPs between a and b (both positive or both negative; the tests
// only produce positive results between closest neighbors).
template <typename T>
int ulp_distance(T a, T b) {
  if (std::isnan(a) || std::isnan(b)) return -1;
  if ((std::signbit(a) != std::signbit(b)) && !(a == 0 && b == 0)) return -1;

  T lo = std::min(a, b);
  T hi = std::max(a, b);

  bool both_positive = !std::signbit(lo);
  if (both_positive) {
    // No sign boundary: the bit-pattern difference is exactly the ULP count.
    return static_cast<int>(bit_pattern(hi) - bit_pattern(lo));
  }

  // Fallback: walk with nextafter, bounded.
  T cur = lo;
  int count = 0;
  constexpr int max_walk = 4096;
  while (cur != hi && count < max_walk) {
    cur = std::nextafter(cur, hi);
    ++count;
  }
  return count;
}

// Runs reduce_over_group over several work groups for the given input pattern,
// checking both the "with init" and the "without init" variants. Each work
// group receives the identical pattern, so every work group must yield the
// same value.
template <typename T, std::size_t N, int Line>
void run_reduce_fp_exact_case(const char *name, const T (&pattern)[N],
                              int max_ulp) {
  sycl::queue queue;

  if constexpr (std::is_same_v<T, double>) {
    if (!queue.get_device().has(sycl::aspect::fp64)) {
      BOOST_TEST_MESSAGE("Skipping test for double since device has no fp64 support");
      return;
    }
  }

  constexpr int num_groups = 4;
  constexpr std::size_t global_size = num_groups * N;

  // Reference: host fold of the pattern. The test cases were chosen such that
  // exhaustively checked summation orders all yield the same number of bits
  // (or differ by at most a few ULPs bounded by max_ulp).
  T expected = T{};
  for (std::size_t j = 0; j < N; ++j)
    expected = expected + pattern[j];

  std::vector<T> in_data(global_size);
  for (std::size_t i = 0; i < global_size; ++i)
    in_data[i] = pattern[i % N];

  std::vector<T> out_data(2 * global_size, T{});

  {
    sycl::buffer<T, 1> in_buf{in_data.data(), sycl::range<1>(global_size)};
    sycl::buffer<T, 1> out_buf{out_data.data(), sycl::range<1>(2 * global_size)};

    queue.submit([&](sycl::handler &cgh) {
      auto in_acc = in_buf.template get_access<sycl::access::mode::read>(cgh);
      auto out_acc = out_buf.template get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for<class fp_exact_reduce_kernel<Line, T>>(
          sycl::nd_range<1>{global_size, N}, [=](sycl::nd_item<1> item) {
            auto g = item.get_group();
            auto gid = item.get_global_linear_id();
            T local_value = in_acc[gid];

            out_acc[2 * gid + 0] =
                sycl::reduce_over_group(g, local_value, T{}, sycl::plus<T>());
            out_acc[2 * gid + 1] =
                sycl::reduce_over_group(g, local_value, sycl::plus<T>());
          });
    });
  }

  const auto check = [&](std::size_t idx, T computed, const char *variant) {
    if (max_ulp == 0) {
      BOOST_TEST(bit_equal(computed, expected),
                 "fp exact reduce regression (" << name << ", " << variant
                 << ", group " << (idx / N) << ", item " << idx % N
                 << "): got bit pattern 0x" << std::hex << bit_pattern(computed)
                 << " (" << +computed << ") instead of 0x"
                 << bit_pattern(expected) << " (" << +expected << ")");
    } else {
      int d = ulp_distance(computed, expected);
      bool ok = (d >= 0) && (d <= max_ulp);
      BOOST_TEST(ok,
                 "fp exact reduce regression (" << name << ", " << variant
                 << ", group " << (idx / N) << ", item " << idx % N
                 << "): got (" << +computed << ") instead of ("
                 << +expected << "), " << d << " ULPs (" << max_ulp
                 << " allowed)");
    }
  };

  for (std::size_t i = 0; i < global_size; ++i) {
    check(i, out_data[2 * i + 0], "with init");
    check(i, out_data[2 * i + 1], "without init");
  }
}

} // namespace fp_exact_detail

/*
 * For each of the following patterns it was exhaustively verified (host-side
 * brute force over all left folds and all binary tree foldings) that every
 * summation order yields a single bit-identical result, so the comparison
 * below is exact and cannot be affected by a legal reordering of the
 * reduction.
 */

BOOST_AUTO_TEST_CASE(group_reduce_fp_exact_double_2) {
  // The canonical repro: reduce [2.6, 0] - the correct result is the
  // bit pattern of 2.6 itself, while the pre-fix implementation returned
  // 2.5999999999999091 (0x4004CCCCCCCCCC00, ~205 ULP below).
  const double pattern[2] = {2.6, 0.0};
  fp_exact_detail::run_reduce_fp_exact_case<double, 2, __LINE__>(
      "double {2.6, 0}", pattern, 0);
}

BOOST_AUTO_TEST_CASE(group_reduce_fp_exact_double_4) {
  // Two copies of the canonical repro aligned to 4 items.
  const double pattern[4] = {2.6, 0.0, 2.6, 0.0};
  fp_exact_detail::run_reduce_fp_exact_case<double, 4, __LINE__>(
      "double {2.6, 0, 2.6, 0}", pattern, 0);
}

BOOST_AUTO_TEST_CASE(group_reduce_fp_exact_double_8) {
  const double pattern[8] = {2.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  fp_exact_detail::run_reduce_fp_exact_case<double, 8, __LINE__>(
      "double {2.6, 0 x7}", pattern, 0);
}

BOOST_AUTO_TEST_CASE(group_reduce_fp_exact_double_mixed) {
  // Order-invariant by construction: exhaustive verification shows that every
  // summation order (linear fold or binary tree) of {0.5, 1.2, -2.8, 3.7}
  // yields the same bits. The pre-fix implementation was ~206 ULP below.
  const double pattern[4] = {0.5, 1.2, -2.8, 3.7};
  fp_exact_detail::run_reduce_fp_exact_case<double, 4, __LINE__>(
      "double {0.5, 1.2, -2.8, 3.7}", pattern, 0);
}

BOOST_AUTO_TEST_CASE(group_reduce_fp_exact_double_stable_sum) {
  // Not bit-order invariant (5 distinct correct fold results, spread of 4
  // ULPs), so compared with a small bounded ULP tolerance. The pre-fix
  // implementation was off by >= 203 ULPs from every correct summation order.
  const double pattern[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  fp_exact_detail::run_reduce_fp_exact_case<double, 8, __LINE__>(
      "double {0.1..0.8}", pattern, 8);
}

BOOST_AUTO_TEST_CASE(group_reduce_fp_exact_float_2) {
  // Canonical float repro: pre-fix implementation returned
  // 2.600006103515625 (0x40266680, ~25 ULP above).
  const float pattern[2] = {2.6f, 0.0f};
  fp_exact_detail::run_reduce_fp_exact_case<float, 2, __LINE__>(
      "float {2.6f, 0}", pattern, 0);
}

BOOST_AUTO_TEST_CASE(group_reduce_fp_exact_float_4) {
  const float pattern[4] = {2.6f, 0.0f, 2.6f, 0.0f};
  fp_exact_detail::run_reduce_fp_exact_case<float, 4, __LINE__>(
      "float {2.6f, 0, 2.6f, 0}", pattern, 0);
}

BOOST_AUTO_TEST_CASE(group_reduce_fp_exact_float_mixed) {
  // 3 distinct correct fold results spread <= 2 ULP; pre-fix implementation
  // was 23-26 ULPs away from every correct order.
  const float pattern[4] = {0.5f, 1.2f, -2.8f, 3.7f};
  fp_exact_detail::run_reduce_fp_exact_case<float, 4, __LINE__>(
      "float {0.5, 1.2, -2.8, 3.7}", pattern, 8);
}

BOOST_AUTO_TEST_SUITE_END()

#endif // HIPSYCL_ENABLE_GROUP_ALGORITHM_TESTS
