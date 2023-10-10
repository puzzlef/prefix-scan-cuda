#pragma once
#include <utility>
#include <vector>
#include <omp.h>
#include "_main.hxx"

using std::vector;
using std::move;




#pragma region TYPES
/**
 * Options for Inclusive/Exclusive Scan algorithm.
 */
struct ScanOptions {
  #pragma region DATA
  /** Number of times to repeat the algorithm [1]. */
  int repeat;
  #pragma endregion


  #pragma region CONSTRUCTORS
  /**
   * Define options for Inclusive/Exclusive Scan algorithm.
   * @param repeat number of times to repeat the algorithm [1]
   */
  ScanOptions(int repeat=1) :
  repeat(repeat) {}
  #pragma endregion
};
#pragma endregion




/**
 * Result of Inclusive/Exclusive Scan algorithm.
 * @tparam T value type
 */
template <class T>
struct ScanResult {
  #pragma region DATA
  /** Values obtained after performing Inclusive/Exclusive Scan. */
  vector<T> values;
  /** Time spent in milliseconds. */
  float time;
  #pragma endregion


  #pragma region CONSTRUCTORS
  /**
   * Result of Inclusive/Exclusive Scan algorithm.
   * @param values values obtained after performing Inclusive/Exclusive Scan
   * @param time time spent in milliseconds
   */
  ScanResult(vector<T>&& values, float time=0) :
  values(values), time(time) {}


  /**
   * Result of Inclusive/Exclusive Scan algorithm.
   * @param values values obtained after performing Inclusive/Exclusive Scan
   * @param time time spent in milliseconds
   */
  ScanResult(vector<T>& values, float time=0) :
  values(move(values)), time(time) {}
  #pragma endregion
};




#pragma region ENVIRONMENT SETUP
#ifdef OPENMP
/**
 * Perform Inclusive/Exclusive Scan algorithm.
 * @param x input values
 * @param o scan options
 * @param fs perform Inclusive/Exclusive Scan (a, buft)
 * @returns scan result
 */
template <class T, class FS>
inline ScanResult<T> scanInvokeOmp(const vector<T>& x, const ScanOptions& o, FS fs) {
  size_t N = x.size();
  int    H = omp_get_max_threads();
  vector<T> a(N);
  vector<T> buft(H);
  float t = measureDurationMarked([&](auto mark) {
    copyValuesOmpW(a, x);
    // Perform Inclusive/Exclusive Scan.
    mark([&]() { fs(a, buft); });
  }, o.repeat);
  // Return result.
  return {a, t};
}
#endif
#pragma endregion




#pragma region INCLUSIVE SCAN
#ifdef OPENMP
/**
 * Perform Inclusive Scan algorithm.
 * @param x input values
 * @param o scan options
 * @returns scan result
 */
template <class T>
inline ScanResult<T> inclusiveScanOmp(const vector<T>& x, const ScanOptions& o={}) {
  auto fs = [&](auto& a, auto& buft) { inclusiveScanOmpW(a, buft, a); };
  return scanInvokeOmp(x, o, fs);
}
#endif
#pragma endregion




#pragma region EXCLUSIVE SCAN
#ifdef OPENMP
/**
 * Perform Exclusive Scan algorithm.
 * @param x input values
 * @param o scan options
 * @returns scan result
 */
template <class T>
inline ScanResult<T> exclusiveScanOmp(const vector<T>& x, const ScanOptions& o={}) {
  auto fs = [&](auto& a, auto& buft) { exclusiveScanOmpW(a, buft, a); };
  return scanInvokeOmp(x, o, fs);
}
#endif
#pragma endregion
