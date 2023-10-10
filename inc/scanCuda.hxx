#pragma once
#include <utility>
#include <vector>
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
/**
 * Perform Inclusive/Exclusive Scan algorithm.
 * @param x input values
 * @param o scan options
 * @param fs perform Inclusive/Exclusive Scan (x, buft, N)
 * @returns scan result
 */
template <class T, class FS>
inline ScanResult<T> scanInvokeCuda(const vector<T>& x, const ScanOptions& o, FS fs) {
  T  *xD    = nullptr;
  T  *buftD = nullptr;
  size_t N = x.size();
  vector<T> a(N);
  // Allocate device memory.
  TRY_CUDA( cudaSetDeviceFlags(cudaDeviceMapHost) );
  TRY_CUDA( cudaMalloc(&xD,    N*sizeof(T)) );
  TRY_CUDA( cudaMalloc(&buftD, N*sizeof(T)) );
  float t = measureDurationMarked([&](auto mark) {
    // Copy data to device.
    TRY_CUDA( cudaMemcpy(xD, x.data(), N*sizeof(T), cudaMemcpyHostToDevice) );
    // Perform Inclusive/Exclusive Scan.
    mark([&]() {
      fs(xD, buftD, N);
      TRY_CUDA( cudaDeviceSynchronize() );
    });
  }, o.repeat);
  // Copy data to host.
  TRY_CUDA( cudaMemcpy(a.data(), xD, N*sizeof(T), cudaMemcpyDeviceToHost) );
  // Free device memory.
  TRY_CUDA( cudaFree(xD) );
  TRY_CUDA( cudaFree(buftD) );
  // Return result.
  return {a, t};
}
#pragma endregion




#pragma region INCLUSIVE SCAN
/**
 * Perform Inclusive Scan algorithm using CUB.
 * @param x input values
 * @param o scan options
 * @returns scan result
 */
template <class T>
inline ScanResult<T> inclusiveScanCudaCub(const vector<T>& x, const ScanOptions& o={}) {
  auto fs = [&](T *xD, T *buftD, size_t N) { inclusiveScanCubW(xD, buftD, xD, N); };
  return scanInvokeCuda(x, o, fs);
}


/**
 * Perform Inclusive Scan algorithm using Thrust.
 * @param x input values
 * @param o scan options
 * @returns scan result
 */
template <class T>
inline ScanResult<T> inclusiveScanCudaThrust(const vector<T>& x, const ScanOptions& o={}) {
  auto fs = [&](T *xD, T *buftD, size_t N) { inclusiveScanThrustW(xD, xD, N); };
  return scanInvokeCuda(x, o, fs);
}
#pragma endregion




#pragma region EXCLUSIVE SCAN
/**
 * Perform Exclusive Scan algorithm using CUB.
 * @param x input values
 * @param o scan options
 * @returns scan result
 */
template <class T>
inline ScanResult<T> exclusiveScanCudaCub(const vector<T>& x, const ScanOptions& o={}) {
  auto fs = [&](T *xD, T *buftD, size_t N) { exclusiveScanCubW(xD, buftD, xD, N); };
  return scanInvokeCuda(x, o, fs);
}


/**
 * Perform Exclusive Scan algorithm using Thrust.
 * @param x input values
 * @param o scan options
 * @returns scan result
 */
template <class T>
inline ScanResult<T> exclusiveScanCudaThrust(const vector<T>& x, const ScanOptions& o={}) {
  auto fs = [&](T *xD, T *buftD, size_t N) { exclusiveScanThrustW(xD, xD, N); };
  return scanInvokeCuda(x, o, fs);
}
#pragma endregion
