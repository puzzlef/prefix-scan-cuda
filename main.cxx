#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>
#include "inc/main.hxx"

using namespace std;




#pragma region CONFIGURATION
#ifndef MAX_THREADS
/** Maximum number of threads to use. */
#define MAX_THREADS 64
#endif
#ifndef REPEAT_METHOD
/** Number of times to repeat each method. */
#define REPEAT_METHOD 5
#endif
#pragma endregion




#pragma region METHODS
#pragma region PERFORM EXPERIMENT
/**
 * Perform the experiment.
 */
void runExperiment() {
  int repeat = REPEAT_METHOD;
  // Follow a specific result logging format, which can be easily parsed later.
  auto glog = [&](const auto& ans, const char *technique, size_t N) {
    printf(
      "{%.3e values} -> {%09.1fms, %zu last_value} %s\n",
      double(N), ans.time, ans.values[N-1], technique
    );
  };
  for (size_t N=1000000; N<=1000000000; N*=10) {
    vector<size_t> x(N);
    // Populate the array with random values.
    for (size_t i=0; i<N; ++i)
      x[i] = i % 4;
    // Perform Inclusive Scan.
    auto a1 = inclusiveScanCudaCub(x, {repeat});
    glog(a1, "inclusiveScanCudaCub", N);
    auto a2 = inclusiveScanCudaThrust(x, {repeat});
    glog(a2, "inclusiveScanCudaThrust", N);
    // Perform Exclusive Scan.
    auto b1 = exclusiveScanCudaCub(x, {repeat});
    glog(b1, "exclusiveScanCudaCub", N);
    auto b2 = exclusiveScanCudaThrust(x, {repeat});
    glog(b2, "exclusiveScanCudaThrust", N);
  }
}


/**
 * Main function.
 * @param argc argument count
 * @param argv argument values
 * @returns zero on success, non-zero on failure
 */
int main(int argc, char **argv) {
  install_sigsegv();
  omp_set_num_threads(MAX_THREADS);
  LOG("OMP_NUM_THREADS=%d\n", MAX_THREADS);
  runExperiment();
  printf("\n");
  return 0;
}
#pragma endregion
#pragma endregion
