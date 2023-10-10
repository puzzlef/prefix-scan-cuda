Design of an efficient algorithm for parallel prefix-scan of a large array of
values on GPUs.

It appears both CUB and Thrust have a similar performance.

<br>

```log
OMP_NUM_THREADS=64
{1.000e+06 values} -> {0000000.8ms, 1500000 last_value} inclusiveScanOmp
{1.000e+06 values} -> {0000000.4ms, 1500000 last_value} inclusiveScanCudaCub
{1.000e+06 values} -> {0000000.3ms, 1500000 last_value} inclusiveScanCudaThrust
{1.000e+06 values} -> {0000000.8ms, 1499997 last_value} exclusiveScanOmp
{1.000e+06 values} -> {0000000.1ms, 1499997 last_value} exclusiveScanCudaCub
{1.000e+06 values} -> {0000000.3ms, 1499997 last_value} exclusiveScanCudaThrust
{1.000e+07 values} -> {0000004.9ms, 15000000 last_value} inclusiveScanOmp
{1.000e+07 values} -> {0000000.3ms, 15000000 last_value} inclusiveScanCudaCub
{1.000e+07 values} -> {0000000.5ms, 15000000 last_value} inclusiveScanCudaThrust
{1.000e+07 values} -> {0000004.9ms, 14999997 last_value} exclusiveScanOmp
{1.000e+07 values} -> {0000000.3ms, 14999997 last_value} exclusiveScanCudaCub
{1.000e+07 values} -> {0000000.5ms, 14999997 last_value} exclusiveScanCudaThrust
{1.000e+08 values} -> {0000074.7ms, 150000000 last_value} inclusiveScanOmp
{1.000e+08 values} -> {0000002.1ms, 150000000 last_value} inclusiveScanCudaCub
{1.000e+08 values} -> {0000002.4ms, 150000000 last_value} inclusiveScanCudaThrust
{1.000e+08 values} -> {0000076.0ms, 149999997 last_value} exclusiveScanOmp
{1.000e+08 values} -> {0000002.1ms, 149999997 last_value} exclusiveScanCudaCub
{1.000e+08 values} -> {0000002.3ms, 149999997 last_value} exclusiveScanCudaThrust
{1.000e+09 values} -> {0000815.9ms, 1500000000 last_value} inclusiveScanOmp
{1.000e+09 values} -> {0000020.3ms, 1500000000 last_value} inclusiveScanCudaCub
{1.000e+09 values} -> {0000020.7ms, 1500000000 last_value} inclusiveScanCudaThrust
{1.000e+09 values} -> {0000733.8ms, 1499999997 last_value} exclusiveScanOmp
{1.000e+09 values} -> {0000020.1ms, 1499999997 last_value} exclusiveScanCudaCub
{1.000e+09 values} -> {0000020.4ms, 1499999997 last_value} exclusiveScanCudaThrust
```


## References

- [CUB: cub::DeviceScan Struct Reference](https://nvlabs.github.io/cub/structcub_1_1_device_scan.html#a02b2d2e98f89f80813460f6a6ea1692b)
- [Introduction — thrust 12.2 documentation](https://docs.nvidia.com/cuda/thrust/index.html)
- [Thrust: thrust::exclusive_scan](https://thrust.github.io/doc/group__prefixsums_ga7be5451c96d8f649c8c43208fcebb8c3.html)
- [GPU Gems 3: Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
- [Fetch-and-add using OpenMP atomic operations](https://stackoverflow.com/a/7918281/1413259)

<br>
<br>


[![](https://img.youtube.com/vi/yqO7wVBTuLw/maxresdefault.jpg)](https://www.youtube.com/watch?v=yqO7wVBTuLw)<br>
[![ORG](https://img.shields.io/badge/org-puzzlef-green?logo=Org)](https://puzzlef.github.io)
