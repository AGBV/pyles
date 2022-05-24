# GPU Implementation

GPU functionality is enabled using [numba](https://numba.pydata.org/). Sadly only [CUDA](https://developer.nvidia.com/cuda-toolkit) is compatible with and AMD support has been depricated since version [0.54.0](https://github.com/numba/numba/releases/tag/0.54.0) ([release notes](https://numba.readthedocs.io/en/0.54.0/release-notes.html)).

The CPU version of the code also runs parallel on multiple threads and is not that much slower than the GPU version! This could be since there is a lot of data to be moved to the GPU and back.. Further optimizations would be needed.

<del>There are multiple ways to implement this on the GPU.
A naive way would be to use the already implemented [CUDA file from CELES](https://github.com/disordered-photonics/celes/blob/master/src/scattering/coupling_matrix_multiply_CUDA.cu), i.e., [`coupling_matrix_multiply_CUDA.cu`](coupling_matrix_multiply_CUDA.cu). It needs to be done in a python compatible manner. Furthermore, one could also try to translate the problem from for-loops to a large matrix products, at which the GPU excels most.