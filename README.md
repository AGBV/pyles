# Pyles

Pyles (**Py**thon Ce**les**) is a python implementation of the [CELES](https://github.com/disordered-photonics/celes) framework developed by Egel A, Pattelli L, Mazzamuto G, Wiersma DS, and Lemmer U.

It is primarily developed so anyone could run it due to python being [FOSS](https://en.wikipedia.org/wiki/Free_and_open-source_software) MATLAB, on which CELES is implemented, is not.

Pyles will primarily be CPU-driven with a plan on utilizing [OpenCL](https://www.khronos.org/opencl/) down the road for more parallelization of the processes.

# Code Improvements

- Uses the indices provided by the numpy `unique` function (just like the matlab function) and bypasses the usage of a similar function to `dsearchn` in matlab. **Note**: Same improvement can be done in Matlab too!
- The provided function for the Wigner 3j symbold `wigner3j` is not numerically stable for large values of `j` (upper row). As an alterntive the `pywigxjpf` package is used which is a python wrapper for the [wigxjpf](http://fy.chalmers.se/subatom/wigxjpf/) C++ framework. Due to some mathematical *tricks*, a speedup is enabled which also results in running more stable than the [Matlab alternative](https://de.mathworks.com/matlabcentral/fileexchange/5275-wigner3j-m).

# Devlopment
Before a commit can be executed, the `pre-commit` hook will be triggered. This will run all tests, and only then will a commit be approved! 

To enable this on your end, please execute `git config --local core.hooksPath .githooks/` so the hooks in `.githooks` are used.

If you rather prefere using [pre-commit](https://pre-commit.com/), a `.pre-commit-config.yaml` is also provided :)