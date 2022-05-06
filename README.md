# Pyles

Pyles (**Py**thon Ce**les**) is a python implementation of the [CELES](https://github.com/disordered-photonics/celes) framework developed by Egel A, Pattelli L, Mazzamuto G, Wiersma DS, and Lemmer U.

It is primarily developed so anyone could run it due to python being [FOSS](https://en.wikipedia.org/wiki/Free_and_open-source_software) MATLAB, on which CELES is implemented, is not.

Pyles will primarily be CPU-driven with a plan on utilizing [OpenCL](https://www.khronos.org/opencl/) down the road for more parallelization of the processes.

## Performance
The latest profiler result can be viewser [here](https://www.speedscope.app/#title=Pyles%20main.py%20profile&profileURL=https%3A%2F%2Fraw.githubusercontent.com%2FAGBV%2Fpyles%2Fgh-pages%2Fprofile.speedscope.json).
On each commit to the `pyles` folder or the `main.py` file ([master](https://github.com/AGBV/pyles/tree/master) branch), the [py-spy](https://github.com/benfred/py-spy) profiler is run. The data is saved in the [gh-pages](https://github.com/AGBV/pyles/tree/gh-pages) branch and can be viewed using [speedscope](https://github.com/jlfwong/speedscope).



# Code Improvements

- Uses the indices provided by the numpy `unique` function (just like the matlab function) and bypasses the usage of a similar function to `dsearchn` in matlab. **Note**: Same improvement can be done in Matlab too!
- The provided function for the Wigner 3j symbold `wigner3j` is not numerically stable for large values of `j` (upper row). As an alterntive the `pywigxjpf` package is used which is a python wrapper for the [wigxjpf](http://fy.chalmers.se/subatom/wigxjpf/) C++ framework. Due to some mathematical *tricks*, a [speedup](https://paperzz.com/doc/8141260/wigner-3j--6j-and-9j-symbols) is enabled which also results in running more stable than the [Matlab alternative](https://de.mathworks.com/matlabcentral/fileexchange/5275-wigner3j-m).

# Devlopment
Before a commit can be executed, the `pre-commit` hook will be triggered. This will run all tests, and only then will a commit be approved! 

To enable this on your end, please execute `git config --local core.hooksPath .githooks/` so the hooks in `.githooks` are used.

If you rather prefere using [pre-commit](https://pre-commit.com/), a `.pre-commit-config.yaml` is also provided :)