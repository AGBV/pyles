# Pyles

Pyles (**Py**thon Ce**les**) is a python implementation of the [CELES](https://github.com/disordered-photonics/celes) framework developed by Egel A, Pattelli L, Mazzamuto G, Wiersma DS, and Lemmer U.
It is primarily developed so anyone could run it due to python being [FOSS](https://en.wikipedia.org/wiki/Free_and_open-source_software), while MATLAB, on which CELES is implemented, is not.

<del>Pyles will primarily be CPU-driven with a plan on utilizing [OpenCL](https://www.khronos.org/opencl/) down the road for more parallelization of the processes.

The heavy lifting of pyles is realised using [numba](https://numba.pydata.org/) to parallelise the coupling matrix computation either on the CPU (see [here](https://numba.readthedocs.io/en/stable/user/performance-tips.html) some nice performance tips) or even GPU using CUDA (numba provides a CUDA [implementation](https://numba.readthedocs.io/en/stable/cuda/index.html)). This project is in early works and still incomplete, so performance improvements down the road are possible.

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

# Credits

- This project is based on the already available work of the people from [disordered-photonics](https://github.com/disordered-photonics) and their [CELES](https://github.com/disordered-photonics/celes) implementation in Matlab. For the prototype, I am currently mostly implementing their functions 1:1 to python with small tweaks here and there. For more information regarding their work, lookup their mentioned repository or their [published paper](https://www.sciencedirect.com/science/article/abs/pii/S0022407317301772) on CELES. When this project hits its release and you consider useing it for your research, please cite their work [bibtext](https://github.com/disordered-photonics/celes/blob/master/doc/celes.bib)!
- Thanks to H. T. Johansson and C. Forssén for the CPython implementation of their [WIGXJPF](http://fy.chalmers.se/subatom/wigxjpf/) project. More information on how it works is provided in [their publication](https://epubs.siam.org/doi/10.1137/15M1021908).# pyopenmensa-dortmund
