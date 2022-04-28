# Pyles
Pyles (**Py**thon Ce**les**) is a python implementation of the [CELES](https://github.com/disordered-photonics/celes) framework developed by Egel A, Pattelli L, Mazzamuto G, Wiersma DS, and Lemmer U.

It is primarily developed so anyone could run it, due to python being [FOSS](https://en.wikipedia.org/wiki/Free_and_open-source_software), while MATLAB, on which CELES is implemented, is nott.

Pyles will be primarly be CPU driven with a plan on utilizing opencl down the road for more parallelization of the processes.

# Devlopment
Before a commit can be executed, the `pre-commit` hook will be triggered. This will run all tests and only then will a commit be approved! 

To enable this on your end, please execute `git config --local core.hooksPath .githooks/` so the hooks in `.githooks` are used.

If you rather prefere using [pre-commit](https://pre-commit.com/), a `.pre-commit-config.yaml` is also provided :)