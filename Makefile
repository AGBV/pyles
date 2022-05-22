.DEFAULT_GOAL := help

help:
	@echo "This is a makefile"

cython: Makefile
	cd pyles/functions/coupling_matrix/cpu/cython; python setup.py build_ext; rm -r build

ctypes: Makefile
	cd pyles/functions/coupling_matrix/cpu/ctypes; gcc -shared -o libcoupling.so -fPIC coupling_matrix_multiply_cpu.c;

cffi: Makefile
	cd pyles/functions/coupling_matrix/cpu/cffi; python compile.py;