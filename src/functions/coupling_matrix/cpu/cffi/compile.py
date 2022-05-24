import pathlib
import cffi
from sympy import half_gcdex

""" Build the CFFI Python bindings """
print("Building CFFI Module")
ffi = cffi.FFI()

# this_dir = pathlib.Path().absolute() / "pyles/functions/coupling_matrix/cpu/cffi"
this_dir = pathlib.Path().absolute()
h_file_name = this_dir / "coupling_matrix_multiply_cpu.h"
with open(h_file_name) as h_file:
    h_data = ""
    for line in h_file.readlines():
        if not line.startswith("#"):
            h_data += line
    ffi.cdef(h_data)

source = """
#include "coupling_matrix_multiply_cpu.h"
#include <stdlib.h>
"""
ffi.set_source(
    "coupling_matrix",
    # Since you're calling a fully-built library directly, no custom source
    # is necessary. You need to include the .h files, though, because behind
    # the scenes cffi generates a .c file that contains a Python-friendly
    # wrapper around each of the functions.
    source,
    # The important thing is to include the pre-built lib in the list of
    # libraries you're linking against:
    sources=["coupling_matrix_multiply_cpu.c"],
    # libraries=["coupling_matrix"],
    # library_dirs=[this_dir.as_posix()],
    # extra_link_args=["-Wl,-rpath=."],
)

ffi.compile()
print("CFFI Module Compiled")