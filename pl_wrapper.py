## Module to wrap the CUDA function

import ctypes as C

# find and load the library
lpl = C.cdll.LoadLibrary("./libplasmaline.so")
# set the argument types for input to the wrapped function
lpl.process_echoes.argtypes = [C.POINTER(C.c_float), C.POINTER(C.c_float),\
                               C.c_int, C.c_int, C.c_int,\
                               C.POINTER(C.c_float),\
                               C.c_int, C.c_int, C.c_int]
# set the return type
lpl.process_echoes.restype = None

# python wrapper method for CUDA process
def process_echoes(tx_conj, echo, tx_length, ipp_length, n_ipp,spectrum, n_range_gates,\
                   range_gate_step, range_gate_start):
    """ Wrapper for process_echoes in plasmaline.cu and libplasmaline.so """

    lpl.process_echoes(tx_conj.ctypes.data_as(C.POINTER(C.c_float)),\
                       echo.ctypes.data_as(C.POINTER(C.c_float)),\
                       C.c_int(tx_length),\
                       C.c_int(ipp_length),\
                       C.c_int(n_ipp),\
                       spectrum.ctypes.data_as(C.POINTER(C.c_float)),\
                       C.c_int(n_range_gates),\
                       C.c_int(range_gate_step),\
                       C.c_int(range_gate_start))
