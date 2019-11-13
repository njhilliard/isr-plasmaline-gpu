#!/usr/bin/env python

import numpy
import ctypes
import pl_example
import matplotlib.pyplot as plt 
import ctypes as C

# find and load the library
lpl = ctypes.cdll.LoadLibrary("libplasmaline.so")
# set the argument type
lpl.process_echoes.argtypes = [C.POINTER(C.c_float),C.POINTER(C.c_float),C.c_int,C.c_int,C.c_int,C.POINTER(C.c_float),C.c_int,C.c_int,C.c_int]
# set the return type
lpl.process_echoes.restype = None
lpl.hello.argtypes = [C.POINTER(C.c_float),C.POINTER(C.c_float),C.c_int]

def process_echoes(tx,echo,tx_length,ipp_length,n_ipp,spectrum,n_range_gates,range_gate_step,range_gate_start):
    ''' Wrapper for process_echoes in fft.cu and libplasmaline.so '''
    #lpl.hello(tx.ctypes.data_as(C.POINTER(C.c_float)),spectrum.ctypes.data_as(C.POINTER(C.c_float)),C.c_int(tx_length*n_range_gates))
    lpl.process_echoes(tx.ctypes.data_as(C.POINTER(C.c_float)),echo.ctypes.data_as(C.POINTER(C.c_float)),C.c_int(tx_length),C.c_int(ipp_length),C.c_int(n_ipp),
                       spectrum.ctypes.data_as(C.POINTER(C.c_float)),C.c_int(n_range_gates),C.c_int(range_gate_step),C.c_int(range_gate_start))
 

if __name__ == "__main__":
    n_ipp = 20
    tx_length = 16384
    ipp_length = 250000
    n_range_gates = 4096
    range_gate_step = 25
    range_gate_start = 0
    tx = numpy.zeros([tx_length*n_ipp],dtype=numpy.complex64)
    echo = numpy.zeros([ipp_length*n_ipp],dtype=numpy.complex64)
    spectrum = numpy.zeros([tx_length*n_range_gates],dtype=numpy.float32)

    # simulate plasma line echoes
    for i in range(n_ipp):
        (echo_ipp,tx_ipp)=pl_example.get_simulated_ipp(L=25*10000,txlen_us=400,alt=0)
        tx[i*tx_length + numpy.arange(25*400)]=numpy.conj(tx_ipp)
        echo[i*ipp_length + numpy.arange(ipp_length)]=echo_ipp
            
    process_echoes(tx, echo, tx_length, ipp_length, n_ipp, spectrum, n_range_gates, range_gate_step, range_gate_start)

    spectrum.shape=(n_range_gates,tx_length)
    plt.plot(numpy.linspace(-12.5,12.5,num=tx_length),numpy.fft.fftshift(spectrum[0,:]))
    plt.show()
    smin = numpy.median(10.0*numpy.log10(spectrum))
    smax = numpy.max(10.0*numpy.log10(spectrum))
    plt.imshow(10.0*numpy.log10(spectrum[2000:0:-1,:]),aspect="auto",extent=[-12.5,12.5,0,1000],vmin=smin,vmax=100,cmap="nipy_spectral")
#    plt.pcolormesh(spectrum)
    plt.colorbar()
    plt.show()

