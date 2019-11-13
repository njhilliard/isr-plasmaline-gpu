#!/usr/bin/python

#This iteration works on all elements when block dims = 8x8
# When other dimensions, there is a mismatch in S[1999]

#Added complex conjugatin on tx
#Beginning adding fft in 6_* and 7_*
#Python FFT implementation is very abstracted and indirect, I'm going to 
#  try to build a C implementation reading the output S from this file.

import sys
import numpy as np
import example as ex
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

(echo,tx) = ex.get_simulated_ipp()
ranges = np.arange(4000,6000).astype(np.int32)
sr = ex.sr

def tx_cmplxMult():
    """GPU implementation of signal multiplied into the echo along ranges specified.
    """
    #pull simulated data and get info about it
    S = np.zeros([len(ranges),len(tx)],dtype=np.complex64)
    nCols = len(S[0])
    nRows = len(S)

    #allocating gpu memory
    echo_gpu   = cuda.mem_alloc(echo.size * echo.dtype.itemsize)
    tx_gpu     = cuda.mem_alloc(tx.size * tx.dtype.itemsize)
    ranges_gpu = cuda.mem_alloc(ranges.size * ranges.dtype.itemsize)
    S_gpu      = cuda.mem_alloc(S.size * S.dtype.itemsize)

    #copying arrays to gpu memory
    cuda.memcpy_htod(echo_gpu, echo)
    cuda.memcpy_htod(tx_gpu, tx)
    cuda.memcpy_htod(ranges_gpu, ranges)
    cuda.memcpy_htod(S_gpu, S)


    block_dim_x = 8                                    #thread number is product of block dims,
    block_dim_y = 8                                    # want a multiple of 32 (warp multiple)
    blocks_x = np.ceil(len(ranges)/block_dim_x).astype(np.int32).item()
    blocks_y = np.ceil(len(tx)/block_dim_y).astype(np.int32).item()
    total = block_dim_x * block_dim_y * blocks_x * blocks_y
    block = (block_dim_x,block_dim_y,1)
    grid = (blocks_x, blocks_y)

    kernel_code="""
    #include <cuComplex.h>

    __global__ void complex_mult(
      cuFloatComplex *tx, cuFloatComplex *echo, cuFloatComplex *result, 
      int *ranges, int sr,
      int nCols, int nRows, int total)
    {
        unsigned int block_num        = blockIdx.x + blockIdx.y * gridDim.x;
        unsigned int thread_num       = threadIdx.x + threadIdx.y * blockDim.x;
        unsigned int threads_in_block = blockDim.x * blockDim.y;
        unsigned int idx              = threads_in_block * block_num + thread_num;

        //aligning the i,j to idx

        int i = idx / nCols;
        int j = idx % nCols;

        result[idx] = cuCmulf(echo[j+ranges[i]*sr], cuConjf(tx[j]));
    }
    """

    #includes directory of where cuComplex.h is located
    mod = SourceModule(kernel_code, include_dirs=['/usr/local/cuda-7.0/include/'])
    complex_mult = mod.get_function("complex_mult")

    arg_types = ("P", "P", "P",
                 "P", np.int32,
                 np.int32, np.int32, np.int32)
    args      = (tx_gpu, echo_gpu, S_gpu,
                 ranges_gpu, sr,
                 nCols, nRows, total)

    complex_mult.prepare(arg_types)
    complex_mult.prepared_call(grid, block, *args)
    cuda.memcpy_dtoh(S, S_gpu)

    #complex_mult(cuda.In(tx), cuda.In(echo), cuda.Out(S),
                #cuda.In(ranges), np.int32(sr),
                #np.int32(nCols), np.int32(nRows), np.int32(total),
                #block=(block_dim_x,block_dim_y,1),
                #grid=(blocks_x,blocks_y))
    return(S)


if __name__ == "__main__":

    S = tx_cmplxMult()
    try: 
        something = sys.argv[1]
    except IndexError:
        print "Failed to specify parameter."
        print "  Usage: $python filename option."
        print "  Allowed options: compare, output, print"
        sys.exit(0)
    if sys.argv[1] == "compare":
    ### compares CPU vs GPU calcs ###
        compare = np.zeros_like(S)
        txidx   = np.arange(len(tx))

        for ri,r in enumerate(ranges):
                compare[ri,:] = echo[txidx+r*sr]*np.conj(tx)

        diff = np.subtract(S, compare)
        print diff

    elif sys.argv[1] == "output":
    ### write S to binary output file ###
        with open("fft_input.dat",'wb') as f:
            f.write(S)
        with open("fft_cpu.dat", 'wb') as f:
            f.write(np.fft.fft(S))
            print np.fft.fft(S)
    elif sys.argv[1] == "print":
        print S

