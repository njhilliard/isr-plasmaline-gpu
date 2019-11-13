#!/usr/bin/python

import numpy as np
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule

ary_in = np.ones((16,32), dtype=np.int32)
nRows = len(ary_in)
nCols = len(ary_in[0])
print ary_in
ary_out = np.zeros((16,32), dtype=np.int32)
stride = 4

block_dim_x = 4
block_dim_y = 4
blocks_x    = 1
blocks_y    = 8
limit = block_dim_x * block_dim_y * blocks_x * blocks_y

mod = SourceModule("""
__global__ void indexing_order(int *ary_in, int *ary_out, int nCols, int stride)
{
  unsigned int block_num        = blockIdx.x + blockIdx.y * gridDim.x;
  unsigned int thread_num       = threadIdx.x + threadIdx.y * blockDim.x;
  unsigned int threads_in_block = blockDim.x * blockDim.y;
  unsigned int idx              = stride * (threads_in_block * block_num) + thread_num;

  int i = idx / nCols;
  int j = idx % nCols;
  int ei = j + (i + 500) * 25;

  for (int k = 0; k < stride; k++) {
    int l = k * threads_in_block;
    ary_out[idx + l] = block_num;
    __syncthreads();
  }
}
""")

indexing_order = mod.get_function("indexing_order")

indexing_order(drv.In(ary_in), drv.Out(ary_out), np.int32(nCols), np.int32(stride),
               block=(block_dim_x,block_dim_y,1),
               grid=(blocks_x,blocks_y))
print ary_out
