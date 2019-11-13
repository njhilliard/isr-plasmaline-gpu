#ifndef PLASMA_LINE
#define PLASMA_LINE

// system includes
#include <stdio.h>
#include <time.h>

// CUDA includes
#include <cuComplex.h>
#include <cufft.h>

extern "C" void process_echoes(float *tx, float *echo,
                               int tx_length, int ipp_length, int n_ipp,
                               float *spectrum,
                               int n_range_gates, int range_gate_step, int range_gate_start);

__global__ void complex_mult(cufftComplex *tx, cufftComplex *echo, cufftComplex *batch,
                             int tx_length, int n_range_gates,
                             int range_gate_step, int range_gate_start);

__global__ void square_and_accumulate_sum(cufftComplex *z, float *spectrum);

#endif


