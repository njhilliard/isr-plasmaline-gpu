/*

This file reads in the binary from the defined INPUT, a binary python output array into a C
array and then runs an inplace FFT using the GPU through cufft.

****Still needs the main() segregated into supporting functions****
*/

#include <stdio.h>
#include <cuComplex.h>
#include <cufft.h>
#include <errno.h>
#include <time.h>
#include "cublas_v2.h"

//Parameters based on input array
#define IPP 250000 // 25 MHz sample rate 10 ms IPP
#define N_RANGE_GATES 4096 // 1 microsecond range gates, we want power of two
#define RANGE_GATE_STEP 25 // 1 microsecond
#define TX_LENGTH 16384 // transmit pulse length in 25 MHz sample rate (power of two)
#define RANGE_START 500
// 

/* Kernel for complex conjugate multiplication */
__global__ void
complex_conj_mult(cufftComplex *tx, cufftComplex *echo, cufftComplex *batch)
{
    unsigned int block_num        = blockIdx.x + blockIdx.y * gridDim.x;
    unsigned int thread_num       = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int threads_in_block = blockDim.x * blockDim.y;
    unsigned int idx              = threads_in_block * block_num + thread_num;

    int i = idx / TX_LENGTH;
    int j = idx % TX_LENGTH;
    int ei = j + (i + RANGE_START) * RANGE_GATE_STEP;

    batch[idx] = cuCmulf(echo[ei], tx[j]);
}


__global__ void square_and_accumulate_sum(cufftComplex *z, float *spectrum)
{
  unsigned int block_num        = blockIdx.x + blockIdx.y * gridDim.x;
  unsigned int thread_num       = threadIdx.x + threadIdx.y * blockDim.x;
  unsigned int threads_in_block = blockDim.x * blockDim.y;
  unsigned int idx              = threads_in_block * block_num + thread_num;

  spectrum[idx] += z[idx].x*z[idx].x + z[idx].y*z[idx].y;
}


int
main() 
{
/**********************************************************************/
/**** Complex multiplication of tx with echo along the range gates ****/
/**********************************************************************/

    // cufftComplex is single precision interleaved float
    cufftComplex *z_tx = (cufftComplex *)malloc(TX_LENGTH * sizeof(cufftComplex));
    cufftComplex *z_echo = (cufftComplex *)malloc(IPP * sizeof(cufftComplex));
    float *spectrum = (float *)malloc(TX_LENGTH*N_RANGE_GATES*sizeof(float));

    // initializing pointers to device memory
    cufftComplex *d_z_tx;
    cufftComplex *d_z_echo;
    cufftComplex *d_z_batch;

    // allocating device memory to the above pointers
    if (cudaMalloc((void **) &d_z_tx, sizeof(cufftComplex)*TX_LENGTH) != cudaSuccess) {
        fprintf(stderr, "Cuda error: Failed to allocate\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void **) &d_z_echo, sizeof(cufftComplex)*IPP) != cudaSuccess) {
        fprintf(stderr, "Cuda error: Failed to allocate\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void **) &d_z_batch, sizeof(cufftComplex)*TX_LENGTH*N_RANGE_GATES) != cudaSuccess)
    {
        fprintf(stderr, "Cuda error: Failed to allocate d_z_batch\n");
        exit(EXIT_FAILURE);
    }

    // setup execution parameters
    dim3 dimBlock(16,16);
    dim3 dimGrid(N_RANGE_GATES/16,TX_LENGTH/16);

/*************************************************/
/**** Complex multiplication ended, begin FFT ****/
/*************************************************/

    // initializing in-place FFT plan
    cufftHandle plan;
    if (cufftPlan1d(&plan, TX_LENGTH, CUFFT_C2C, N_RANGE_GATES) != CUFFT_SUCCESS) {
      fprintf(stderr, "CUFFT error: Plan creation failed\n");
      exit(EXIT_FAILURE);
    }

    /*// executing FFT
    if (cufftExecC2C(plan, (cufftComplex *)d_z_batch, (cufftComplex *)d_z_batch, CUFFT_FORWARD)
	    != CUFFT_SUCCESS)
    {
        fprintf(stderr, "CUFFT error: ExecC2C Forward failed\n");
        exit(EXIT_FAILURE);
    }

    // memory clean up
    if (cufftDestroy(plan) != CUFFT_SUCCESS) {
        fprintf(stderr, "Cuda error: Failed to destroy\n");
        exit(EXIT_FAILURE);
    }*/

/************************************************/
/**** In-place square; spectrum accumulation ****/
/************************************************/

    // initializing spectrum pointer
    float *d_spectrum;

    // allocating device memory
    if (cudaMalloc((void **) &d_spectrum, sizeof(float)*TX_LENGTH*N_RANGE_GATES) != cudaSuccess)
    {
        fprintf(stderr, "Cuda error: Failed to allocate d_spectrum\n");
        exit(EXIT_FAILURE);
    }


/**** timing the process ****/
    clock_t start, end;
    int n_reps = 1000;
    start=clock();
    for( int i=0 ; i<n_reps ; i++)
    {
      if (cudaMemcpy(d_z_tx, z_tx, sizeof(cufftComplex)*TX_LENGTH, cudaMemcpyHostToDevice)
	  != cudaSuccess)
      {
        fprintf(stderr, "Cuda error: Memory copy failed, HtD\n");
        exit(EXIT_FAILURE);
      }
      if (cudaMemcpy(d_z_echo, z_echo, sizeof(cufftComplex)*IPP, cudaMemcpyHostToDevice)
	  != cudaSuccess)
      {
        fprintf(stderr, "Cuda error: Memory copy failed, HtD\n");
        exit(EXIT_FAILURE);
      }
      // form tx*echo, assume tx is already conjugated!
      complex_conj_mult<<< dimGrid, dimBlock >>>(d_z_tx, d_z_echo, d_z_batch);
      if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "Cuda error: Failed to launch kernel\n");
        exit(EXIT_FAILURE);
      }
      if (cufftExecC2C(plan, (cufftComplex *)d_z_batch, (cufftComplex *)d_z_batch, CUFFT_FORWARD)
	      != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: ExecC2C Forward failed\n");
        exit(EXIT_FAILURE);
      }
    }
    // copy results back to device
    if (cudaMemcpy(spectrum, d_spectrum, sizeof(float)*TX_LENGTH*N_RANGE_GATES, cudaMemcpyDeviceToHost)
        != cudaSuccess)
    {
        fprintf(stderr, "Cuda error: Memory copy failed, HtD\n");
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();
    end=clock();
    cufftDestroy(plan);
    cublasDestroy(handle);
    cudaFree(d_z_tx);
    cudaFree(d_z_echo);
    cudaFree(d_z_batch);
    cudaFree(d_spectrum);
    free(z_tx);
    free(z_echo);
    double dt = ((double) (end-start))/CLOCKS_PER_SEC;
    printf("Time elapsed %1.2f s / 1000 echoes %1.2f speed ratio\n", dt, ((double)n_reps*0.01)/dt );
    
    return 0;
}


//void process_echoes(float **tx, float **echo, int n_ipp, float *spectrum)
//{
//}

