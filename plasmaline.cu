/* Project that takes signal and echo ISR data and generates a power spectrum using NVIDIA
** CUDA GPGPU programming. Function main() tests empty data sets to check if the kernels 
** (GPU-based algorithms), allocations, and data transfers are working properly.
**
** The central power of this program is from the kernels below tied together nicely
** in the process_echoes() function.
**
** GPU timing and progression is turned off by default, however, code is left commented if wanted.
*/

// header file for the plasmaline project
#include "plasmaline.h"

/* Kernel for complex conjugate multiplication
**  This algorithm runs on the GPU, taking input transmit pulse conjugate (tx)
**  and echo, cufftComplex types, casted from cuFloatComplex types, which are Float2
**  types (single precision), and outputs the same type (batch). Batch has dimensions
**  of [n_range_gates] by [tx_length], specified by kernel dimensions in function
**  process_echoes() below main(). To aid processing speed, dimensions should be
**  powers of two and evenly divisible by 16. If this is not the case, it is suggested
**  to pad the data until it is. Otherwise, a catching if statement will be needed
**  where ei cannot outrun echo and idx cannot outrun the elements in batch.
*/
__global__ void complex_mult(cufftComplex *tx_conj, cufftComplex *echo, cufftComplex *batch, 
		                     int tx_length, int n_range_gates,
                             int range_gate_step, int range_gate_start)
{
    // this provides mapping from i/o to 1D GPU threading
    unsigned int block_num        = blockIdx.x + blockIdx.y * gridDim.x;
    unsigned int thread_num       = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int threads_in_block = blockDim.x * blockDim.y;
    unsigned int idx              = threads_in_block * block_num + thread_num;

    // mapping for range-gate-stepping through echo, multiplied into signal conjugate
    //  i provides mapping to first dimension in batch
    //  j provides mapping to second dimension in batch (== idx % tx_length)
    //  ei provides index of the echo element needed for the range gate
    int i = idx / tx_length;
    int j = idx - (i * tx_length);
    int ei = j + (i + range_gate_start) * range_gate_step;

    // complex multiplication (single precision float) of echo and signal,
    // placed into a linearized 2D array, each row corresponding to a range bin
    batch[idx] = cuCmulf(echo[ei], tx_conj[j]);
}

/* Kernel for generating spectrum
**  This algorithm will take the ouput cufftComplex from an FFT done below and
**  run an in-place square. This will be added to the single precision float
**  spectrum of the same array dimensions. Spectrum will be gathering a sum of
**  this operation as transmit signal (tx) changes in subsequent time steps.
**  The same guidelines and mapping in the first kernel apply here as well.
*/
__global__ void square_and_accumulate_sum(cufftComplex *z, float *spectrum)
{
    unsigned int block_num        = blockIdx.x + blockIdx.y * gridDim.x;
    unsigned int thread_num       = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int threads_in_block = blockDim.x * blockDim.y;
    unsigned int idx              = threads_in_block * block_num + thread_num;

    // spectrum gathers squares from the interleaved complex z
    spectrum[idx] += z[idx].x * z[idx].x + z[idx].y * z[idx].y;
}

/**********************/
/**** Program Main ****/
/**********************/

/* As noted at the top of the code, the main() here runs a test run over 100
** inter-pulse periods (ipp), if this fails, an error should be thrown and the
** program aborted.
*/

int main(int argc, char **argv) 
{
    // some example parameters
    int n_ipp = 100;
    int ipp = 250000; // 25 MHz sample rate 10 ms ipp
    int n_range_gates = 4096; // 1 microsecond range gates
    int range_gate_step = 25; // 1 microsecond
    int tx_length = 16384; // transmit pulse length in 25 MHz sample rate
    int range_start = 500;

    // host memory allocation, not needed in process_echoes() since these should
    //  be allocated in the code which hosts this C code
    float *spectrum = (float *)malloc(tx_length * n_range_gates * sizeof(float));
    if (spectrum == NULL) {
        printf("Host error: Failed to allocate spectrum\n");
        exit(EXIT_FAILURE);
    }
    // z_ denotes complex, tx here holds n_ipp transmit signals
    cufftComplex *z_tx_conj = (cufftComplex *)malloc(n_ipp * tx_length * sizeof(cufftComplex));
    if (z_tx_conj == NULL) {
        printf("Host error: Failed to allocate tx\n");
        exit(EXIT_FAILURE);
    }
    // echo also holds n_ipp return signals
    cufftComplex *z_echo = (cufftComplex *)malloc(n_ipp * ipp * sizeof(cufftComplex));
    if (z_echo == NULL) {
        printf("Host error: Failed to allocate echo\n");
        exit(EXIT_FAILURE);
    }
    process_echoes((float *)z_tx_conj, (float *)z_echo, tx_length, ipp, n_ipp,
                   spectrum, n_range_gates, range_gate_step, range_start);
    free(z_tx_conj);
    free(z_echo);
    free(spectrum);
    return 0;
}

/* This is the primary function in this program, meant to be embedded into other
**  programs. The transmit signal, tx, should be complex conjugated prior to use here.
**  The float types for tx and echo are useful when this function is embedded; the extern
**  "C" is also here for that purpose. Process_echoes sets up and runs the kernels on
**  the GPU, complex_mult, a 1D FFT, and a spectrum accumulation. Host spectrum is not
**  freed, so as to be taken and analyzed.
*/
extern "C" void process_echoes(float *tx_conj, float *echo,
		                       int tx_length, int ipp_length, int n_ipp,
		                       float *spectrum,
		                       int n_range_gates, int range_gate_step, int range_gate_start)
{
 /**** Allocation/Initialization ****/
    printf("\nProcessing with GPU\n");

    // initializing pointers to device (GPU) memory, denoted with "d_"
    cufftComplex *d_z_tx_conj;
    cufftComplex *d_z_echo;
    cufftComplex *d_z_batch;
    float *d_spectrum;

    // necessary casts for host (CPU) data, complex denoted with "z_"
    cufftComplex *z_tx_conj = (cufftComplex *)tx_conj;
    cufftComplex *z_echo = (cufftComplex *)echo;
  
    // allocating device memory to the above pointers
    // the signal and echo here are only one row of the CPU data (one time step)
    if (cudaMalloc((void **) &d_z_tx_conj, sizeof(cufftComplex) * tx_length) != cudaSuccess)
    {
        fprintf(stderr, "Cuda error: Failed to allocate tx\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void **) &d_z_echo, sizeof(cufftComplex) * ipp_length) != cudaSuccess)
    {
        fprintf(stderr, "Cuda error: Failed to allocate echo\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void **) &d_z_batch, sizeof(cufftComplex) * tx_length * n_range_gates)
       != cudaSuccess)
    {
        fprintf(stderr, "Cuda error: Failed to allocate batch\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void **) &d_spectrum, sizeof(float) * tx_length * n_range_gates)
       != cudaSuccess)
    {
        fprintf(stderr, "Cuda error: Failed to allocate spectrum\n");
        exit(EXIT_FAILURE);
    }

    // ensure empty device spectrum
    if (cudaMemset(d_spectrum, 0, sizeof(float) * tx_length * n_range_gates) != cudaSuccess)
    {
        fprintf(stderr, "Cuda error: Failed to zero device spectrum\n");
        exit(EXIT_FAILURE);
    }
  
    // setup custom kernel execution parameters

    // this sets how the GPU divides up the data to be run on different processors
    dim3 dimBlock(16, 16); // a 16x16 block is chosen - portable and efficient

    // this grid setup assumes n_range_gates and tx_length both be divisible by 
    //  block dimensions, otherwise something like the following will be necessary:
    //  (n_range_gates + (dimBlock.x - 1)) / dimBlock.x, and a catch if() in the kernel
    dim3 dimGrid( (n_range_gates / dimBlock.x), (tx_length / dimBlock.y) );


    // initializing 1D FFT plan, this will tell cufft execution how to operate
    // cufft is well optimized and will run with different parameters than above
    cufftHandle plan;
    if (cufftPlan1d(&plan, tx_length, CUFFT_C2C, n_range_gates) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: Plan creation failed\n");
        exit(EXIT_FAILURE);
    }

 /**** Execution and timing ****/

    // execution timing, done with CPU
    clock_t start, end;
    start=clock();

    // execution of the prepared kernels n_ipp times
    for (int i = 0 ; i < n_ipp ; i++)
    {
        //printf("ipp %d\n",i); // view computation progress
        fprintf(stderr, ".");

        // copying n_ipp'th row of host data into device
        if (cudaMemcpy(d_z_tx_conj, &z_tx_conj[i * tx_length], sizeof(cufftComplex) * tx_length,
            cudaMemcpyHostToDevice) != cudaSuccess)
        {
	        fprintf(stderr, "Cuda error: Memory copy failed, tx HtD\n");
	        exit(EXIT_FAILURE);
        }
        if (cudaMemcpy(d_z_echo, &z_echo[i * ipp_length], sizeof(cufftComplex) * ipp_length,
            cudaMemcpyHostToDevice) != cudaSuccess)
        {
	        fprintf(stderr, "Cuda error: Memory copy failed, echo HtD\n");
	        exit(EXIT_FAILURE);
        }
        
        // complex_mult kernel execution (<<<>>>)
        complex_mult<<< dimGrid, dimBlock >>>(d_z_tx_conj, d_z_echo, d_z_batch,
                                              tx_length, n_range_gates, range_gate_step,
                                              range_gate_start);
        if (cudaGetLastError() != cudaSuccess) {
            fprintf(stderr, "Cuda error: Failed to launch kernel\n");
            exit(EXIT_FAILURE);
        }

        // cufft kernel execution
        if (cufftExecC2C(plan, (cufftComplex *)d_z_batch, (cufftComplex *)d_z_batch, CUFFT_FORWARD)
	       != CUFFT_SUCCESS)
        {
	        fprintf(stderr, "CUFFT error: ExecC2C Forward failed\n");
	        exit(EXIT_FAILURE);
        }

        // spectrum accumulation kernel execution
        square_and_accumulate_sum<<< dimGrid, dimBlock >>>(d_z_batch, d_spectrum);
        if (cudaGetLastError() != cudaSuccess) {
            fprintf(stderr, "Cuda error: Kernel failure, square_and_accumulate_sum\n");
            exit(EXIT_FAILURE);
        }
    }

    // execution timing and comparison to real-time data collection speed
    end=clock();
    double dt = ((double) (end-start))/CLOCKS_PER_SEC;
    printf("Time elapsed %1.3f s / %d echoes %1.3f speed ratio\n", dt, n_ipp,
           ((double)n_ipp * 0.01) / dt);

 /**** Obtaining results and clean up ****/

    // copying device resultant spectrum to host, now able to be manipulated
    if (cudaMemcpy(spectrum, d_spectrum, sizeof(float) * n_range_gates * tx_length,
        cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        fprintf(stderr, "Cuda error: Memory copy failed, spectrum DtH\n");
        exit(EXIT_FAILURE);
    }

    // memory clean up
    if (cudaFree(d_z_tx_conj) != cudaSuccess) {
        fprintf(stderr, "Cuda error: Failed to free tx\n");
        exit(EXIT_FAILURE);
    }
    if (cudaFree(d_z_echo) != cudaSuccess) {
        fprintf(stderr, "Cuda error: Failed to free echo\n");
        exit(EXIT_FAILURE);
    }
    if (cudaFree(d_z_batch) != cudaSuccess) {
        fprintf(stderr, "Cuda error: Failed to free batch\n");
        exit(EXIT_FAILURE);
    }
    if (cudaFree(d_spectrum) != cudaSuccess) {
        fprintf(stderr, "Cuda error: Failed to free spectrum\n");
        exit(EXIT_FAILURE);
    }
    if (cufftDestroy(plan) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: Failed to destroy plan\n");
        exit(EXIT_FAILURE);
    }

    printf("\nGPU processing complete.\n\n");
    // host data needs to be cleaned by host program
}
