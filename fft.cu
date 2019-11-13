/* Project that takes signal and echo data and generates a power spectrum using NVIDIA
*  CUDA GPGPU programming.
*/

// includes, system
#include <stdio.h>
#include <time.h>

// includes CUDA
#include <cuComplex.h>
#include <cufft.h>

// parameters based on input array

#define IPP 250000 // 25 MHz sample rate 10 ms IPP
#define N_RANGE_GATES 4096 // 1 microsecond range gates
#define RANGE_GATE_STEP 25 // 1 microsecond
#define TX_LENGTH 16384 // transmit pulse length in 25 MHz sample rate
#define RANGE_START 500
#define NTHREADS 256

// forward declaration
extern "C" void process_echoes(float *tx, float *echo,
		    int tx_length, int ipp_length, int n_ipp,
		    float *spectrum,
		    int n_range_gates, int range_gate_step, int range_gate_start);

/* Kernel for complex conjugate multiplication */
__global__ void
complex_conj_mult(cufftComplex *tx, cufftComplex *echo, cufftComplex *batch, 
		  int tx_length, int n_range_gates, int range_gate_step, int range_gate_start)
{
    unsigned int block_num        = blockIdx.x + blockIdx.y * gridDim.x;
    unsigned int thread_num       = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int threads_in_block = blockDim.x * blockDim.y;
    unsigned int idx              = threads_in_block * block_num + thread_num;

    int i = idx / tx_length;
    int j = idx % tx_length;
    int ei = j + (i + range_gate_start) * range_gate_step;

    batch[idx] = cuCmulf(echo[ei], tx[j]);
}

/* Kernel for generating spectrum */
__global__ void
square_and_accumulate_sum(cufftComplex *z, float *spectrum)
{
    unsigned int block_num        = blockIdx.x + blockIdx.y * gridDim.x;
    unsigned int thread_num       = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int threads_in_block = blockDim.x * blockDim.y;
    unsigned int idx              = threads_in_block * block_num + thread_num;

    // z = x + yi
    // |z|^2 = x^2 + y^2  
    spectrum[idx] += cuCrealf(z[idx]) * cuCrealf(z[idx]) + cuCimagf(z[idx]) * cuCimagf(z[idx]);
}

/**********************/
/**** Program Main ****/
/**********************/

int main(int argc, char **argv) 
{
    int n_ipp=1000;

    // host memory allocation
    float *spectrum = (float *)malloc(TX_LENGTH * N_RANGE_GATES * sizeof(float));
    if (spectrum == NULL) {
        printf("Host error: Failed to allocate spectrum\n");
        exit(EXIT_FAILURE);
    }
    cufftComplex *z_tx = (cufftComplex *)malloc(n_ipp*TX_LENGTH * sizeof(cufftComplex));
    if (z_tx == NULL) {
        printf("Host error: Failed to allocate tx\n");
        exit(EXIT_FAILURE);
    }
    cufftComplex *z_echo = (cufftComplex *)malloc(n_ipp*IPP * sizeof(cufftComplex));
    if (z_echo == NULL) {
        printf("Host error: Failed to allocate echo\n");
        exit(EXIT_FAILURE);
    }
    process_echoes((float *)z_tx, (float *)z_echo, TX_LENGTH, IPP, n_ipp, 
		   spectrum, N_RANGE_GATES, RANGE_GATE_STEP, RANGE_START);
    return 0;
}

extern "C" void hello(float *cfloat, float *spectrum, int speclen)
{
  printf("hello world");
  int i;
  for(i=0 ; i<100 ; i++){
    printf("%1.2f\n",cfloat[i]);
  }
  for(i=0 ; i<speclen ; i++)
  {
    spectrum[i]=42.0;
  }
}
extern "C" void process_echoes(float *tx, float *echo,
		    int tx_length, int ipp_length, int n_ipp,
		    float *spectrum,
		    int n_range_gates, int range_gate_step, int range_gate_start)
{
  /**** Allocation/Initialization ****/
  // initializing pointers to device memory
  cufftComplex *d_z_tx;
  cufftComplex *d_z_echo;
  cufftComplex *d_z_batch;
  float *d_spectrum;

  cufftComplex *z_tx = (cufftComplex *)tx;
  cufftComplex *z_echo = (cufftComplex *)echo;
  
  clock_t start, end;
  start=clock();
  // allocating device memory to the above pointers
  if (cudaMalloc((void **) &d_z_tx, sizeof(cufftComplex)*tx_length) != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to allocate tx\n");
    exit(EXIT_FAILURE);
  }
  if (cudaMalloc((void **) &d_z_echo, sizeof(cufftComplex)*ipp_length) != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to allocate echo\n");
    exit(EXIT_FAILURE);
  }
  if (cudaMalloc((void **) &d_z_batch, sizeof(cufftComplex)*tx_length*n_range_gates) != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Failed to allocate batch\n");
    exit(EXIT_FAILURE);
  }
  if (cudaMalloc((void **) &d_spectrum, sizeof(cufftComplex)*tx_length*n_range_gates) != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Failed to allocate spectrum\n");
    exit(EXIT_FAILURE);
  }
  cudaMemset(d_spectrum, 0, sizeof(cufftComplex)*tx_length*n_range_gates);
  
  // setup kernel executiion parameters
  dim3 dimBlock(16, 16);
  dim3 dimGrid((n_range_gates + (dimBlock.x - 1)) / dimBlock.x,
	       (tx_length + (dimBlock.y - 1)) / dimBlock.y);

  // initializing in-place FFT plan
  cufftHandle plan;
  if (cufftPlan1d(&plan, tx_length, CUFFT_C2C, n_range_gates) != CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: Plan creation failed\n");
    exit(EXIT_FAILURE);
  }
  end=clock();
  double dt_malloc = ((double) (end-start))/CLOCKS_PER_SEC;
  printf("Time elapsed for malloc %1.2f s / 1000 echoes\n", dt_malloc );

  /**** Execution and timing ****/
  // running n_ipp iterations and timing the loop
  start=clock();
  for (int i = 0 ; i < n_ipp ; i++) {
    printf("ipp %d\n",i);
    // copy host memory to device
    if (cudaMemcpy(d_z_tx, &z_tx[i*tx_length], sizeof(cufftComplex)*tx_length, cudaMemcpyHostToDevice)
	!= cudaSuccess)
      {
	fprintf(stderr, "Cuda error: Memory copy failed, tx HtD\n");
	exit(EXIT_FAILURE);
      }
    if (cudaMemcpy(d_z_echo, &z_echo[i*ipp_length], sizeof(cufftComplex)*ipp_length, cudaMemcpyHostToDevice)
	!= cudaSuccess)
      {
	fprintf(stderr, "Cuda error: Memory copy failed, echo HtD\n");
	exit(EXIT_FAILURE);
      }
    
    //execution
    complex_conj_mult<<< dimGrid, dimBlock >>>(d_z_tx, d_z_echo, d_z_batch, tx_length, n_range_gates, range_gate_step, range_gate_start);
    if (cudaGetLastError() != cudaSuccess) {
      fprintf(stderr, "Cuda error: Failed to launch kernel\n");
      exit(EXIT_FAILURE);
    }
    
    if (cufftExecC2C(plan, (cufftComplex *)d_z_batch, (cufftComplex *)d_z_batch, CUFFT_FORWARD)
	!= CUFFT_SUCCESS)
      {
	fprintf(stderr, "CUFFT error: ExecC2C Forward failed\n");
	exit(EXIT_FAILURE);
      }
    
    square_and_accumulate_sum<<< dimGrid, dimBlock >>>(d_z_batch, d_spectrum);
    if (cudaGetLastError() != cudaSuccess) {
      fprintf(stderr, "Cuda error: Kernel failure, square_and_accumulate_sum\n");
      exit(EXIT_FAILURE);
    }
  }
  end=clock();
  double dt = ((double) (end-start))/CLOCKS_PER_SEC;
  printf("Time elapsed %1.2f s / 1000 echoes %1.2f speed ratio\n", dt, ((double)n_ipp*0.01)/dt );

/**** Obtaining results and clean up ****/

  // copying device resultant spectrum to host
  if (cudaMemcpy(spectrum, d_spectrum, sizeof(float)*n_range_gates*tx_length, cudaMemcpyDeviceToHost)
        != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Memory copy failed, spectrum DtH\n");
    exit(EXIT_FAILURE);
  }
  start=clock();
  // memory clean up
  if (cudaFree(d_z_tx) != cudaSuccess) {
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
  end=clock();
  double dt_free = ((double) (end-start))/CLOCKS_PER_SEC;
  printf("Time elapsed for free %1.2f s / 1000 echoes\n", dt_free );
}
