CC = nvcc
CFLAGS = -I/usr/include/ -L/usr/local/cuda-7.0/lib64/ 
SO_CFLAGS = --ptxas-options=-v --compiler-options '-fPIC' -I/usr/include/ -L/usr/local/cuda-7.0/lib64/  -shared
LDFLAGS = -lcufft -lcublas
SRC_CU = fft.cu
SRC = $(SRC_CU)
OBJ = $(SRC_CU:.cu=.o)

fft: fft.cu
	$(CC) $(CFLAGS) $(LDFLAGS) fft.cu -o $@
libplasmaline.so: fft.cu
	$(CC) $(SO_CFLAGS) $(LDFLAGS) fft.cu -o $@
