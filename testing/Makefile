CC = nvcc
CFLAGS = -I/usr/include/ -L/usr/local/cuda-7.0/lib64/ 
SO_CFLAGS = --ptxas-options=-v --compiler-options '-fPIC' -I/usr/include/ -L/usr/local/cuda-7.0/lib64/  -shared
LDFLAGS = -lcufft
SRC_CU = fft.cu
SRC = $(SRC_CU)
OBJ = $(SRC_CU:.cu=.o)

plasmaline: plasmaline.cu
	$(CC) $(CFLAGS) $(LDFLAGS) plasmaline.cu -o $@
libplasmaline.so: plasmaline.cu
	$(CC) $(SO_CFLAGS) $(LDFLAGS) plasmaline.cu -o $@
