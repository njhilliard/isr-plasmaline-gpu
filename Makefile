CC = nvcc
CFLAGS =
SO_CFLAGS = --ptxas-options=-v --compiler-options '-fPIC' -shared
LDFLAGS = -lcufft

plasmaline: plasmaline.cu
	$(CC) $(CFLAGS) $(LDFLAGS) plasmaline.cu -o $@
libplasmaline.so: plasmaline.cu
	$(CC) $(SO_CFLAGS) $(LDFLAGS) plasmaline.cu -o $@
