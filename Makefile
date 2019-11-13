CC = nvcc
CFLAGS =
SO_CFLAGS = --compiler-options '-fPIC' -shared
LDFLAGS = -lcufft

plasmaline: plasmaline.cu
	$(CC) $(CFLAGS) $(LDFLAGS) plasmaline.cu -o $@
libplasmaline.so: plasmaline.cu
	$(CC) $(SO_CFLAGS) $(LDFLAGS) plasmaline.cu -o $@
