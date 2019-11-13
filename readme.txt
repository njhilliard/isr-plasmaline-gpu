--------ISR Plasma Line GPU Analysis--------

General Description
-------------------

This program uses CUDA C/C++ API, run on an NVIDIA GPU, to take input
transmit pulse, return echo, and range gate specifications to extract a power
spectrum of the plasma line echoes. Parallelization through general purpose
GPU computing conveys order-of-magnitude speedup over traditional CPU
analysis.

Files Included
--------------

plasmaline.cu - CUDA C code including GPU kernels and primary function
                process_echoes().

plasmaline.h  - CUDA C header file

test_gpu      - Python test file. Wraps and runs the process_echoes()
                function with simulated data to test on the current
                system.

Makefile      - Provides make information for standalone plasmaline
                compilation or a library compilation. May need to be
                updated with library or include paths.

System Requirements
-------------------

NVIDIA GPU of compute capability at least 2.0
NVIDIA Toolkit Installed - Recommended Version 7.0
    nvcc  - CUDA compiler
    cufft - CUDA FFT library
Python 2
    NumPy
    Optional - Matplotlib

Instructions
------------

Update include and library paths in the Makefile if needed. Then:

make libplasmaline.so
python test_gpu.py

If the system passes the tests, then your system is ready for
data analysis using the CUDA code.


Additional Information
----------------------

Built on a system utilizing NVIDIA c2050 GPU, achieved computation
speed equivalent to 21% of realtime data acquisition speed.
























