#!/usr/bin/env python

import numpy
import matplotlib.pyplot as plt

sr = 25
def get_simulated_ipp(L=sr*10000,alt=sr*5000,freq=5e6,txlen_us=1000):
    """Builds example sent and return signals.
    """
    txlen=txlen_us*sr
    echo = numpy.zeros(L,dtype=numpy.complex64) + numpy.array((numpy.random.randn(L) + numpy.random.randn(L)*1j),dtype=numpy.complex64)
    tx = numpy.zeros(txlen,dtype=numpy.complex64)
    idx = numpy.arange(txlen_us)*sr
    tx_bits = numpy.sign(numpy.random.randn(txlen_us))

    for i in range(sr):
        tx[idx+i] = tx_bits

    tvec = numpy.arange(txlen)/(float(sr)*1e6)

    echo[alt:(alt+txlen)]= echo[alt:(alt+txlen)] + tx*numpy.exp(1j*2.0*numpy.pi*freq*tvec)

    return((echo,tx))


if __name__ == "__main__":

    n_ipps=3
    ranges = numpy.arange(4000,6000)
    txlen_us = 1000
    S = numpy.zeros([len(ranges),txlen_us*sr],dtype=numpy.float32)
    txidx = numpy.arange(txlen_us*sr)
    for i in range(n_ipps):
        print(".")
        (echo,tx)=get_simulated_ipp()
        for ri,r in enumerate(ranges):
            S[ri,:]+=numpy.fft.fftshift(numpy.abs(numpy.fft.fft(echo[txidx+r*sr]*numpy.conj(tx)))**2.0)
    freqs = numpy.linspace(-12.5,12.5,num=txlen_us*sr)
    plt.imshow(10.0*numpy.log10(S[::-1,:]),aspect="auto",extent=[-12.5,12.5,4000,6000],vmin=60,vmax=80)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Range ($\mu s$)")
    plt.colorbar()
    plt.show()
            
     
