import numpy as np
import numpy.fft as npft
import matplotlib.pyplot as mp
import scipy.signal as sp
from scipy.io import wavfile

def generateSinusoidal(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians):
    input_vector = np.linspace(0, length_secs*sampling_rate_Hz*(frequency_Hz/sampling_rate_Hz), int(length_secs*sampling_rate_Hz));
    input_vector = np.add(phase_radians, np.multiply(2*np.pi, input_vector));
    x = np.multiply(amplitude, np.sin(input_vector));
    t = np.linspace(0, length_secs, int(sampling_rate_Hz*length_secs));
    return x, t

def generateSquare(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians):
    x, t = generateSinusoidal(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians);
    iter_n = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19]);
    for i in iter_n:
        harmonic, garb = generateSinusoidal(amplitude/i, sampling_rate_Hz, i*frequency_Hz, length_secs, phase_radians);
        x = np.add(x,  harmonic);
    x = np.multiply(4/np.pi, x)
    return x, t

def computeSpectrum(x, sample_rate_Hz, window_type='rect'):
    f = np.linspace(0.0, sample_rate_Hz, len(x));
    if window_type == 'hann':
        x = np.multiply(x, np.hanning(len(x)));
    if window_type == 'rect':
        x = np.multiply(x, np.ones((1, len(x))).flatten());
    y = npft.fft(x);
    XAbs = np.absolute(y);
    XPhase = np.angle(y);
    XRe = np.real(y);
    XIm = np.imag(y);
    return f, XAbs, XPhase, XRe, XIm

def main():
    x1, t = generateSinusoidal(1.0, 44100, 400, 0.5, np.pi/2);
    mp.plot(t[0:221], x1[0:221]);
    mp.ylabel('waveform output')
    mp.xlabel('time in s');
    mp.title('sinusoid, amp=1.0, Fs=44.1kHz, f=400Hz, theta=pi/2');
    mp.savefig('results\p1-plot.png');

    x2, t = generateSquare(1.0, 44100, 400, 0.5, 0);
    mp.clf();
    mp.plot(t[0:221], x2[0:221]);
    mp.ylabel('waveform output')
    mp.xlabel('time in s');
    mp.title('square, amp=1.0, Fs=44.1kHz, f=400Hz, theta=0');
    mp.savefig('results\p2-plot.png');
    
    mp.clf();
    mp.subplot(1, 2, 1);
    f, mag, phase, real, imag = computeSpectrum(x1, 44100);
    mp.plot(f[0:250], np.divide(mag, 500)[0:250])
    mp.plot(f[0:250], phase[0:250])
    mp.ylabel('mag/500 (blue), phase in radians (orange)')
    mp.xlabel('frequency in Hz');
    mp.title('dft_cos');
    mp.subplot(1, 2, 2);
    f, mag, phase, real, imag = computeSpectrum(x2, 44100);
    mp.plot(f[0:4500], np.divide(mag, 500)[0:4500])
    mp.plot(f[0:4500], phase[0:4500])
    mp.xlabel('frequency in Hz');
    mp.title('dft_square');
    mp.savefig('results\p3-plot.png');
    
    mp.clf();
    f, mag_h, phase, real, imag = computeSpectrum(x2, 44100, 'hann');
    mp.plot(f[0:4500], mag_h[0:4500]);
    mp.ylabel('magnitude')
    mp.xlabel('frequency in Hz');
    mp.title('square wave, hann window used');
    mp.savefig('results\p4-plot_hann.png');
    
    mp.clf();
    f, mag_r, phase, real, imag = computeSpectrum(x2, 44100, 'rect');
    mp.plot(f[0:4500], mag_r[0:4500]);
    mp.ylabel('magnitude')
    mp.xlabel('frequency in Hz');
    mp.title('square wave, rect window used');
    mp.savefig('results\p4-plot_rect.png');
    
    return

if __name__ == '__main__':
    main()

# python code is absolutely dark magic
# written by Soohyun Kim
