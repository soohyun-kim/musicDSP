import numpy as np
import matplotlib.pyplot as mp
import scipy.signal as sp
from scipy.io import wavfile
import time as timer

# The length of the convolution of two signals is len(x) + len(h) - 1, so 
# if x is of length 200 and h 100, the convolution will be 299 samples long.
def myTimeConv(x, h):
    print('myTimeConv started, convolution length:')
    h = np.flip(h);
    x_padded = np.concatenate((np.zeros(len(h)-1), x, np.zeros(len(h)-1)));
    y = np.zeros(len(x)+len(h)-1);
    print(len(y))
    print('progress:')
    for i in range(len(y)):
        h_padded = np.zeros(2*(len(h)-1)+len(x));
        h_padded[i:i+len(h)] = h;
        y[i] = np.sum(np.multiply(x_padded, h_padded));
        
        # For observing progress
        if (i%10000)==0:
            print(i)
    return y;

def CompareConv(x, h):
    time = np.zeros(2);

    start = timer.time();
    y_sp = sp.convolve(x.astype(float), h.astype(float));
    stop = timer.time();
    time[0] = stop - start;

    start = timer.time();
    y_my = myTimeConv(x, h);
    stop = timer.time();
    time[1] = stop - start;

    # I do not find manual implementation of the below functions to be necessary;
    # their mechanisms are irrelevant to the topic at hand.
    m = np.mean(np.subtract(y_sp, y_my));
    mabs = np.mean(np.absolute(np.subtract(y_sp, y_my)));
    stdev = np.std(np.subtract(y_sp, y_my));

    return (m, mabs, stdev, time);

def main():
    x = np.ones(200);
    h = np.concatenate([np.linspace(0., 1., 26), np.linspace(1., 0., 26)]);
    h = np.delete(h, 26);
    y_time = myTimeConv(x, h);
    mp.plot(np.arange(len(y_time)), y_time);
    mp.title('convolution result of DC and triangular impulse');
    mp.ylabel('value');
    mp.xlabel('sample');
    mp.savefig('results/01-convolution.png');

    results = CompareConv(wavfile.read('piano.wav')[1], wavfile.read('impulse-response.wav')[1]);
    f = open('results/02-comparison.txt', 'w');
    f.write('m: ');
    f.write(str(results[0]));
    f.write('\nmabs: ');
    f.write(str(results[1]));
    f.write('\nstdev: ');
    f.write(str(results[2]));
    f.write('\ntime elapsed in s [scipy, myTimeConv]: ');
    f.write(str(results[3]));
    f.close();
    return

if __name__ == '__main__':
    main()

# python code is absolutely dark magic
# written by Soohyun Kim
