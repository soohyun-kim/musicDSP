import numpy as np
import matplotlib.pyplot as mp
from scipy.io import wavfile

def crossCorr(x, y):
    return np.correlate(x, y);

def loadSoundFile(fileName):
    rate, data = wavfile.read(fileName);
    return data[:, 0].astype(float);

def findSnarePosition(snareFilename, drumloopFilename):
    snare = loadSoundFile(snareFilename);
    drumloop = loadSoundFile(drumloopFilename);
    corr = crossCorr(snare, drumloop);
    f = open('results/02-snareLocation.txt', 'w');
    f.write(np.array2string(np.where(corr>1.5e11)[0]));
    f.close();
    return

def main():
    snare = loadSoundFile('snare.wav');
    drumloop = loadSoundFile('drum_loop.wav');
    corr = crossCorr(snare, drumloop);
    
    mp.plot(np.arange(len(corr)), corr);
    mp.title('correlation of snare vs. drum loop');
    mp.xlabel('index shift');
    mp.ylabel('correlation');
    #mp.show();
    mp.savefig('results/01-correlation.png');

    #findSnarePosition('snare.wav', 'drum_loop.wav');

    return

if __name__ == '__main__':
    main()

# python code is absolutely dark magic
# written by Soohyun Kim
