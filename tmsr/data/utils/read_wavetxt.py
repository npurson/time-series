import numpy as np


def read_wavetxt(path):
    """Reads frequence and wave series from .txt file.
    """
    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            if 'SampleFrequence' in line:
                freq = int(line[16:])
            elif 'DataInput' in line:
                series = np.array(line[10:].split(',')).astype(np.float64)
    return (freq, series)
