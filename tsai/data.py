import os.path as osp
import numpy as np


def sample(input: np.ndarray, src_freq, dst_freq, n_samples=None):
    """
    Samples input series and scales to specified length.
    Suggested `dst_freq` should at least be 2.5MHz.

    Distribution of input sample frequency:
        5e5:  8.4%
        1e6: 25.9%
        2e6: 65.6%
    Distribution of input series length (sampled to 2MHz):
        5e5: 24k
        1e6: 2k, 2k5, 6k
        2e6: 2k4, 2k6, 3k, 3k2, 3k6
    """
    if not n_samples:
        n_samples = int(2400 * dst_freq / 2e6)

    assert src_freq >= dst_freq
    output = input[::int(src_freq / dst_freq)]
    # from scipy.interpolate import interp1d
    # f = interp1d(list(range(len(input))), input, kind='cubic')
    # output = f(np.linspace(0, len(input) - 1, num=int(len(input) * dst_freq / src_freq)))

    if len(output) > n_samples:
        preamble = np.where(abs(output) > 40)[0]
        preamble = preamble[0] if len(preamble) else 0  # no element > 40
        preamble200 = int(200 * dst_freq / 2e6)  # 200 preamble for 2MHz sample frequency
        if preamble > preamble200:
            preamble = min(preamble - preamble200, len(output) - n_samples)
            output = output[preamble:]
        if len(output) > n_samples:
            output = output[:n_samples]
    if len(output) < n_samples:
        output = np.concatenate((output, np.zeros(n_samples - len(output))))
    return output


def read_wavetxt(path):
    """Reads frequency and wave series from .txt file.
    """
    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            if 'SampleFrequence' in line or 'SampleFrequency' in line or 'sampleRate' in line:
                freq = int(line.split('=')[-1])
            elif 'DataInput' in line or 'CurrentDataInput' in line or 'waveData' in line:
                inputs = list(filter(lambda x: x != '', line.split('=')[-1].split(',')))
                series = np.array(inputs).astype(np.float64)
    return (freq, series)


class TSData(object):

    def __init__(self, data_root='/home/jhy/repos/timeseries/data/nari0602', n_samples=5e5):
        self.CLASSES = (
            '雷击', '反击', '绕击', '外破', '山火', '施工碰线', '异物短路', '冰害',
            '冰闪', '覆冰过载', '脱冰跳跃', '舞动', '风偏', '鸟害', '污闪', '其他',
            '非雷击'
        )
        self.n_samples = n_samples
        with open(osp.join(data_root, 'train.txt')) as f:
            self.train_data = [line.strip().split(' ') for line in f.readlines()]
        with open(osp.join(data_root, 'val.txt')) as f:
            self.test_data = [line.strip().split(' ') for line in f.readlines()]

        # self.cls_map = list(range(len(self.CLASSES)))
        # self.cls_map = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1]
        # self.cls_map = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, None, 1]
        # self.cls_map = [0, 0, 0, 15, 15, 15, 6, 15, 15, 15, 15, 15, 12, 13, 15, 15, 15]
        self.cls_map = [0, 0, 0, None, None, None, 6, None, None, None, None, None, 12, 13, None, None, None]
        self.classes = [self.CLASSES[c] for c in set(self.cls_map) if c is not None]

    def data_pipeline(self, path, n_samples):
        freq, x = read_wavetxt(path)
        x = sample(x, freq, n_samples)
        x = x / 5000 if np.max(x) < 5000 else x / np.max(x)
        # dft = DiscreteFourierTransform()
        # xf = dft.fit_transform(x.reshape(1, -1))[0]
        # x = np.vstack((x, xf / np.max(xf)))
        return x

    def __call__(self):
        def extract_data(data):
            X = []; Y = []
            for path, cls in data:
                x = self.data_pipeline(path, n_samples=self.n_samples)
                y = self.CLASSES.index(cls)
                y = self.cls_map[y]
                if y is None:
                    continue
                X.append(x)
                Y.append(y)
            return X, Y
        return extract_data(self.train_data) + extract_data(self.test_data)
