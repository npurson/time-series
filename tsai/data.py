import os.path as osp
import numpy as np

from pyts.approximation import DiscreteFourierTransform


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
            if 'SampleFrequence' in line:
                freq = int(line[16:])
            elif 'DataInput' in line:
                series = np.array(line[10:].split(',')).astype(np.float64)
    return (freq, series)


def get_dataset(data_root='/home/jhy/repos/time-series/data/nari0516'):
    CLASSES = (
        '雷击', '反击', '绕击', '外破', '山火', '施工碰线', '异物短路', '冰害',
        '覆冰过载', '脱冰跳跃', '舞动', '风偏', '鸟害', '污闪', '其他'
    )

    def data_pipeline(path):
        freq, x = read_wavetxt(path)
        x = sample(x, freq, 5e5)
        x = x / 5000 if np.max(x) < 5000 else x / np.max(x)
        # dft = DiscreteFourierTransform()
        # xf = dft.fit_transform(x.reshape(1, -1))[0]
        # x = np.vstack((x, xf / np.max(xf)))
        return x

    with open(osp.join(data_root, 'train.txt')) as f:
        train_data = [line.strip().split(' ') for line in f.readlines()]
    with open(osp.join(data_root, 'val.txt')) as f:
        test_data = [line.strip().split(' ') for line in f.readlines()]

    X_train = []
    y_train = []
    for path, cls in train_data:
        x = data_pipeline(path)
        y = CLASSES.index(cls)
        X_train.append(x)
        y_train.append(y)

    X_valid = []
    y_valid = []
    for path, cls in test_data:
        x = data_pipeline(path)
        y = CLASSES.index(cls)
        X_valid.append(x)
        y_valid.append(y)
    return X_train, y_train, X_valid, y_valid


def get_bdataset(data_root='/home/jhy/repos/time-series/data/nari0516'):
    CLASSES = ('雷击', '非雷击')

    def data_pipeline(path):
        freq, x = read_wavetxt(path)
        x = sample(x, freq, 5e5)
        x = x / 5000 if np.max(x) < 5000 else x / np.max(x)
        # dft = DiscreteFourierTransform()
        # xf = dft.fit_transform(x.reshape(1, -1))[0]
        # x = np.vstack((x, xf / np.max(xf)))
        return x

    with open(osp.join(data_root, 'train.txt')) as f:
        train_data = [line.strip().split(' ') for line in f.readlines()]
    with open(osp.join(data_root, 'val.txt')) as f:
        test_data = [line.strip().split(' ') for line in f.readlines()]

    X_train = []
    y_train = []
    for path, cls in train_data:
        x = data_pipeline(path)
        y = CLASSES.index(cls)
        X_train.append(x)
        y_train.append(y)

    X_valid = []
    y_valid = []
    for path, cls in test_data:
        x = data_pipeline(path)
        y = CLASSES.index(cls)
        X_valid.append(x)
        y_valid.append(y)
    return X_train, y_train, X_valid, y_valid
