import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
# from tsai.all import *
from pyts.image import GramianAngularField
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


def main():
    data_root = '/home/jhy/repos/time-series/data/nari0516'
    with open(osp.join(data_root, 'train.txt'), encoding='utf-8') as f:
        data1 = [line.strip().split(' ') for line in f.readlines()]
    with open(osp.join(data_root, 'val.txt'), encoding='utf-8') as f:
        data2 = [line.strip().split(' ') for line in f.readlines()]
    for d in tqdm(data1 + data2):
        freq, ts = read_wavetxt('/home/jhy/repos/time-series/' + d[0])
        ts = sample(ts, freq, 5e5)

        fig, axs = plt.subplots(4, 1)
        
        # axs[0, 0].plot(range(len(ts)), ts)
        # tf = DiscreteFourierTransform()
        # msp = tf.fit_transform(ts.reshape(1, -1))

        # axs[0, 1].plot(range(len(msp[0])), msp[0])
        # gaf = GramianAngularField(image_size=224, method='summation')
        # axs[1, 0].imshow(gaf.fit_transform(ts.reshape(1, -1))[0])

        # img = plt.imshow(gaf.fit_transform(ts.reshape(1, -1))[0])
        # img = img.make_image(renderer=None)[0]
        # from PIL import Image
        # img = Image.fromarray(img[::-1, :, :3])

        # axs[1, 1].imshow(gaf.fit_transform(msp.reshape(1, -1))[0])
        # plt.show()
        plt.savefig('/home/jhy/repos/time-series/' + d[0].split('/')[-1][:-4] + '.jpg')

        # import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
