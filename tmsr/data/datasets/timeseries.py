import os.path as osp
from torch.utils.data import Dataset

from ..utils import read_wavetxt


class TSDataset(Dataset):
    """Time Series dataset.
    """

    CLASSES = (
        '雷击', '反击', '绕击', '外破', '山火', '施工碰线', '异物短路', '冰害',
        '覆冰过载', '脱冰跳跃', '舞动', '风偏', '鸟害', '污闪', '其他'
    )

    def __init__(self, data_root='data/nari0516', mode='train'):
        assert mode in ['train', 'val']
        with open(osp.join(data_root, mode + '.txt')) as f:
            self.data = [line.strip().split(' ') for line in f.readlines()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> dict:
        path, label = self.data[index]
        freq, series = read_wavetxt(path)
        return (self.transform(freq, series), label)
