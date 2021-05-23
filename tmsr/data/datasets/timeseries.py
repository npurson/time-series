from torch.utils.data import Dataset


class TimeSeries(Dataset):
    """Time Series dataset.
    """

    CLASSES = (
        'leiji', 'fanji', 'raoji', 'waipo', 'shanhuo',
        'shigongpengxian', 'yiwuduanlu', 'binghai',
        'fubingguozai', 'tuobingtiaoyue', 'wudong',
        'fengpian', 'niaohai', 'wushan', 'others'
    )

    def __init__(self, data_root):
        ...

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> dict:
        return self.data[index]
