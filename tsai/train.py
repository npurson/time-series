import itertools
import torch
import matplotlib.pyplot as plt

from collections import Counter
from tsai.all import *
from sklearn.metrics import confusion_matrix
from matplotlib import font_manager

from data import get_dataset, get_bdataset


def plot_confusion_matrix(cm):
    CLASSES = (
        '雷击', '反击', '绕击', '外破', '山火', '施工碰线', '异物短路', '冰害',
        '覆冰过载', '脱冰跳跃', '舞动', '风偏', '鸟害', '污闪', '其他'
    )

    font = font_manager.FontProperties(fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(np.arange(len(CLASSES)), CLASSES, rotation=45, fontproperties=font)
    plt.yticks(np.arange(len(CLASSES)), CLASSES, fontproperties=font)

    for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i, j], 'd'), horizontalalignment='center',
            color = 'white' if cm[i,j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    plt.xlabel('Pred')
    plt.ylabel('Label')
    plt.savefig('cm.png')


def main():

    import pdb; pdb.set_trace()

    X_train, y_train, X_valid, y_valid = get_bdataset('/home/jhy/repos/time-series/data/nari0823')
    print(f'Train count: {Counter(y_train)}')
    print(f'Valid count: {Counter(y_valid)}')

    train_ds = TSDatasets(X_train, y_train, types=(TSTensor, TensorCategory))
    valid_ds = TSDatasets(X_valid, y_valid, types=(TSTensor, TensorCategory))
    train_dl = TSDataLoader(train_ds, num_workers=4, bs=32)
    valid_dl = TSDataLoader(valid_ds, num_workers=4, bs=1)
    dls = TSDataLoaders(train_dl, valid_dl, device=torch.device('cuda:0'))

    model = InceptionTimeXLPlus(1, 2)
    lr = Learner(dls, model, metrics=accuracy, loss_func=nn.CrossEntropyLoss())
    lr.fit_one_cycle(50, lr_max=1e-3)


    X_train, y_train, X_valid, y_valid = get_dataset('/home/jhy/repos/time-series/data/nari0516')
    print(f'Train count: {Counter(y_train)}')
    print(f'Valid count: {Counter(y_valid)}')

    train_ds = TSDatasets(X_train, y_train, types=(TSTensor, TensorCategory))
    valid_ds = TSDatasets(X_valid, y_valid, types=(TSTensor, TensorCategory))
    train_dl = TSDataLoader(train_ds, num_workers=4, bs=32)
    valid_dl = TSDataLoader(valid_ds, num_workers=4, bs=1)
    dls = TSDataLoaders(train_dl, valid_dl, device=torch.device('cuda:0'))

    try:
        model.head = InceptionTimeXLPlus(1, 15).head
        lr = Learner(dls, model, metrics=accuracy, loss_func=nn.CrossEntropyLoss())
        lr.fit_one_cycle(50, lr_max=1e-3)
    except:
        import pdb; pdb.set_trace()

    probs = lr.get_preds(dl=valid_dl)[0]
    preds = torch.argmax(probs, axis=1)
    cm = confusion_matrix(y_valid, preds)
    plot_confusion_matrix(cm)


if __name__ == '__main__':
    main()
