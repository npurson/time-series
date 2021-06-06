import argparse
import itertools

import torch
import matplotlib.pyplot as plt

from collections import Counter
from tsai.all import *
from sklearn.metrics import confusion_matrix
from matplotlib import font_manager

from data import TSData


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str)
    parser.add_argument('--load', type=str)
    return parser.parse_args()


def plot_confusion_matrix(cm, classes):

    font = font_manager.FontProperties(fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(np.arange(len(classes)), classes, rotation=45, fontproperties=font)
    plt.yticks(np.arange(len(classes)), classes, fontproperties=font)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment='center',
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    plt.xlabel('Pred')
    plt.ylabel('Label')
    plt.savefig('demo/cm.png')


def main(args):

    tsdata = TSData('/home/jhy/repos/time-series/data/nari0602')
    X_train, y_train, X_valid, y_valid = tsdata()
    print(f'Train count: {Counter(y_train)}')
    print(f'Valid count: {Counter(y_valid)}')

    train_ds = TSDatasets(X_train, y_train, types=(TSTensor, TensorCategory))
    valid_ds = TSDatasets(X_valid, y_valid, types=(TSTensor, TensorCategory))
    train_dl = TSDataLoader(train_ds, num_workers=4, bs=32)
    valid_dl = TSDataLoader(valid_ds, num_workers=4, bs=32, drop_last=False)
    dls = TSDataLoaders(train_dl, valid_dl, device=torch.device('cuda:0'))

    model = InceptionTimeXLPlus(1, len(tsdata.CLASSES))
    # model = InceptionTimeXLPlus(1, len(tsdata.CLASSES))
    if args.load:
        model.load_state_dict(torch.load('save/' + args.load))
    # model.head = InceptionTimeXLPlus(1, 15).head

    lr = Learner(dls, model, metrics=accuracy, loss_func=nn.CrossEntropyLoss())
    lr.fit_one_cycle(50, lr_max=1e-3)

    if args.save:
        torch.save(model.state_dict(), 'save/' + args.save)

    probs = lr.get_preds(dl=valid_dl)[0]
    preds = torch.argmax(probs, axis=1)
    cm = confusion_matrix(y_valid, preds)
    plot_confusion_matrix(cm, tsdata.CLASSES)


if __name__ == '__main__':
    args = parse_args()
    main(args)
