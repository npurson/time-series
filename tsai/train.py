import argparse
import itertools
from os import X_OK
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


def plot_confusion_matrix(cm, classes, normalize=False):

    font = font_manager.FontProperties(fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        cm = cm.astype('int')
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.xticks(np.arange(len(classes)), classes, rotation=45, fontproperties=font)
    plt.yticks(np.arange(len(classes)), classes, fontproperties=font)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment='center',
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    plt.xlabel('Pred')
    plt.ylabel('Label')
    if not normalize:
        plt.savefig('demo/cm.png')
    else:
        plt.savefig('demo/cmp.png')
    plt.cla()


def main(args):

    tsdata = TSData('/home/jhy/repos/time-series/data/nari0616')
    X_train, y_train, X_valid, y_valid = tsdata()

    # tsdata = TSData('/home/jhy/repos/time-series/data/nari0823')
    # X_train2, y_train2, X_valid2, y_valid2 = tsdata()
    # X_train += X_train2
    # y_train += y_train2
    # X_valid += X_valid2
    # y_valid += y_valid2

    tsdata = TSData('/home/jhy/repos/time-series/data/nari0602')
    X_train2, y_train2, X_valid2, y_valid2 = tsdata()
    X_train += X_train2
    y_train += y_train2
    X_valid += X_valid2
    y_valid += y_valid2


    print(f'Train count: {Counter(y_train)}')
    print(f'Valid count: {Counter(y_valid)}')

    tfms = [
        TSIdentity,
        TSMagAddNoise,
        TSMagMulNoise,
        TSTimeNoise,
        TSRandomFreqNoise,
        TSShuffleSteps,
        TSRandomTimeScale,
        TSRandomTimeStep,
        TSMagScale,
        TSBlur,
        TSSmooth,
        TSCutOut,
        TSTimeWarp,
    ]
    # tfms = None

    train_ds = TSDatasets(X_train, y_train, types=(TSTensor, TensorCategory))
    valid_ds = TSDatasets(X_valid, y_valid, types=(TSTensor, TensorCategory))
    train_dl = TSDataLoader(train_ds, num_workers=4, batch_tfms=tfms, bs=32, drop_last=False)
    valid_dl = TSDataLoader(valid_ds, num_workers=4, bs=32, drop_last=False)
    dls = TSDataLoaders(train_dl, valid_dl, device=torch.device('cuda:0'))

    model = InceptionTimeXLPlus(1, len(tsdata.classes))
    # model = xresnet1d50_deeperplus(1, len(tsdata.classes))
    # for param in model.backbone.parameters():
    #     param.requires_grad = False
    if args.load:
        model.load_state_dict(torch.load('save/' + args.load))

    lr = Learner(dls, model, metrics=accuracy, loss_func=nn.CrossEntropyLoss())
    lr.fit_one_cycle(80, lr_max=5e-3)

    if args.save:
        torch.save(model.state_dict(), 'save/' + args.save)

    probs = lr.get_preds(dl=valid_dl)[0]
    preds = torch.argmax(probs, axis=1)
    print('acc:', sum(preds.numpy() == np.array(y_valid)) / len(y_valid))
    cm = confusion_matrix(y_valid, preds)
    plot_confusion_matrix(cm, tsdata.classes, normalize=True); plot_confusion_matrix(cm, tsdata.classes)


if __name__ == '__main__':
    args = parse_args()
    main(args)
