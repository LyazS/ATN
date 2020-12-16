import os
import random
import numpy as np
from torchvision import transforms as tfs
import torch
import cv2
cv2.setNumThreads(0)
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class AverageValueMeter():
    def __init__(self):
        self.N = 0
        self.ave = 0

    def add(self, x):
        self.ave = self.N / (1 + self.N) * self.ave + 1 / (1 + self.N) * x
        self.N += 1

    def reset(self):
        self.N = 0
        self.ave = 0

    def value(self):
        return self.ave, self.N


def read_as_list(rootpath):
    datalist = os.listdir(rootpath)
    datapath = [os.path.join(rootpath, i) for i in datalist]
    return datapath


def readimglab(img_root, lab_root):
    ip = read_as_list(img_root)
    lp = read_as_list(lab_root)
    return ip, lp


def random_crop(img, lab, crop_size, rc=True):
    wd, ht = crop_size
    w, h = img.size
    wd = min(wd, w)
    ht = min(ht, h)
    wd = wd - wd % 16
    ht = ht - ht % 16
    if rc:
        if w == wd:
            i = 0
        else:
            i = random.randint(0, w - wd)
        if h == ht:
            j = 0
        else:
            j = random.randint(0, h - ht)
    else:
        i = 0
        j = 0

    return img.crop((i, j, i + wd, j + ht)), lab.crop((i, j, i + wd, j + ht))


def img_transforms(img, lab_d, crop_size, rc=True):
    # img, lab = random_crop(img, lab, crop_size, rc=True)
    imgray = np.array(img)
    img_tfs = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = img_tfs(imgray)

    lab = np.array(lab_d)
    lab[lab != 1] = 0
    lab = torch.from_numpy(lab).long()
    return img, lab


class MapDataset(torch.utils.data.Dataset):
    """
    Map dataset
    """
    def __init__(self, crop_size, transform, img_root, lab_root, rc=True):
        self.crop_size = crop_size
        self.rc = rc
        self.transform = transform
        self.data_list, self.label_list = readimglab(img_root, lab_root)
        print("Read " + str(len(self.data_list)) + " images")

    def __getitem__(self, index):
        img = self.data_list[index]
        # lab = self.label_list[index]
        lab = img.replace("image", "label").replace("jpg", "png")
        if not os.path.exists(lab):
            raise Exception
        img = Image.open(img)
        lab = Image.open(lab)
        img, lab = self.transform(img, lab, self.crop_size, rc=self.rc)
        return img, lab

    def __len__(self):
        return len(self.data_list)


class MapDataset_v2(torch.utils.data.Dataset):
    def __init__(self, crop_size, transform, file_path, file_prefix):
        self.crop_size = crop_size
        self.transform = transform
        self.file_path = file_path
        self.file_prefix = file_prefix
        self.data_list = []
        with open(file_path, "r") as f:
            file_list = f.readlines()
            for p in file_list:
                il = p.replace('\n', '').replace('\r', '')
                self.data_list.append(il.split(" "))

    def __getitem__(self, idx):
        img_p, lab_p = self.data_list[idx]
        img_p = os.path.join(self.file_prefix, img_p)
        lab_p = os.path.join(self.file_prefix, lab_p)
        if not os.path.exists(img_p) or not os.path.exists(lab_p):
            print(img_p, "  ", lab_p)
        img = Image.open(img_p)
        lab = Image.open(lab_p)
        img, lab = self.transform(img, lab, self.crop_size)
        return img, lab

    def __len__(self):
        return len(self.data_list)


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) +
                       label_pred[mask],
                       minlength=n_class**2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) -
                              np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc
