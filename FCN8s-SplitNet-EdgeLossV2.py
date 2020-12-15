#!/usr/bin/env python
# coding: utf-8

import os
import torch
import numpy as np
import random
from torch import nn
from torchvision import transforms as tfs
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from tqdm import tqdm
import hiddenlayer as hl
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pdb
import cv2
device = torch.device("cuda")


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
    #     img, lab = random_crop(img, lab, crop_size, rc=True)
    # 转化为SAR图像，即，灰度化，加椒盐噪声
    imgray = np.array(img)
    #     imgray = imgray[:, :, np.newaxis]
    #     imgray = np.asarray(img)
    img_tfs = tfs.Compose([
        tfs.ToTensor(),
        #         tfs.Normalize([0.5], [0.225]),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = img_tfs(imgray)

    lab_d = np.array(lab_d)
    lab = np.ones(lab_d.shape[:2], dtype=np.int)
    b = lab_d[:, :, 0] == 0
    g = lab_d[:, :, 1] == 0
    r = lab_d[:, :, 2] == 128
    lab[~b] = 0
    lab[~g] = 0
    lab[~r] = 0
    return img, torch.from_numpy(lab)


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
        lab = self.label_list[index]
        img = Image.open(img)
        #         lab = Image.open(lab)
        lab = cv2.imread(lab)
        img, lab = self.transform(img, lab, self.crop_size, rc=self.rc)
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


def bilinear_kernel(in_channels, out_channels, kernei_size):
    factor = (kernei_size + 1) // 2
    if kernei_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernei_size, :kernei_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 -
                                                 abs(og[1] - center) / factor)
    filtr = filt.reshape(-1)
    filtr = np.tile(filtr, (in_channels, out_channels))
    weight = filtr.reshape(in_channels, out_channels, filt.shape[0],
                           filt.shape[1]).astype(np.float32)
    #     weight = np.zeros((in_channels, out_channels, kernei_size, kernei_size),
    #                       dtype="float32")
    #     print(weight.shape, filt.shape)
    #     weight[range(in_channels), range(out_channels), :, :] = filt

    return torch.from_numpy(weight)


class MFM(nn.Module):
    def __init__(self):
        super(MFM, self).__init__()

    def forward(self, x):
        channels = x.shape[1]
        xo = x[:, 0, :, :]
        xo = torch.unsqueeze(xo, 1)
        for k in range(0, channels, 2):
            xout = torch.cat([
                torch.unsqueeze(x[:, k, :, :], 1),
                torch.unsqueeze(x[:, k + 1, :, :], 1)
            ],
                             dim=1)
            xout, _ = torch.max(xout, dim=1)
            xout = torch.unsqueeze(xout, 1)
            xo = torch.cat([xo, xout], dim=1)
        xo = xo[:, 1:, :, :]
        return xo


class FCN(nn.Module):
    """
    每一层的卷积深度都是64，应该逐层增加，可以为 64,128,256,512，递增
    """
    def __init__(self, in_ch, num_classes):
        super(FCN, self).__init__()
        self.downsample = nn.Conv2d(in_ch, 3, 3, stride=2,
                                    padding=1)  # 相当于一个 preconv 吧
        self.CBM1_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            MFM(),
        )
        self.CBM1_2 = nn.Sequential(
            nn.Conv2d(64 // 2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            MFM(),
        )
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=False)
        self.CBM2_1 = nn.Sequential(
            nn.Conv2d(64 // 2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            MFM(),
        )
        self.CBM2_2 = nn.Sequential(
            nn.Conv2d(64 // 2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            MFM(),
        )
        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=False)

        self.CBM3_1 = nn.Sequential(
            nn.Conv2d(64 // 2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            MFM(),
        )
        self.CBM3_2 = nn.Sequential(
            nn.Conv2d(64 // 2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            MFM(),
        )
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=False)

        self.CBM4_1 = nn.Sequential(
            nn.Conv2d(64 // 2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            MFM(),
        )
        self.CBM4_2 = nn.Sequential(
            nn.Conv2d(64 // 2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            MFM(),
        )

        #         self.upsample_pool1 = nn.ConvTranspose2d(
        #             64 // 2, 64, 4, stride=2,padding=1, bias=False)
        #         self.upsample_pool1.weight.data = bilinear_kernel(64 // 2, 64, 4)
        self.upsample_pool1 = nn.Conv2d(64 // 2, 64, 3, stride=1, padding=1)

        self.upsample_pool2 = nn.ConvTranspose2d(64 // 2,
                                                 64,
                                                 4,
                                                 stride=2,
                                                 padding=1,
                                                 bias=False)
        self.upsample_pool2.weight.data = bilinear_kernel(64 // 2, 64, 4)

        self.upsample_pool3 = nn.ConvTranspose2d(64 // 2,
                                                 64,
                                                 4,
                                                 stride=4,
                                                 padding=0,
                                                 bias=False)
        self.upsample_pool3.weight.data = bilinear_kernel(64 // 2, 64, 4)

        self.upsample_pool4 = nn.ConvTranspose2d(64 // 2,
                                                 64,
                                                 8,
                                                 stride=8,
                                                 padding=0,
                                                 bias=False)
        self.upsample_pool4.weight.data = bilinear_kernel(64 // 2, 64, 8)

        self.fcn1 = nn.Conv2d(64 * 4, 1024, 1, stride=1, padding=0)
        self.fcn2 = nn.Conv2d(1024, num_classes, 1, stride=1, padding=0)

        self.upsample_out_1 = nn.ConvTranspose2d(num_classes,
                                                 num_classes,
                                                 4,
                                                 stride=2,
                                                 padding=1,
                                                 bias=False)
        self.upsample_out_1.weight.data = bilinear_kernel(
            num_classes, num_classes, 4)


#         self.upsample_out_2 = nn.ConvTranspose2d(
#             num_classes, num_classes, 4, stride=2, padding=1,bias=False)
#         self.upsample_out_2.weight.data = bilinear_kernel(
#             num_classes, num_classes, 4)

    def forward(self, x, construct=False):
        h = x
        h = self.downsample(h)
        #         print("downsample",h.shape)#(128,128)
        h = self.CBM1_1(h)
        h = self.CBM1_2(h)
        p1 = h
        h = self.pool1(h)
        h = self.CBM2_1(h)
        h = self.CBM2_2(h)
        p2 = h
        h = self.pool2(h)
        h = self.CBM3_1(h)
        h = self.CBM3_2(h)
        p3 = h
        h = self.pool3(h)
        h = self.CBM4_1(h)
        h = self.CBM4_2(h)

        #         print("p123h",p1.shape, p2.shape, p3.shape, h.shape)
        p1u = self.upsample_pool1(p1)
        p2u = self.upsample_pool2(p2)
        p3u = self.upsample_pool3(p3)
        p4u = self.upsample_pool4(h)
        #         print("p1234u",p1u.shape, p2u.shape, p3u.shape, p4u.shape)
        h = torch.cat((p1u, p2u, p3u, p4u), dim=1)
        if construct:
            return h
        h = self.fcn1(h)
        h = self.fcn2(h)
        h = self.upsample_out_1(h)
        #         h = self.upsample_out_2(h)
        return h


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 -
                                                 abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class bnFCN8s(nn.Module):
    def __init__(self, encoder, n_class, th_ch=1):
        super().__init__()
        self.n_class = n_class
        self.encoder = encoder
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512,
                                          512,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          dilation=1,
                                          output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512,
                                          256,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          dilation=1,
                                          output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256,
                                          128,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          dilation=1,
                                          output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128,
                                          64,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          dilation=1,
                                          output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64,
                                          32,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          dilation=1,
                                          output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

        self.relu2 = nn.ReLU(inplace=True)
        self.deconv12 = nn.ConvTranspose2d(512,
                                           512,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           dilation=1,
                                           output_padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.deconv22 = nn.ConvTranspose2d(512,
                                           256,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           dilation=1,
                                           output_padding=1)
        self.bn22 = nn.BatchNorm2d(256)
        self.deconv32 = nn.ConvTranspose2d(256,
                                           128,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           dilation=1,
                                           output_padding=1)
        self.bn32 = nn.BatchNorm2d(128)
        self.deconv42 = nn.ConvTranspose2d(128,
                                           64,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           dilation=1,
                                           output_padding=1)
        self.bn42 = nn.BatchNorm2d(64)
        self.deconv52 = nn.ConvTranspose2d(64,
                                           32,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           dilation=1,
                                           output_padding=1)
        self.bn52 = nn.BatchNorm2d(32)
        self.classifier2 = nn.Conv2d(32, th_ch, kernel_size=1)

    def forward(self, x):
        output = self.encoder(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)

        score = self.relu(self.deconv1(x5))  # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score +
                         x4)  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(score))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score +
                         x3)  # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(
            self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(
            self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(
            self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # size=(N, n_class, x.H/1, x.W/1)

        score2 = self.relu2(self.deconv12(x5))  # size=(N, 512, x.H/16, x.W/16)
        score2 = self.bn12(
            score2 + x4)  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score2 = self.relu2(
            self.deconv22(score2))  # size=(N, 256, x.H/8, x.W/8)
        score2 = self.bn22(score2 +
                           x3)  # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score2 = self.bn32(self.relu2(
            self.deconv32(score2)))  # size=(N, 128, x.H/4, x.W/4)
        score2 = self.bn42(self.relu2(
            self.deconv42(score2)))  # size=(N, 64, x.H/2, x.W/2)
        score2 = self.bn52(self.relu2(
            self.deconv52(score2)))  # size=(N, 32, x.H, x.W)
        score2 = self.classifier2(score2)  # size=(N, n_class, x.H/1, x.W/1)

        return score, score2  # size=(N, n_class, x.H/1, x.W/1)


from torchvision.models.vgg import VGG


class VGGNet(VGG):
    def __init__(self,
                 model='vgg16',
                 batch_norm=False,
                 requires_grad=True,
                 remove_fc=True,
                 show_params=False,
                 in_channels=3):
        super().__init__(
            make_layers(cfg[model], batch_norm, in_channels=in_channels))
        if batch_norm:
            self.ranges = ranges[model + 'bn']
        else:
            self.ranges = ranges[model]

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d" % (idx + 1)] = x

        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, in_channels=3):
    layers = []
    in_channels = in_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13':
    [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    'vgg19': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
        512, 'M', 512, 512, 512, 512, 'M'
    ],
}

ranges = {
    'vgg11': ((0, 3), (3, 6), (6, 11), (11, 16), (16, 21)),
    'vgg11bn': ((0, 4), (4, 8), (8, 15), (15, 22), (22, 29)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg13bn': ((0, 7), (7, 14), (14, 21), (21, 28), (28, 35)),
    'vgg16': ((0, 5), (5, 12), (12, 22), (22, 32), (32, 42)),
    'vgg16bn': ((0, 7), (7, 14), (14, 24), (24, 34), (34, 44)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37)),
    'vgg19bn': ((0, 7), (7, 14), (14, 27), (27, 40), (40, 53))
}

# ## 边缘检测


class Edgenet(nn.Module):
    def __init__(self, ):
        """
        算子: 
            Sobel 一阶微分算子, 包括纵横双斜角,共四个
            Laplace 二阶微分算子,共一个
        """
        super(Edgenet, self).__init__()
        f_x = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ])
        f_y = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ])
        f_tl_br = torch.tensor([
            [-2, -1, 0],
            [-1, 0, 1],
            [0, 1, 2],
        ])
        f_tr_bl = torch.tensor([
            [0, -1, -2],
            [1, 0, -1],
            [2, 1, 0],
        ])
        f_operator = [
            f_x,
            f_y,
            f_tl_br,
            f_tr_bl,
        ]
        self.conv_operator = nn.ModuleList()
        for i, operator_init in enumerate(f_operator):
            self.conv_operator.append(
                nn.Conv2d(
                    1,
                    1,
                    3,
                    padding=1,
                    bias=False,
                ))
            self.conv_operator[i].weight.data[:, :] = operator_init

    def forward(self, lab):
        edge = 0
        lab_l = lab.float()
        for i, conv_op in enumerate(self.conv_operator):
            edge += torch.abs(conv_op(lab_l))
        edge[edge > 0] = 1

        return edge.squeeze().long()


class Sobelnet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, coef_sobel=1.0):
        """
        算子: 
            Sobel 一阶微分算子, 包括纵横双斜角,共四个
            Laplace 二阶微分算子,共一个
        """
        super(Sobelnet, self).__init__()
        f_x = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ])
        f_y = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ])
        f_tl_br = torch.tensor([
            [-2, -1, 0],
            [-1, 0, 1],
            [0, 1, 2],
        ])
        f_tr_bl = torch.tensor([
            [0, -1, -2],
            [1, 0, -1],
            [2, 1, 0],
        ])
        f_laplace = torch.tensor([
            [1, 1, 1],
            [1, 8, 1],
            [1, 1, 1],
        ])

        f_operator = [
            f_x,
            f_y,
            f_tl_br,
            f_tr_bl,
            #             f_laplace,
        ]
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.coef_sobel = coef_sobel

        self.conv_operator = nn.ModuleList()
        for i, operator_init in enumerate(f_operator):
            self.conv_operator.append(
                nn.Conv2d(
                    self.in_ch,
                    self.out_ch,
                    3,
                    padding=1,
                    bias=False,
                ))
            self.conv_operator[i].weight.data[:, :] = operator_init

    def forward(self, output_th, pred_seg, lab):
        lab_1 = lab.float() * 1.8 - 0.9
        Res_p = pred_seg - output_th
        loss = 0
        coploss = []
        for i, conv_op in enumerate(self.conv_operator):
            copl = conv_op(lab_1)
            copr = conv_op(Res_p)
            coploss.append(copl)
            #             loss += torch.mean(((1 - lab) * pred_seg.unsqueeze(1) + lab *
            #                                (1 - pred_seg.unsqueeze(1))**0.75 )*
            #                                (conv_op(lab_1) - conv_op(Res_p))**2)
            loss += torch.mean(
                ((1 - lab) * (0.25 + pred_seg.unsqueeze(1)) + lab *
                 (1 - pred_seg.unsqueeze(1))**0.75) * (copl - copr)**2)


#             loss += torch.mean((conv_op(lab_1) - conv_op(Res_p))**2)

        return loss, coploss


class ThresholdLoss(nn.Module):
    def __init__(self, threshold_rate_0, threshold_rate_1):
        super(ThresholdLoss, self).__init__()
        self.threshold_rate_0 = threshold_rate_0
        self.threshold_rate_1 = threshold_rate_1

    def forward(self, output_th, pred_seg, lab):
        lab = lab.float()
        #         print(output_th.shape, pred_seg.shape, lab.shape)
        targets = lab * (pred_seg * self.threshold_rate_1 - 0.) + (1 - lab) * (
            (1 - (1 - pred_seg) * self.threshold_rate_0) + 0.)
        loss = torch.mean((output_th - targets)**2)
        return loss


# # train.py

img_root = "road_dataset/image_train/"
lab_root = "road_dataset/label_train/"

input_size = (256, 256)
batch_size = 8
NUM_CLASSES = 2
load_path = "FCN8s_SplitNet_EdgeLossV2.pth"
save_path = "FCN8s_SplitNet_EdgeLossV2.pth"
EPOCH_start = 0
EPOCH = 1000
base_LR = 1e-4
decay_step = 100
decay_rate = 0.5
wd = 1e-3
mIU_benchmark = 0

train_data = MapDataset(input_size,
                        img_transforms,
                        img_root,
                        lab_root,
                        rc=False)
train_dataloader = torch.utils.data.DataLoader(train_data,
                                               batch_size,
                                               True,
                                               num_workers=32,
                                               drop_last=True)

img_root_t = "road_dataset/image_val/"
lab_root_t = "road_dataset/label_val/"

test_data = MapDataset(input_size,
                       img_transforms,
                       img_root_t,
                       lab_root_t,
                       rc=False)
test_dataloader = torch.utils.data.DataLoader(test_data,
                                              8,
                                              shuffle=False,
                                              num_workers=8,
                                              drop_last=True)

encoder_one = VGGNet('vgg16', batch_norm=True, in_channels=3)
net_one = bnFCN8s(encoder_one, NUM_CLASSES, 1).to(device)
thloss = Sobelnet().to(device)
edgenet = Edgenet().to(device)
if load_path is not None:
    if os.path.isfile(load_path):
        try:
            checkpoint = torch.load(load_path)
            net_one.load_state_dict(checkpoint['state_one'])
            mIU_benchmark = checkpoint['mIU']
            print("Load last checkpoint OK ")
            print("mIU=", mIU_benchmark)
        except:
            print("Can't Load the checkpoint QAQ")
            mIU_benchmark = 0
    else:
        EPOCH_start = 0
        print("Can't find the checkpoint ,start train from epoch 0 ...")

his = hl.History()
canv = hl.Canvas()
optm_one = torch.optim.Adam(net_one.parameters(), lr=base_LR, weight_decay=wd)
scheduler_lr_one = torch.optim.lr_scheduler.StepLR(optm_one,
                                                   step_size=decay_step,
                                                   gamma=decay_rate)


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


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir=os.path.join("./log"))
# Loss = nn.NLLLoss()
Loss = nn.CrossEntropyLoss()

l_seg = AverageValueMeter()
l_th = AverageValueMeter()
l_all = AverageValueMeter()
train_miou_seg = AverageValueMeter()
train_miou_th = AverageValueMeter()

test_acc = AverageValueMeter()
test_mean_iu = AverageValueMeter()
test_acc_e = AverageValueMeter()
test_mean_iu_e = AverageValueMeter()

best_mIU = 0
best_mIU_2 = mIU_benchmark
EPOCH_start = 0
for ep in tqdm(range(EPOCH_start, EPOCH_start + EPOCH), dynamic_ncols=True):
    train_length = len(train_dataloader)
    l_seg.reset()
    l_th.reset()
    l_all.reset()
    train_miou_seg.reset()
    train_miou_th.reset()
    net_one.train()
    for i, data in tqdm(enumerate(train_dataloader),
                        total=train_length,
                        leave=False):
        if i > 10:
            break
        img, lab = data
        img, lab = img.to(device), lab.to(device)
        seg_output, th_output = net_one(img)
        seg_output_p = torch.softmax(seg_output, dim=1)[:, 1]
        loss_seg = Loss(seg_output, lab)
        loss_th, cl = thloss(th_output,
                             seg_output_p.detach().unsqueeze(1),
                             lab.unsqueeze(1))
        loss = loss_seg + loss_th
        optm_one.zero_grad()
        loss.backward()
        optm_one.step()

        l_seg.add(loss_seg.item())
        l_th.add(loss_th.item())
        l_all.add(loss.item())

        seg_pred = seg_output.max(dim=1)[1]
        th_pred = torch.zeros_like(seg_pred).squeeze()
        th_pred[seg_output_p > th_output.squeeze()] = 1

        _, _, train_mean_iu_i, _ = label_accuracy_score(
            lab.data.cpu().numpy(),
            seg_pred.data.cpu().numpy(), NUM_CLASSES)
        _, _, train_mean_iu_i_2, _ = label_accuracy_score(
            lab.data.cpu().numpy(),
            th_pred.data.cpu().numpy(), NUM_CLASSES)

        train_miou_seg.add(train_mean_iu_i)
        train_miou_th.add(train_mean_iu_i_2)

    scheduler_lr_one.step()  # 动态学习率

    with torch.no_grad():
        test_acc.reset()
        test_mean_iu.reset()
        test_acc_e.reset()
        test_mean_iu_e.reset()
        net_one.eval()
        for _ in tqdm(range(1), ):
            for i, data in tqdm(enumerate(test_dataloader),
                                total=test_dataloader.__len__(),
                                leave=False):
                if i > 5:
                    break
                img, lab = data
                img, lab = img.to(device), lab.to(device)

                seg_output, th_output = net_one(img)
                seg_output_p = torch.softmax(seg_output, dim=1)[:, 1]
                seg_pred = seg_output.max(dim=1)[1]
                th_pred = torch.zeros_like(seg_pred).squeeze()
                th_pred[seg_output_p > th_output.squeeze()] = 1

                test_acc_i, _, test_mean_iu_i, _ = label_accuracy_score(
                    lab.data.cpu().numpy(),
                    seg_pred.data.cpu().numpy(), NUM_CLASSES)
                test_acc_i_e, _, test_mean_iu_i_e, _ = label_accuracy_score(
                    lab.data.cpu().numpy(),
                    th_pred.data.cpu().numpy(), NUM_CLASSES)

                test_acc.add(test_acc_i)
                test_mean_iu.add(test_mean_iu_i)
                test_acc_e.add(test_acc_i_e)
                test_mean_iu_e.add(test_mean_iu_i_e)

    #画图
    writer.add_scalars(
        "TrainLoss",
        {
            "loss_seg": l_seg.value()[0],
            "loss_th": l_th.value()[0],
            "loss_all": l_all.value()[0],
        },
        ep,
    )
    writer.add_scalars(
        "miou",
        {
            "train_miou_seg": train_miou_seg.value()[0],
            "train_miou_th": train_miou_th.value()[0],
            "test_miou": test_mean_iu.value()[0],
            "test_miou2": test_mean_iu_e.value()[0],
            "best_mIU": best_mIU,
            "best_mIU2": best_mIU_2,
        },
        ep,
    )
    image_batch = torch.zeros((3, 1, input_size[0], input_size[0])).float()
    image_batch[0, 0] = seg_pred[0].float()
    image_batch[1, 0] = th_pred[0].float()
    image_batch[2, 0] = lab[0].float()
    writer.add_images("test_img",
                      image_batch,
                      global_step=ep,
                      dataformats="NCHW")

    if test_mean_iu_e.value()[0] >= best_mIU_2:
        best_mIU = train_miou_seg.value()[0]
        best_mIU_2 = test_mean_iu_e.value()[0]
        state = {
            'state_one': net_one.state_dict(),
            'EPOCH_start': ep,
            'mIU': best_mIU_2
        }
        torch.save(state, save_path)
