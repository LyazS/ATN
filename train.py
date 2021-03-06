import os
import torch
import numpy as np
from torch import nn
from torch.utils.checkpoint import checkpoint
# from tqdm import tqdm
# from tqdm.auto import tqdm
import time
import segmentation_models_pytorch as smp
from dataset import MapDataset_v2, img_transforms, label_accuracy_score, AverageValueMeter
import matplotlib.pyplot as plt
from FCN8s import VGGNet, bnFCN8s, Sobelnet


def tqdm(dataset, total=0, leave=False):
    return dataset


train_root = "road_dataset/road_train_list.txt"
test_root = "road_dataset/road_val_list.txt"
file_prefix = "road_dataset"

input_size = (256, 256)
batch_size = 2
NUM_CLASSES = 2
load_path = "dlpeb7.pth"
save_path = "dlpeb7.pth"
EPOCH_start = 0
EPOCH = 1000
base_LR = 1e-4
decay_step = 100
decay_rate = 0.5
wd = 1e-3
mIU_benchmark = 0
iter_show = 100
show_plt = False

device = torch.device("cuda")
# net_one = smp.Unet(
#     encoder_name="efficientnet-b7",
#     encoder_weights=None,
#     encoder_depth=5,
#     in_channels=3,
#     classes=NUM_CLASSES,
# ).to(device)
encoder_one = VGGNet('vgg16', batch_norm=True, in_channels=3)
net_one = bnFCN8s(encoder_one, NUM_CLASSES, 1).to(device)
thloss = Sobelnet().to(device)

train_data = MapDataset_v2(input_size, img_transforms, train_root, file_prefix)
train_dataloader = torch.utils.data.DataLoader(train_data,
                                               batch_size,
                                               True,
                                               num_workers=4,
                                               drop_last=True)

test_data = MapDataset_v2(input_size, img_transforms, test_root, file_prefix)
test_dataloader = torch.utils.data.DataLoader(test_data,
                                              8,
                                              shuffle=False,
                                              num_workers=4,
                                              drop_last=True)

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

optm_one = torch.optim.Adam(net_one.parameters(), lr=base_LR, weight_decay=wd)
scheduler_lr_one = torch.optim.lr_scheduler.StepLR(optm_one,
                                                   step_size=decay_step,
                                                   gamma=decay_rate)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir=os.path.join("./log"))
Loss = nn.CrossEntropyLoss()

l_seg = AverageValueMeter()
train_miou = AverageValueMeter()
test_miou = AverageValueMeter()

best_mIU = 0

EPOCH_start = 0
for ep in tqdm(range(EPOCH_start, EPOCH_start + EPOCH)):
    l_seg.reset()
    train_miou.reset()
    net_one.train()
    last_time = time.time()
    for i, data in tqdm(enumerate(train_dataloader),
                        total=len(train_dataloader),
                        leave=False):
        if i > 20: break
        img, lab = data
        img, lab = img.to(device), lab.to(device)
        mask, threshold = net_one(img)
        loss_seg = Loss(mask, lab)

        mask_prob = torch.softmax(mask, dim=1)[:, 1]
        loss_th, cl = thloss(threshold,
                             mask_prob.detach().unsqueeze(1), lab.unsqueeze(1))

        optm_one.zero_grad()
        (loss_seg + loss_th).backward()
        optm_one.step()

        l_seg.add(loss_seg.item())
        mask = torch.argmax(mask, dim=1)
        th_pred = torch.zeros_like(mask).squeeze()
        th_pred[mask_prob > threshold.squeeze()] = 1
        _, _, train_mean_iu_i, _ = label_accuracy_score(
            lab.detach().cpu().numpy(),
            th_pred.detach().cpu().numpy(), NUM_CLASSES)

        train_miou.add(train_mean_iu_i)
        if i % iter_show == 0:
            now_time = time.time()
            timeiter = (now_time - last_time) / iter_show
            print(
                "Train epoch: {0}, iter: {1}/{2}. Time/iter: {3}. Time remain: {4}"
                .format(ep, i, len(train_dataloader), timeiter,
                        int((len(train_dataloader) - i) * timeiter)))
            last_time = now_time

    scheduler_lr_one.step()  # 动态学习率

    test_miou.reset()
    net_one.eval()
    for i, data in tqdm(enumerate(test_dataloader),
                        total=len(test_dataloader)):
        if i > 20: break
        img, lab = data
        img, lab = img.to(device), lab.to(device)
        with torch.no_grad():
            mask,threshold = net_one(img)
            mask_prob = torch.softmax(mask, dim=1)[:, 1]
            mask = torch.argmax(mask, dim=1)
        th_pred = torch.zeros_like(mask).squeeze()
        th_pred[mask_prob > threshold.squeeze()] = 1
        
        _, _, test_mean_iu_i, _ = label_accuracy_score(
            lab.detach().cpu().numpy(),
            th_pred.detach().cpu().numpy(), NUM_CLASSES)

        test_miou.add(test_mean_iu_i)
        if i % iter_show == 0:
            now_time = time.time()
            timeiter = (now_time - last_time) / iter_show
            print(
                "Test epoch: {0}, iter: {1}/{2}. Time/iter: {3}. Time remain: {4}"
                .format(ep, i, len(test_dataloader), timeiter,
                        int((len(test_dataloader) - i) * timeiter)))
            last_time = now_time

    #画图
    writer.add_scalars(
        "TrainLoss",
        {
            "loss_seg": l_seg.value()[0],
        },
        ep,
    )
    writer.add_scalars(
        "miou",
        {
            "train_miou": train_miou.value()[0],
            "test_miou": test_miou.value()[0],
            "best_mIU": best_mIU,
        },
        ep,
    )

    show_idx = np.random.randint(0, 8)
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax1.imshow(img[show_idx].permute(1, 2, 0).detach().cpu().numpy() * 0.225 +
               0.5)
    ax2.imshow(lab[show_idx].detach().cpu().numpy())
    ax3.imshow(mask[show_idx].detach().cpu().numpy())
    ax3.imshow(th_pred[show_idx].detach().cpu().numpy())

    if show_plt:
        plt.show()
    else:
        writer.add_figure("TrainFig", fig, ep, close=True)

    if test_miou.value()[0] >= best_mIU:
        best_mIU = test_miou.value()[0]
        state = {
            'state_one': net_one.state_dict(),
            'EPOCH_start': ep,
            'mIU': best_mIU
        }
        print("Saving model")
        torch.save(state, save_path)
