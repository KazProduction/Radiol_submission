import argparse
import os

import time

import SimpleITK as sitk

from metrics.myloss import MutualInformation, MIND_SSC_loss, NCC, Get_Ja
# from Dataset.loader import learn2regDS, PadResize, ToTensor
from Dataset.loader1 import Pretrain, learn2regDS, Padding, Resize, ToTensor, CreateOnehotLabel, NormThresholding

import monai.metrics
import torch
# from networks.recursive_cascade_networks import RecursiveCascadeNetwork
from networks.RCN_lung import RecursiveCascadeNetwork
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
# from metrics.losses import total_loss
from monai.losses import DiceLoss, MultiScaleLoss, DiceCELoss, BendingEnergyLoss

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from monai.data import DataLoader, Dataset, CacheDataset, decollate_batch
from monai.metrics import DiceMetric
from monai.networks.blocks import Warp
from monai.transforms import (
    Compose,
    SpatialPadd,
    LoadImaged,
    RandAffined,
    Resized,
    ScaleIntensityRanged,
    ScaleIntensityd,
    Rand3DElasticd,
    ThresholdIntensityd,
    EnsureChannelFirstd,
    Orientationd,
    AsDiscrete,
    AddChanneld,
    RandGaussianNoised
)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    # img.SetOrigin(ref_img.GetOrigin())
    # img.SetDirection(ref_img.GetDirection())
    # img.SetSpacing(ref_img.GetSpacing())
    spacing = (5.0, 5.0, 5.0)
    img.SetSpacing(spacing)
    sitk.WriteImage(img, os.path.join('', name))
    # sitk.WriteImage(img, os.path.join('Result/ct2mr_all/preprocess_mr', name))

def save_ddf(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, ...].cpu().detach().numpy())
    # img.SetOrigin(ref_img.GetOrigin())
    # img.SetDirection(ref_img.GetDirection())
    # img.SetSpacing(ref_img.GetSpacing())
    # spacing = (5.0, 5.0, 5.0)
    # img.SetSpacing(spacing)
    sitk.WriteImage(img, os.path.join('', name))


def show_slice_img(moving_image, fixed_image, slice_num):
    plt.figure("check", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("moving_image")
    plt.imshow(moving_image.cpu().squeeze().detach().numpy()[slice_num, :, :], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("fixed_image")
    plt.imshow(fixed_image.cpu().squeeze().detach().numpy()[slice_num, :, :], cmap="gray")
    plt.show()

def show_registered_slices(moving_image, registered_img, fixed_image, slice_num):
    plt.figure("check", (12, 6))
    plt.subplot(1, 4, 1)
    plt.title("moving")
    plt.imshow(moving_image.cpu().squeeze().detach().numpy()[slice_num, :, :], cmap="gray")
    plt.subplot(1, 4, 2)
    plt.title("warped")
    plt.imshow(registered_img.cpu().squeeze().detach().numpy()[slice_num, :, :], cmap="gray")
    plt.subplot(1, 4, 3)
    plt.title("fixed")
    plt.imshow(fixed_image.cpu().squeeze().detach().numpy()[slice_num, :, :], cmap="gray")
    plt.subplot(1, 4, 4)
    plt.title("registration")
    plt.imshow(fixed_image.cpu().squeeze().detach().numpy()[slice_num, :, :], cmap="gray")
    # plt.imshow(moving_image.cpu().squeeze().detach().numpy()[slice_num, :, :], cmap="Reds")
    plt.imshow(registered_img.cpu().squeeze().detach().numpy()[slice_num, :, :], cmap="Blues", alpha=0.7)
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('--b', "--batch_size", type=int, default=1)
parser.add_argument('--n', "--n_cascades", type=int, default=5)
parser.add_argument('--e', "--n_epochs", type=int, default=500)
parser.add_argument('--i', "--n_iters_train", type=int, default=2000)
parser.add_argument('--iv', "--n_iters_val", type=int, default=2000)
parser.add_argument('--c', "--checkpoint_frequency", type=int, default=20)
parser.add_argument('--lr', "--learning_rate", type=float, default=5e-6)
args = parser.parse_args()




def load_data(moving_path, m_mask_path, fixed_path, f_mask_path):
    moving_list = []
    m_mask_list = []
    fixed_list = []
    f_mask_list = []
    for moving_file, m_mask, fixed_file, f_mask in zip(os.listdir(moving_path),
                                                       os.listdir(m_mask_path),
                                                       os.listdir(fixed_path),
                                                       os.listdir(f_mask_path)):
        moving_list.append(os.path.join(moving_path, moving_file))
        m_mask_list.append(os.path.join(m_mask_path, m_mask))
        fixed_list.append(os.path.join(fixed_path, fixed_file))
        f_mask_list.append(os.path.join(f_mask_path, f_mask))

    return np.array(moving_list), np.array(m_mask_list), np.array(fixed_list), np.array(f_mask_list)

def main():

    warped_dir = r''
    ref_img = sitk.ReadImage(
        r'')

    writer = SummaryWriter('./Log/Finetuning')

    '''Loading data'''

    '''ct2mr reg'''
    train_ct_img_path = r''
    train_ct_mask_path = r''
    train_mri_img_path = r''
    train_mri_mask_path = r''


    ct_img, ct_mask, mr_img, mr_mask = load_data(train_ct_img_path, train_ct_mask_path,
                                                                             train_mri_img_path, train_mri_mask_path)

    data_dic = [{"ct": ct_img, "ct_mask": ct_mask, "mri": mri, "mri_mask": mri_mask}
                   for ct_img, ct_mask, mri, mri_mask in
                   zip(ct_img, ct_mask, mr_img, mr_mask)]

    img_size = (128, 128, 128)

    trans = transforms.Compose([
        NormThresholding(ct_low=10, ct_up=90, mri_low=10, mri_up=90),
        # NormThresholding(ct_low=10, ct_up=90, mri_low=0, mri_up=100),
        AddChanneld(keys=["ct", "ct_mask", "mri", "mri_mask"]),
        SpatialPadd(keys=['ct', 'ct_mask'],
                    spatial_size=img_size),
        SpatialPadd(keys=['mri', 'mri_mask'],
                    spatial_size=img_size),
        # RandGaussianNoised(keys=["ct", "ct_mask", "mri", "mri_mask"],
        #                    prob=0.1),
        ToTensor(),
    ])

    test_ds = learn2regDS(data=data_dic, transform=trans)
    test_loader = DataLoader(test_ds, batch_size=args.b, shuffle=False)

    if not os.path.exists('./ckp/model_wts'):
        print("Creating ckp dir")
        os.makedirs('./ckp/model_wts')

    if not os.path.exists('./ckp/visualization'):
        print("Creating visualization dir")
        os.makedirs('./ckp/visualization')

    model = RecursiveCascadeNetwork(n_cascades=args.n, im_size=img_size)

    warp_layer = Warp().cuda()
    warp_layer_val = Warp(mode="nearest").cuda()

    trainable_params = []
    for submodel in model.stems:
        trainable_params += list(submodel.parameters())

    trainable_params += list(model.reconstruction.parameters())

    # Saving the losses
    train_loss_log = []
    reg_loss_log = []
    val_loss_log = []
    val_interval = 5
    val_vis_interval = 100
    save_iter = 100
    start_epoch = 0

    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    best_metric = -1
    best_metric_epoch = -1


    train_vis_interval = 50

    '''load your model'''
    ckp_path = r'./ckp/ct2mr/epoch_300.pth'
    ckp = torch.load(ckp_path)
    for n, submodel in enumerate(model.stems):
        submodel.load_state_dict(ckp[f"cascade {n}"])



    '''calculate initial dice score'''
    with torch.no_grad():
        metric_values = []
        init_dice = []
        for i, val_batch in enumerate(test_loader):

            img_name = os.path.basename(ct_img[i])
            fixed_img_name = os.path.basename(mr_img[i])

            moving = val_batch["ct"].cuda()
            fixed = val_batch["mri"].cuda()

            moving_label = val_batch["ct_mask"].cuda()
            fixed_label = val_batch["mri_mask"].cuda()

            '''vis'''
            # show_slice_img(moving, fixed, 64)

            dice = dice_metric(moving_label, fixed_label)
            print("*****" + img_name + "*****")
            print("dice: ", torch.mean(dice))
            init_dice.append(torch.mean(dice).detach().cpu().numpy())


        #     '''save transforming img'''
        #     moving_name = str(i) + "_moving.nii.gz"
        #     save_image(moving, ref_img, moving_name)
        #     fixed_name = fixed_img_name[:15] + "_processed.nii.gz"
        #     save_image(fixed, ref_img, fixed_name)
        #
            '''test'''
            warped, flows, ddf = model(fixed, moving)


            '''save img'''
            # warped_name = img_name[:15] + "_warped.nii.gz"
            # save_image(warped[-1], ref_img, warped_name)
            ddf_name = img_name[:15] + "_ddf.nii.gz"
            save_ddf(ddf, ref_img, ddf_name)

            '''vis registered img'''
            # show_registered_slices(moving, warped[-1], fixed, 64)

        #     registered_label = warp_layer_val(moving_label, ddf)
        #     dice = dice_metric(registered_label.detach().cpu(), fixed_label.detach().cpu())
        #     print("dice: ", dice)
        #     metric_values.append(torch.mean(dice).detach().numpy())
        # print("---Summary---")
        # print("mean DSC for inital img: {:.4f}, std: {:.4f}".format(np.mean(init_dice), np.std(init_dice)))
        # print("mean DSC for registered img: {:.4f}, std: {:.4f}".format(np.mean(metric_values), np.std(metric_values)))

if __name__ == '__main__':
    main()