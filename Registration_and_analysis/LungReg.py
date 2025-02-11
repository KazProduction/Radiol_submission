import argparse
import os

import time

import SimpleITK as sitk

from metrics.myloss import MutualInformation, MIND_SSC_loss, NCC, Get_Ja
from Dataset.loader1 import Pretrain, learn2regDS, Padding, Resize, ToTensor, CreateOnehotLabel, NormThresholding
import monai.metrics
import torch
from networks.RCN_lung import RecursiveCascadeNetwork
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
# from metrics.losses import total_loss
from monai.losses import DiceLoss, MultiScaleLoss, DiceCELoss, BendingEnergyLoss
from metrics.losses import regularisation
from networks.spatial_transformer import SpatialTransform
from data_util.ctscan import sample_generator
from monai.networks.utils import one_hot
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
    sitk.WriteImage(img, os.path.join('Result', name))

def save_ddf(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, ...].cpu().detach().numpy())
    # img.SetOrigin(ref_img.GetOrigin())
    # img.SetDirection(ref_img.GetDirection())
    # img.SetSpacing(ref_img.GetSpacing())
    # spacing = (5.0, 5.0, 5.0)
    # img.SetSpacing(spacing)
    sitk.WriteImage(img, os.path.join('Result', name))


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
    # plt.imshow(fixed_image.cpu().squeeze().detach().numpy()[slice_num, :, :], cmap="gray")
    plt.imshow(moving_image.cpu().squeeze().detach().numpy()[slice_num, :, :], cmap="Reds")
    plt.imshow(registered_img.cpu().squeeze().detach().numpy()[slice_num, :, :], cmap="Blues", alpha=0.7)
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('--b', "--batch_size", type=int, default=1)
parser.add_argument('--n', "--n_cascades", type=int, default=5)
parser.add_argument('--e', "--n_epochs", type=int, default=500)
parser.add_argument('--i', "--n_iters_train", type=int, default=2000)
parser.add_argument('--iv', "--n_iters_val", type=int, default=2000)
parser.add_argument('--c', "--checkpoint_frequency", type=int, default=20)
parser.add_argument('--fixed_sample', type=int, default=100)
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

    writer = SummaryWriter('./Log')

    '''Loading data'''
    '''xe2mr reg'''
    train_ct_img_path = r''
    train_ct_mask_path = r''
    train_mri_img_path = r''
    train_mri_mask_path = r''


    train_ct_data, train_ct_mask, train_mri_data, train_mri_mask = load_data(train_ct_img_path, train_ct_mask_path,
                                                                             train_mri_img_path, train_mri_mask_path)

    ct_train, ct_test, ct_mask_train, ct_mask_test, mri_train, mri_test, mri_mask_train, mri_mask_test = \
        train_test_split(train_ct_data, train_ct_mask, train_mri_data, train_mri_mask, test_size=0.1, random_state=1)

    train_files = [{"ct": ct_img, "mri": mri, "ct_mask": ct_mask, "mri_mask": mri_mask}
                   for ct_img, mri, ct_mask, mri_mask in
                   zip(ct_train, mri_train, ct_mask_train, mri_mask_train)]
    # 将验证集变成dict形式
    val_files = [{"ct": ct_img, "mri": mri, "ct_mask": ct_mask, "mri_mask": mri_mask}
                 for ct_img, mri, ct_mask, mri_mask in
                 zip(ct_test, mri_test, ct_mask_test, mri_mask_test)]


    image_size = (128, 128, 128)
    img_size = (128, 128, 128)

    trans = transforms.Compose([
        NormThresholding(ct_low=1, ct_up=99, mri_low=10, mri_up=90),
        AddChanneld(keys=["ct", "ct_mask", "mri", "mri_mask"]),
        SpatialPadd(keys=['ct', 'ct_mask'],
                    spatial_size=img_size),
        SpatialPadd(keys=['mri', 'mri_mask'],
                    spatial_size=img_size),
        RandGaussianNoised(keys=["ct", "ct_mask", "mri", "mri_mask"],
                           prob=0.1),
        ToTensor(),
    ])

    val_trans = transforms.Compose([
        NormThresholding(ct_low=1, ct_up=99, mri_low=10, mri_up=90),
        AddChanneld(keys=["ct", "ct_mask", "mri", "mri_mask"]),
        SpatialPadd(keys=['ct', 'ct_mask'],
                    spatial_size=img_size),
        SpatialPadd(keys=['mri', 'mri_mask'],
                    spatial_size=img_size),
        ToTensor(),
    ])

    train_ds = learn2regDS(data=train_files, transform=trans)
    train_loader = DataLoader(train_ds, batch_size=args.b, shuffle=True)

    val_ds = learn2regDS(data=val_files, transform=val_trans)
    val_loader = DataLoader(val_ds, batch_size=args.b, shuffle=False)

    if not os.path.exists('./ckp/model_wts'):
        print("Creating ckp dir")
        os.makedirs('./ckp/model_wts')

    if not os.path.exists('./ckp/visualization'):
        print("Creating visualization dir")
        os.makedirs('./ckp/visualization')

    model = RecursiveCascadeNetwork(n_cascades=args.n, im_size=image_size)


    warp_layer = Warp().cuda()
    warp_layer_val = Warp(mode="nearest").cuda()

    trainable_params = []
    for submodel in model.stems:
        trainable_params += list(submodel.parameters())

    trainable_params += list(model.reconstruction.parameters())

    '''count parameter'''
    # num_parameters = 0
    # for submodel in model.stems:
    #     num_parameters += count_parameters(submodel)
    # print("Total number of parameters: ", num_parameters)
    # print("Reconstruction parameters: ", count_parameters(model.reconstruction))

    optim = Adam(trainable_params, lr=1e-5)
    scheduler = StepLR(optimizer=optim, step_size=10, gamma=0.96)


    # Saving the losses
    train_loss_log = []
    reg_loss_log = []
    val_loss_log = []
    val_interval = 5
    val_vis_interval = 100
    save_iter = 100
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    best_metric = -1
    best_metric_epoch = -1

    train_vis_interval = 50

    '''loss function'''
    intensity_loss_fn = MutualInformation(num_bin=48)
    dice_loss_fn = DiceLoss(include_background=False)
    label_loss_fn = MultiScaleLoss(dice_loss_fn, scales=[0, 1, 2, 4, 8, 16])

    reg_bending = BendingEnergyLoss()
    reg_weight = 2.0

    '''calculate initial dice score'''
    with torch.no_grad():
        metric_values = []
        for i, val_batch in enumerate(val_loader):
            moving = val_batch["ct"].cuda()
            fixed = val_batch["mri"].cuda()

            moving_label = val_batch["ct_mask"].cuda()
            fixed_label = val_batch["mri_mask"].cuda()
            # print("MRI shape: ", moving.shape)
            # print("CT shape: ", fixed.shape)
            '''vis'''
            show_slice_img(moving, fixed, 64)

            dice = dice_metric(moving_label, fixed_label)
            print("dice: ", torch.mean(dice))

    model.train()
    warp_layer.train()


    for epoch in range(1, args.e + 1):
        print("-" * 10)
        print(f"epoch {epoch}/{args.e}")
        epoch_loss = 0
        epoch_intensity_loss = 0
        epoch_label_loss = 0
        epoch_reg_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            moving = batch_data["ct"].cuda()
            fixed = batch_data["mri"].cuda()
            moving_label = batch_data["ct_mask"].cuda()
            fixed_label = batch_data["mri_mask"].cuda()

            warped, flows, ddf = model(fixed, moving)

            moving_label = warp_layer(moving_label, ddf)


            intensity_loss = intensity_loss_fn(fixed, warped[-1])

            '''label_loss'''
            label_loss = label_loss_fn(moving_label, fixed_label)


            reg = sum([reg_bending(flow) for flow in flows])

            loss = intensity_loss + label_loss + reg_weight * reg



            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            epoch_intensity_loss += intensity_loss.item()
            epoch_label_loss += label_loss.item()
            epoch_reg_loss += reg.item()

            '''vis'''
            # if epoch % train_vis_interval == 0:
            #      show_slice_img(warped[-1], moving_label, fixed, fixed_label, 64)

        epoch_loss /= step
        epoch_intensity_loss /= step
        epoch_label_loss /= step
        epoch_reg_loss /= step

        print(f"epoch {epoch} "
              f"average loss: {epoch_loss:.4f} "
              f"intensity loss: {epoch_intensity_loss:.4f} "
              f"label loss: {epoch_label_loss:.4f} "
              f"reg loss: {epoch_reg_loss:.4f}"
              )

        writer.add_scalar('Loss', epoch_loss, epoch)
        writer.add_scalar('Intensity Loss', epoch_intensity_loss, epoch)
        writer.add_scalar('Label Loss', epoch_label_loss, epoch)
        writer.add_scalar('Reg', epoch_reg_loss, epoch)

        if epoch % val_interval == 0:
            epoch_val_loss = 0
            epoch_val_intensity_loss = 0
            epoch_val_label_loss = 0
            epoch_val_reg_loss = 0
            model.eval()
            warp_layer.eval()
            warp_layer_val.eval()
            with torch.no_grad():
                metric_values = []
                Jac_lst = []
                for i, val_batch in enumerate(val_loader):
                    moving = val_batch["ct"].cuda()
                    fixed = val_batch["mri"].cuda()

                    moving_label = val_batch["ct_mask"].cuda()
                    fixed_label = val_batch["mri_mask"].cuda()

                    warped, flows, ddf = model(fixed, moving)

                    # moving_label = warp_layer_val(moving_label, flows[0]) # affine
                    moving_label = warp_layer_val(moving_label, ddf)

                    intensity_loss = intensity_loss_fn(fixed, warped[-1])
                    label_loss = label_loss_fn(moving_label, fixed_label)

                    # val_reg = regularisation(flows)
                    val_reg = sum([reg_bending(flow) for flow in flows])
                    val_loss = intensity_loss + label_loss + reg_weight * val_reg
                    # val_loss = val_intensity_loss + reg_weight * val_reg

                    epoch_val_loss += val_loss.item()
                    epoch_val_intensity_loss += intensity_loss.item()
                    epoch_val_label_loss += label_loss.item()
                    epoch_val_reg_loss += val_reg.item()

                    '''Jac'''
                    neg_jac = np.where(Get_Ja(ddf.detach().cpu().numpy()) <= 0, 1, 0)
                    neg_sum = np.sum(neg_jac)
                    Jac_lst.append(neg_sum)

                    '''vis'''
                    if epoch % val_vis_interval == 0:
                        show_slice_img(warped[-1], fixed, 64)
                        # show_slice_img(warped[0], fixed, 64)

                    '''save'''
                    if epoch % save_iter == 0:
                        warped_name = "epoch" + str(epoch) + "_" + str(i) + "_warped.nii.gz"
                        save_image(warped[-1], ref_img, warped_name)
                        print("warped images have saved.")


                    dice = dice_metric(moving_label.detach().cpu(), fixed_label.detach().cpu())
                    metric_values.append(torch.mean(dice).detach().numpy())
                metric = np.mean(metric_values)
                mean_Jac = np.mean(Jac_lst)

                print()
                print("-----Validation-----")
                print(f"current epoch: {epoch}\n"
                      f"average loss: {epoch_val_loss / 10.0:.4f} "
                      f"intensity loss: {epoch_val_intensity_loss / 10.0:.4f} "
                      f"label loss: {epoch_val_label_loss / 10.0:.4f} "
                      f"reg loss: {epoch_val_reg_loss / 10.0:.4f} \n"
                      f"val_Jac: {mean_Jac} \n"
                      f"mean_dice: {metric:.4f}")

                writer.add_scalar('val_loss', epoch_val_loss / 10.0, epoch)
                writer.add_scalar('val_DSC', metric, epoch)
                writer.add_scalar('val_Jac', mean_Jac, epoch)

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch
                print(f"best mean dice: {best_metric:.4f} "
                      f"at epoch: {best_metric_epoch}"
                     )
            model.train()
            warp_layer.train()


        scheduler.step()

        if epoch % args.c == 0:
            ckp = {}
            for i, submodel in enumerate(model.stems):
                ckp[f"cascade {i}"] = submodel.state_dict()

            ckp['train_loss'] = train_loss_log
            ckp['val_loss'] = val_loss_log
            ckp['epoch'] = epoch

            torch.save(ckp, f'./ckp/model_wts/epoch_{epoch}.pth')


if __name__ == '__main__':
    main()
