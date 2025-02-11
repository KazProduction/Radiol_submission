import torch
from metrics.myloss import MutualInformation, MIND_SSC_loss, NCC
from monai.losses import MultiScaleLoss, DiceLoss, multi_scale, BendingEnergyLoss, DiceCELoss, SSIMLoss, ContrastiveLoss
from torch.nn import MSELoss
import numpy as np
import torch.nn.functional as F



def pearson_correlation(fixed, warped):
    flatten_fixed = torch.flatten(fixed, start_dim=1)
    flatten_warped = torch.flatten(warped, start_dim=1)

    mean1 = torch.mean(flatten_fixed)
    mean2 = torch.mean(flatten_warped)
    var1 = torch.mean((flatten_fixed - mean1) ** 2)
    var2 = torch.mean((flatten_warped - mean2) ** 2)

    cov12 = torch.mean((flatten_fixed - mean1) * (flatten_warped - mean2))
    eps = 1e-6
    pearson_r = cov12 / torch.sqrt((var1 + eps) * (var2 + eps))

    raw_loss = 1 - pearson_r

    return raw_loss


def regularize_loss(flow):
    """
    flow has shape (batch, 2, 521, 512)
    """
    dx = (flow[..., 1:, :] - flow[..., :-1, :]) ** 2
    dy = (flow[..., 1:] - flow[..., :-1]) ** 2

    d = torch.mean(dx) + torch.mean(dy)

    return d / 2.0


def regularize_loss_3d(flow):
    """
    flow has shape (batch, 3, 512, 521, 512)
    """
    dy = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
    dx = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
    dz = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]

    d = torch.mean(dx**2) + torch.mean(dy**2) + torch.mean(dz**2)

    return d / 3.0


def dice_loss(fixed_mask, warped):
    """
    Dice similirity loss
    """

    epsilon = 1e-6

    flat_mask = torch.flatten(fixed_mask, start_dim=1)
    flat_warp = torch.abs(torch.flatten(warped, start_dim=1))
    intersection = torch.sum(flat_mask * flat_warp)
    denominator = torch.sum(flat_mask) + torch.sum(flat_warp) + epsilon
    dice = (2.0 * intersection + epsilon) / denominator

    return 1 - dice


def jacobian_det(flow):
    raise NotImplementedError

'''intensity-based loss'''
# def total_loss(fixed, moving, flows):
#     # sim_loss = pearson_correlation(fixed, moving)
#     # sim_loss_f1 = MutualInformation(num_bin=48)
#     # sim_loss_f1 = NCC().loss
#     # sim_loss_f1 = SSIMLoss(spatial_dims=3)
#     # data_range = torch.tensor(fixed).max().unsqueeze(0)
#     # sim_loss = sim_loss_f1(torch.tensor(fixed), torch.tensor(moving), data_range)
#
#     # sim_loss_f2 = MIND_SSC_loss()
#     # sim_loss = sim_loss_f1(torch.tensor(fixed), torch.tensor(moving))
#
#     # sim_loss = sim_loss_f1(torch.tensor(fixed), torch.tensor(moving)) + sim_loss_f2(torch.tensor(fixed), torch.tensor(moving))
#     # Regularize all flows
#     if len(fixed.size()) == 4: #(N, C, H, W)
#         reg_loss = sum([regularize_loss(flow) for flow in flows])
#     else:
#         reg_loss = sum([regularize_loss_3d(flow) for flow in flows])
#     return sim_loss, reg_loss


'''multi-scale dice loss'''
# def total_loss(fixed, moving, flows):
#
#     label_loss = DiceLoss()
#     # label_loss = MultiScaleLoss(label_loss, scales=[0, 1, 2, 4])
#     label_loss = MultiScaleLoss(label_loss, scales=[0, 1, 2, 4, 8, 16])
#
#     cls_lst = [1, 2, 3, 4]
#     dice_loss_lst = []
#     for cls in cls_lst:
#         pred = (moving == cls).float()
#         gt = (fixed == cls).float()
#         loss_label_reg = label_loss(pred, gt)
#         dice_loss_lst.append(loss_label_reg)
#
#     sim_loss = torch.mean(torch.tensor(dice_loss_lst))
#
#     # Regularize all flows
#     # if len(fixed.size()) == 4: #(N, C, H, W)
#     #     reg_loss = sum([regularize_loss(flow) for flow in flows])
#     # else:
#     #     reg_loss = sum([regularize_loss_3d(flow) for flow in flows])
#
#     regularisation = BendingEnergyLoss()
#     if len(fixed.size()) == 4: #(N, C, H, W)
#         reg_loss = sum([regularisation(flow) for flow in flows])
#     else:
#         reg_loss = sum([regularisation(flow) for flow in flows])
#
#     return sim_loss, reg_loss

# def total_loss(fixed, moving, flows):
#
#     label_loss = MSELoss()
#     label_loss = MultiScaleLoss(label_loss, scales=[0, 1, 2, 4, 8])
#
#     cls_lst = [1, 2, 3, 4]
#     dice_loss_lst = []
#     for cls in cls_lst:
#         pred = (moving == cls).float()
#         gt = (fixed == cls).float()
#         loss_label_reg = label_loss(pred, gt)
#         dice_loss_lst.append(loss_label_reg)
#
#     sim_loss = torch.mean(torch.tensor(dice_loss_lst))
#
#
#     # Regularize all flows
#     if len(fixed.size()) == 4: #(N, C, H, W)
#         reg_loss = sum([regularize_loss(flow) for flow in flows])
#     else:
#         reg_loss = sum([regularize_loss_3d(flow) for flow in flows])
#
#     # regularisation = BendingEnergyLoss()
#     # if len(fixed.size()) == 4: #(N, C, H, W)
#     #     reg_loss = sum([regularisation(flow) for flow in flows])
#     # else:
#     #     reg_loss = sum([regularisation(flow) for flow in flows])
#     #
#     return sim_loss, reg_loss

# def label_loss(fixed, moving):
#     label_loss = DiceLoss(include_background=False)
#     # label_loss = MSELoss()
#     label_loss = MultiScaleLoss(label_loss, scales=[0, 1, 2, 4, 8, 16])
#
#     # sim_loss = label_loss(fixed, moving)
#     cls_lst = [1, 2, 3, 4]
#     dice_loss_lst = []
#
#     for cls in cls_lst:
#         pred = (moving == cls).float()
#         gt = (fixed == cls).float()
#         loss_label_reg = label_loss(pred, gt)
#         dice_loss_lst.append(loss_label_reg)
#     # sim_loss = torch.mean(dice_loss_lst)
#     sim_loss = torch.mean(torch.tensor(dice_loss_lst, requires_grad=True))
#     return sim_loss
#
#
# def intensity_loss(fixed, moving):
#     sim_loss_fn = MutualInformation(num_bin=48)
#     # sim_loss_fn = NCC().loss
#     # sim_loss_fn = MIND_SSC_loss()
#     sim_loss = sim_loss_fn(fixed, moving)
#
#     return sim_loss

def regularisation(flows):
    reg_loss = sum([regularize_loss_3d(flow) for flow in flows])
    return reg_loss




