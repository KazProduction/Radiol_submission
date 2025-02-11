import os
import torch
import numpy as np
from glob import glob
import SimpleITK as sitk
from torch.utils.data import Dataset
from skimage.transform import resize
import random

class Pretrain(Dataset):
    '''inter patient pretrain'''
    def __init__(self, ct_path, mri_path, ct_seg_path, mri_seg_path, transform=None):
        self.ct_path = ct_path
        self.mri_path = mri_path
        self.ct_seg_path = ct_seg_path
        self.mri_seg_path = mri_seg_path
        self.transform = transform
        print("total {} samples".format(len(os.listdir(self.ct_path))))
    def __len__(self):
        return (len(os.listdir(self.ct_path)))
    def __getitem__(self, idx):
        '''随机取CT，并且取其对应的label；随机取MR,并且取其对应的label'''
        CT_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.ct_path, os.listdir(self.ct_path)[idx])))
        CT_label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.ct_seg_path, os.listdir(self.ct_seg_path)[idx])))
        idx_mr = np.random.randint(0, len(os.listdir(self.mri_path)))
        MRI_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.mri_path, os.listdir(self.mri_path)[idx_mr])))
        MRI_label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.mri_seg_path, os.listdir(self.mri_seg_path)[idx_mr])))
        sample = {"ct" : CT_image, "mri" : MRI_image, "ct_mask" : CT_label, "mri_mask" : MRI_label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class learn2regDS(Dataset):
    '''Dataset for Learn2Reg challenge'''
    def __init__(self, data=None, transform=None):
        self.data = data
        self.transform = transform
        print("total {} samples".format(len(self.data)))

    def __len__(self):
        return (len(self.data))

    def __getitem__(self, idx):
        CT_image = self.data[idx]["ct"]
        # print(CT_image)
        CT_image = sitk.GetArrayFromImage(sitk.ReadImage(CT_image))

        # CT_image = sitk.GetArrayFromImage(sitk.ReadImage(CT_image)).transpose(0, 2, 1)
        # CT_image = np.flip(CT_image, axis=0)
        # CT_image = np.flip(CT_image, axis=1)
        # CT_image = np.flip(CT_image, axis=2)

        MRI_image = self.data[idx]["mri"]
        # print(MRI_image)
        MRI_image = sitk.GetArrayFromImage(sitk.ReadImage(MRI_image))
        CT_label = self.data[idx]["ct_mask"]
        CT_label = sitk.GetArrayFromImage(sitk.ReadImage(CT_label))
        MRI_label = self.data[idx]["mri_mask"]
        MRI_label = sitk.GetArrayFromImage(sitk.ReadImage(MRI_label))

        sample = {"ct" : CT_image, "mri" : MRI_image, "ct_mask" : CT_label, "mri_mask" : MRI_label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Padding(object):
    def __init__(self,  target_size):
        self.target_size = target_size

    def __call__(self, sample):
        # image, label = sample['image'], sample['label']
        ct, mri, ct_mask, mri_mask = sample['ct'], sample['mri'], sample['ct_mask'], sample['mri_mask']

        # pad the sample if necessary
        if ct.shape[0] <= self.target_size[0] or ct.shape[1] <= self.target_size[1] or ct.shape[2] <= \
                self.target_size[2]:
            pw = max((self.target_size[0] - ct.shape[0]) // 2, 0)
            ph = max((self.target_size[1] - ct.shape[1]) // 2, 0)
            pd = max((self.target_size[2] - ct.shape[2]) // 2, 0)

            ct = np.pad(ct, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            mri = np.pad(mri, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            ct_mask = np.pad(ct_mask, [(0, 0), (pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            mri_mask = np.pad(mri_mask, [(0, 0), (pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            # ct_mask = np.pad(ct_mask, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            # mri_mask = np.pad(mri_mask, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        #     label_size = ((5,) + self.target_size)
        #
        # return {"ct": resize(ct, self.target_size), "mri": resize(mri, self.target_size),
        #         "ct_mask": resize(ct_mask, label_size), "mri_mask": resize(mri_mask, label_size)}

        # label_size = ((5,) + self.target_size)

        return {"ct": ct, "mri": mri, "ct_mask": ct_mask, "mri_mask": mri_mask}

class Resize(object):
    def __init__(self, target_size):
        self.target_size = target_size
    def __call__(self, sample):
        ct, mri, ct_mask, mri_mask = sample['ct'], sample['mri'], sample['ct_mask'], sample['mri_mask']
        label_size = ((ct_mask.shape[0],) + self.target_size)
        return {"ct": resize(ct, self.target_size), "mri": resize(mri, self.target_size),
                "ct_mask": resize(ct_mask, label_size), "mri_mask": resize(mri_mask, label_size)}


class NormThresholding(object):
    def __init__(self, ct_low=70, ct_up=97, mri_low=70, mri_up=97):
        self.ct_low = ct_low
        self.ct_up = ct_up
        self.mri_low = mri_low
        self.mri_up = mri_up
    def __call__(self, sample):
        ct, mri, ct_mask, mri_mask = sample['ct'], sample['mri'], sample['ct_mask'], sample['mri_mask']

        low_bound_ct = np.percentile(ct, self.ct_low)
        up_bound_ct = np.percentile(ct, self.ct_up)
        ct[ct < low_bound_ct] = low_bound_ct
        ct[ct > up_bound_ct] = up_bound_ct
        ct = (ct - low_bound_ct) / (up_bound_ct - low_bound_ct)

        low_bound_mr = np.percentile(mri, self.mri_low)
        up_bound_mr = np.percentile(mri, self.mri_up)
        mri[mri < low_bound_mr] = low_bound_mr
        mri[mri > up_bound_mr] = up_bound_mr
        mri = (mri - low_bound_mr) / (up_bound_mr - low_bound_mr)

        return {"ct":ct, "mri": mri, "ct_mask": ct_mask, "mri_mask": mri_mask}

class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        ct, mri, ct_mask, mri_mask = sample['ct'], sample['mri'], sample['ct_mask'], sample['mri_mask']

        onehot_ct_label = np.zeros((self.num_classes, ct_mask.shape[0], ct_mask.shape[1], ct_mask.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_ct_label[i, :, :, :] = (ct_mask == i).astype(np.float32)

        onehot_mri_label = np.zeros((self.num_classes, mri_mask.shape[0], mri_mask.shape[1], mri_mask.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_mri_label[i, :, :, :] = (mri_mask == i).astype(np.float32)

        return {"ct": ct, "mri": mri, "ct_mask": onehot_ct_label, "mri_mask": onehot_mri_label}

# class Cutout(object):
#     def __init__(self, n_holes, length):
#         self.n_holes, self.length = n_holes, length
#         self.h_rand_ct = [random.random() for i in range(self.n_holes)]
#         self.w_rand_ct = [random.random() for i in range(self.n_holes)]
#         self.d_rand_ct = [random.random() for i in range(self.n_holes)]
#         self.h_rand_mri = [random.random() for i in range(self.n_holes)]
#         self.w_rand_mri = [random.random() for i in range(self.n_holes)]
#         self.d_rand_mri = [random.random() for i in range(self.n_holes)]
#
#
#     def __call__(self, sample):
#         ct_img = sample["ct"]
#         ct_label = sample["ct_mask"]
#         mri_img = sample["mri"]
#         mri_label = sample["mri_mask"]
#
#         cutout_ct_img, cutout_ct_label = cutout(ct_img, ct_label, self.n_holes, self.length, self.h_rand_ct, self.w_rand_ct, self.d_rand_ct)
#         cutout_mri_img, cutout_mri_label = cutout(mri_img, mri_label, self.n_holes, self.length, self.h_rand_mri, self.w_rand_mri, self.d_rand_mri)
#         return {"ct": cutout_ct_img, "mri": cutout_mri_img, "ct_mask": cutout_ct_label, "mri_mask": cutout_mri_label}
#
# def cutout(img, label, n_holes, length, h_rand, w_rand, d_rand):
#     h, w, d = img.shape
#     mask = np.ones((h, w, d), np.int32)
#     for n in range(n_holes):
#         x = int(h * h_rand[n])
#         y = int(w * w_rand[n])
#         z = int(d * d_rand[n])
#
#         x1 = int(np.clip(x - length / 2, 0, h))
#         x2 = int(np.clip(x + length / 2, 0, h))
#         y1 = int(np.clip(y - length / 2, 0, w))
#         y2 = int(np.clip(y + length / 2, 0, w))
#         z1 = int(np.clip(z - length / 2, 0, d))
#         z2 = int(np.clip(z + length / 2, 0, d))
#         mask[x1:x2, y1:y2, z1:z2] = 0
#     img_mask = mask[:, :, :]
#     label_mask = mask[None, :, :, :]
#     img = img * img_mask
#     label = label * label_mask
#     return img, label

class Cutout(object):
    def __init__(self, mask_size, p=1.0, cutout_inside=1):
        self.mask_size = mask_size
        self.p = p
        self.cutout_inside = cutout_inside

    def __call__(self, sample):
        ct_img = sample["ct"]
        ct_label = sample["ct_mask"]
        mri_img = sample["mri"]
        mri_label = sample["mri_mask"]
        cutout_ct_img, cutout_ct_label = cutout(ct_img, ct_label, self.mask_size, self.p, self.cutout_inside)
        cutout_mri_img, cutout_mri_label = cutout(mri_img, mri_label, self.mask_size, self.p, self.cutout_inside)
        return {"ct": cutout_ct_img, "mri": cutout_mri_img, "ct_mask": cutout_ct_label, "mri_mask": cutout_mri_label}


def cutout(image, label, mask_size, p, cutout_inside, mask_color=0):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    if np.random.random() > p:
        return image, label

    h, w, d = image.shape

    if cutout_inside:
        cxmin, cxmax = mask_size_half, w + offset - mask_size_half
        cymin, cymax = mask_size_half, h + offset - mask_size_half
        czmin, czmax = mask_size_half, d + offset - mask_size_half
    else:
        cxmin, cxmax = 0, w + offset
        cymin, cymax = 0, h + offset
        czmin, czmax = 0, d + offset

    cx = np.random.randint(cxmin, cxmax)
    cy = np.random.randint(cymin, cymax)
    cz = np.random.randint(czmin, czmax)
    xmin = cx - mask_size_half
    ymin = cy - mask_size_half
    zmin = cz - mask_size_half

    xmax = xmin + mask_size
    ymax = ymin + mask_size
    zmax = zmin + mask_size

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    zmin = max(0, zmin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)
    zmax = min(d, zmax)

    image[ymin:ymax, xmin:xmax, zmin:zmax] = mask_color
    # label[:, ymin:ymax, xmin:xmax, zmin:zmax] = mask_color   # one-hot
    label[ymin:ymax, xmin:xmax, zmin:zmax] = mask_color        # multi-classed
    return image, label


# def cutout(img, label, n_holes, length, h_rand, w_rand, d_rand):
#     h, w, d = img.shape
#     mask = np.ones((h, w, d), np.int32)
#     for n in range(n_holes):
#         x = int(h * h_rand[n])
#         y = int(w * w_rand[n])
#         z = int(d * d_rand[n])
#
#         x1 = int(np.clip(x - length / 2, 0, h))
#         x2 = int(np.clip(x + length / 2, 0, h))
#         y1 = int(np.clip(y - length / 2, 0, w))
#         y2 = int(np.clip(y + length / 2, 0, w))
#         z1 = int(np.clip(z - length / 2, 0, d))
#         z2 = int(np.clip(z + length / 2, 0, d))
#         mask[x1:x2, y1:y2, z1:z2] = 0
#     img_mask = mask[:, :, :]
#     label_mask = mask[None, :, :, :]
#     img = img * img_mask
#     label = label * label_mask
#     return img, label


# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""
#
#     def __call__(self, sample):
#         ct, mri = sample['ct'], sample['mri']
#         ct = ct.reshape(1, ct.shape[0], ct.shape[1], ct.shape[2]).astype(np.float32)
#         mri = mri.reshape(1, mri.shape[0], mri.shape[1], mri.shape[2]).astype(np.float32)
#         return {'ct': torch.from_numpy(ct),
#                 'mri': torch.from_numpy(mri),
#                 'ct_mask': torch.from_numpy(sample['ct_mask'].astype(np.float32)),
#                 'mri_mask': torch.from_numpy(sample['mri_mask'].astype(np.float32))}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return {'ct': torch.from_numpy(sample['ct'].astype(np.float32)),
                'mri': torch.from_numpy(sample['mri'].astype(np.float32)),
                'ct_mask': torch.from_numpy(sample['ct_mask'].astype(np.float32)),
                'mri_mask': torch.from_numpy(sample['mri_mask'].astype(np.float32))}

