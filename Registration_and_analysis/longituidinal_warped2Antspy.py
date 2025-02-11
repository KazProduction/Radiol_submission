import os
import torch
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import SpatialPad, Compose, AddChannel
from monai.networks.blocks import Warp
from utils import *

def load_data(ct_path, xe_path, ddf_ct2mr_path, ddf_xe2ct_path):
    ct_list = []
    xe_list = []
    ddf_ct2mr_list = []
    ddf_xe2ct_list = []
    for ct_file, xe_file, ddf_ct2mr_file, ddf_xe2ct_file in zip(os.listdir(ct_path),
                                                                os.listdir(xe_path),
                                                                os.listdir(ddf_ct2mr_path),
                                                                os.listdir(ddf_xe2ct_path)):
        ct_list.append(os.path.join(ct_path, ct_file))
        xe_list.append(os.path.join(xe_path, xe_file))
        ddf_ct2mr_list.append(os.path.join(ddf_ct2mr_path, ddf_ct2mr_file))
        ddf_xe2ct_list.append(os.path.join(ddf_xe2ct_path, ddf_xe2ct_file))

    return np.array(ct_list), np.array(xe_list), np.array(ddf_ct2mr_list), np.array(ddf_xe2ct_list)

def min_max_normalize(image):
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image

def preprocess_image(image_path, pad_size=(128, 128, 128)):
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    # print(image.shape)
    image = min_max_normalize(image)
    transform = Compose([AddChannel(), SpatialPad(spatial_size=pad_size, method='symmetric')])
    image = transform(image)
    return image

# def save_image_as_nifti(image_tensor, save_path):
#     # image_np = image_tensor.squeeze().numpy()
#     image_np = image_tensor
#     image_sitk = sitk.GetImageFromArray(image_np)
#     image_sitk.SetSpacing((5, 5, 5))
#     sitk.WriteImage(image_sitk, save_path)

def save_image_as_nifti(image_tensor, save_path):
    if isinstance(image_tensor, torch.Tensor):
        image_np = image_tensor.squeeze().numpy()
    elif isinstance(image_tensor, np.ndarray):
        image_np = image_tensor
    else:
        raise TypeError("image_tensor must be a PyTorch tensor or a NumPy array")

    image_sitk = sitk.GetImageFromArray(image_np)
    image_sitk.SetSpacing((5, 5, 5))
    sitk.WriteImage(image_sitk, save_path)


def preprocess_ddf(ddf, pad_size=(128, 128, 128)):
    ddf = torch.tensor(ddf, dtype=torch.float32).permute(3, 0, 1, 2) # Shape: [3, D, H, W]
    # ddf = torch.flip(ddf, [2])
    # ddf = ddf. permute(3, 0, 1, 2)
    print(ddf.shape)
    pad_transform = SpatialPad(spatial_size=pad_size, method='symmetric')
    ddf_padded = pad_transform(ddf).unsqueeze(0)
    return ddf_padded.cuda()

def ddf_reconstruction(ddf_xe_1_path, transform_path, warp_layer):
    ddf_xe_1 = sitk.GetArrayFromImage(sitk.ReadImage(ddf_xe_1_path))
    transform = sitk.GetArrayFromImage(sitk.ReadImage(transform_path))
    ddf_xe_1_tensor = torch.tensor(ddf_xe_1, dtype=torch.float32).unsqueeze(0).cuda()
    transform_tensor = preprocess_ddf(transform)
    ddf_combined = warp_layer(ddf_xe_1_tensor, transform_tensor) + transform_tensor
    return ddf_combined.squeeze(0)

def apply_ddf(moving_image, ddf, warp_layer):
    ddf = torch.tensor(ddf).unsqueeze(0).cuda()
    moving_image_tensor = torch.tensor(moving_image).unsqueeze(0).cuda()
    warped_image = warp_layer(moving_image_tensor, ddf)
    return warped_image.squeeze().cpu().numpy()


def visualize_and_save(ct_before_registration, xe_before_registration, ct_after_registration, xe_after_registration, title, save_path):
    ct_before_registration_slice = ct_before_registration.squeeze()[ct_before_registration.shape[2] // 2, :, :]
    xe_before_registration_slice = xe_before_registration.squeeze()[xe_before_registration.shape[2] // 2, :, :]
    ct_after_registration_slice = ct_after_registration.squeeze()[ct_after_registration.shape[2] // 2, :, :]
    xe_after_registration_slice = xe_after_registration.squeeze()[xe_after_registration.shape[2] // 2, :, :]

    # Apply masking to filter out very small values in XE images
    xe_before_registration_masked = np.ma.masked_where(xe_before_registration_slice < 0.2, xe_before_registration_slice)
    xe_after_registration_masked = np.ma.masked_where(xe_after_registration_slice < 0.2, xe_after_registration_slice)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=16)

    # Overlay before registration
    axes[0].imshow(ct_before_registration_slice, cmap="gray")
    axes[0].imshow(xe_before_registration_masked, cmap="hot", alpha=0.5)
    axes[0].set_title("Overlay Before Registration", fontsize=14)
    axes[0].axis("off")

    # Overlay after registration
    axes[1].imshow(ct_after_registration_slice, cmap="gray")
    axes[1].imshow(xe_after_registration_masked, cmap="hot", alpha=0.5)
    axes[1].set_title("Overlay After Registration", fontsize=14)
    axes[1].axis("off")

    plt.subplots_adjust(top=0.85, wspace=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # plt.savefig(save_path)
    plt.show()

# Paths
original_id = ""
base_path = ''

ct_0_path = f''
ct_1_path = f''
ct_0_label_path = f''
ct_1_label_path = f''

xe_0_path = f''
xe_1_path = f''

ddf_xe_0_path = f''
ddf_xe_1_path = f''
ddf_ct_0_path = f''
ddf_ct_1_path = f''

save_ct_0_padd_path = f''
save_ct_0_label_padd_path = f''
save_ct_1_padd_path = f''
save_ct_1_label_padd_path = f''
save_xe_0_padd_path = f''

save_ct_0_path = f''
save_ct_0_label_path = f''
save_ct_1_path = f''
save_ct_1_label_path = f''
save_xe_1_warped_path = f''

# Load data and Padding
ct_0_image = preprocess_image(ct_0_path)
ct_1_image = preprocess_image(ct_1_path)
ct_0_label = preprocess_image(ct_0_label_path)
ct_1_label = preprocess_image(ct_1_label_path)
xe_0_image = preprocess_image(xe_0_path)
xe_1_image = preprocess_image(xe_1_path)

# Save padd data
save_image_as_nifti(ct_0_image, save_ct_0_padd_path)
save_image_as_nifti(ct_0_label, save_ct_0_label_padd_path)
save_image_as_nifti(ct_1_image, save_ct_1_padd_path)
save_image_as_nifti(ct_1_label, save_ct_1_label_padd_path)
save_image_as_nifti(xe_0_image, save_xe_0_padd_path)


ddf_xe_0 = sitk.GetArrayFromImage(sitk.ReadImage(ddf_xe_0_path))
ddf_xe_1 = sitk.GetArrayFromImage(sitk.ReadImage(ddf_xe_1_path))
ddf_ct_0 = sitk.GetArrayFromImage(sitk.ReadImage(ddf_ct_0_path))
ddf_ct_1 = sitk.GetArrayFromImage(sitk.ReadImage(ddf_ct_1_path))


# Initialize warp layer
warp_layer = Warp().cuda()
warp_label_layer = Warp(mode="nearest").cuda()

ct_0_after_registration = apply_ddf(ct_0_image, ddf_ct_0, warp_layer)
ct_1_after_registration = apply_ddf(ct_1_image, ddf_ct_1, warp_layer)
ct_0_label_after_registration = apply_ddf(ct_0_label, ddf_ct_0, warp_label_layer)
ct_1_label_after_registration = apply_ddf(ct_1_label, ddf_ct_1, warp_label_layer)

xe_1_after_registration = apply_ddf(xe_1_image, ddf_xe_1, warp_layer)

save_image_as_nifti(ct_0_after_registration, save_ct_0_path)
save_image_as_nifti(ct_0_label_after_registration, save_ct_0_label_path)
save_image_as_nifti(ct_1_after_registration, save_ct_1_path)
save_image_as_nifti(ct_1_label_after_registration, save_ct_1_label_path)
save_image_as_nifti(xe_1_after_registration, save_xe_1_warped_path)



