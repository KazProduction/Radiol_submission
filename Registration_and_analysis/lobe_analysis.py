
'''
Channel 0 has labels: [0. 1. 2. 3. 4. 5.]
Channel 1 has labels: [0. 1. 2.]
'''

import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import SpatialPad, Compose, AddChannel,EnsureChannelFirst
import torch
from monai.networks.blocks import Warp


# Define the names for the regions in channel 0
region_names_channel_0 = {
    1: 'Left Upper Lobe',
    2: 'Left Lower Lobe',
    3: 'Right Upper Lobe',
    4: 'Right Middle Lobe',
    5: 'Right Lower Lobe'
}

# Define the names for the regions in channel 1
region_names_channel_1 = {
    1: 'Right Lung',
    2: 'Left Lung'
}

# def visualize_lobe_segmentation(ct_lobe_seg_path):#
#     file_name = os.listdir(ct_lobe_seg_path)[0]
#     file_path = os.path.join(ct_lobe_seg_path, file_name)
#
#
#     image = sitk.ReadImage(file_path)
#     image_array = sitk.GetArrayFromImage(image)
#
#
#     if image_array.ndim == 4:
#         num_channels = image_array.shape[-1]
#         print(f'Number of channels in {file_name}: {num_channels}')
#
#
#         fig, axes = plt.subplots(2, 6, figsize=(18, 6))
#
#
#         channel_0_data = image_array[:, :, :, 0]
#         unique_labels_0 = np.unique(channel_0_data)
#         print(f'Channel 0 has labels: {unique_labels_0}')
#
#         for i, label in enumerate(unique_labels_0):
#             label_mask = (channel_0_data == label)
#             mid_slice = label_mask[label_mask.shape[0] // 2, :, :]
#             ax = axes[0, i]
#             ax.imshow(mid_slice, cmap='gray')
#             ax.set_title(f'Channel 0 - Label {label}')
#             ax.axis('off')
#
#
#         channel_1_data = image_array[:, :, :, 1]
#         unique_labels_1 = np.unique(channel_1_data)
#         print(f'Channel 1 has labels: {unique_labels_1}')
#
#         for i, label in enumerate(unique_labels_1):
#             if i < 3:
#                 label_mask = (channel_1_data == label)
#                 mid_slice = label_mask[label_mask.shape[0] // 2, :, :]
#                 ax = axes[1, i]
#                 ax.imshow(mid_slice, cmap='gray')
#                 ax.set_title(f'Channel 1 - Label {label}')
#                 ax.axis('off')
#
#
#         for j in range(len(unique_labels_0), 6):
#             axes[0, j].axis('off')
#         for j in range(3, 6):
#             axes[1, j].axis('off')
#
#     else:
#         print(f'The image is not 4-dimensional, its shape is: {image_array.shape}')
#
#     plt.show()


# def min_max_normalize(image):
#     image = image.astype(np.float32)
#     min_val = np.min(image)
#     max_val = np.max(image)
#     normalized_image = (image - min_val) / (max_val - min_val)
#     return normalized_image


def preprocess_image(image_path, pad_size=(128, 128, 128), if_label=False, if_RBCTPmap=False):
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    if if_RBCTPmap:
        image = image.astype(np.float32)
        image[(image < 0.05) | (image > 0.8)] = 0

    else:
        # image = min_max_normalize(image)
        image = image.astype(np.float32)
    if if_label:
        image = torch.tensor(image).permute(3, 0, 1, 2)
        # transform = Compose([EnsureChannelFirst(), SpatialPad(spatial_size=pad_size, method='symmetric')])
        transform = Compose([SpatialPad(spatial_size=pad_size, method='symmetric')])
    else:
        transform = Compose([AddChannel(), SpatialPad(spatial_size=pad_size, method='symmetric')])
    image = transform(image)
    # print(image.shape)
    return image

def apply_ddf(moving_image, ddf, warp_layer):
    ddf = torch.tensor(ddf).unsqueeze(0).cuda()
    moving_image_tensor = torch.tensor(moving_image).unsqueeze(0).cuda()
    warped_image = warp_layer(moving_image_tensor, ddf)
    return warped_image.squeeze().cpu().numpy()

def apply_window(image, window_center, window_width):
    """
    Apply windowing to the image.

    Args:
    - image (np.array): The input normalized image (values between 0 and 1).
    - window_center (float): The window center (level), also in the range [0, 1].
    - window_width (float): The window width, also in the range [0, 1].

    Returns:
    - windowed_image (np.array): The windowed image.
    """
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    windowed_image = np.clip(image, min_val, max_val)
    windowed_image = (windowed_image - min_val) / (max_val - min_val)
    windowed_image = np.clip(windowed_image, 0, 1)  # Ensure the output is in [0, 1]
    return windowed_image


def visualize_and_save(ct_after_registration, xe_after_registration, ct_label_after_registration, title, save_path):
    # Windowing parameters
    # window_center = 0.56
    # window_width = 0.35
    #
    # ct_after_registration = apply_window(ct_after_registration, window_center, window_width)

    ct_after_registration_slice = ct_after_registration[ct_after_registration.shape[0] // 2, :, :]
    xe_after_registration_slice = xe_after_registration[xe_after_registration.shape[0] // 2, :, :]

    # 获取 ct_label_after_registration 的标签
    ct_label_channel_1 = ct_label_after_registration[1, :, :, :]
    ct_label_channel_1_mask = (ct_label_channel_1 == 0).astype(np.uint8)  # 反转，0变为1，1变为0
    ct_label_channel_1_mask = ct_label_channel_1_mask[ct_after_registration.shape[0] // 2, :, :]

    masked_rbctpmap = np.ma.masked_where(ct_label_channel_1_mask == 0, xe_after_registration_slice)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=16)

    # 显示叠加图像
    axes[0].imshow(ct_after_registration_slice, cmap="gray")
    # axes[0].imshow(xe_after_registration_slice, cmap="hot", alpha=1.0)
    axes[0].imshow(masked_rbctpmap, cmap="hot", alpha=1.0)
    axes[0].set_title("Overlay Before Registration", fontsize=14)
    axes[0].axis("off")

    # Overlay after registration
    axes[1].imshow(ct_after_registration_slice, cmap="gray")
    # axes[1].imshow(ct_label_channel_1_mask, cmap="jet", alpha=0.3)
    axes[1].imshow(masked_rbctpmap, cmap="hot", alpha=0.5)
    axes[1].set_title("Overlay After Registration", fontsize=14)
    axes[1].axis("off")

    plt.subplots_adjust(top=0.85, wspace=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # plt.savefig(save_path)
    plt.show()

def calculate_region_rbctp_means(ct_lobe_seg_after_registration, RBCTPmap_after_registration):
    region_means = {}
    for channel in range(ct_lobe_seg_after_registration.shape[0]):
        unique_labels = np.unique(ct_lobe_seg_after_registration[channel])
        # print(f"Channel {channel} unique labels: {unique_labels}")
        for label in unique_labels:
            if label == 0:
                continue
            mask = (ct_lobe_seg_after_registration[channel] == label)
            region_values = RBCTPmap_after_registration[mask]
            mean_value = np.mean(region_values[region_values > 0])
            std_value = np.std(region_values[region_values > 0])
            label = int(label)  # 确保标签为整数
            if channel == 0:
                region_name = region_names_channel_0.get(label, f'Unknown Region {label}')
            else:
                region_name = region_names_channel_1.get(label, f'Unknown Region {label}')
            # print(f"Channel {channel}, Label {label}, Region {region_name}, Mean Value: {mean_value}, Std Value: {std_value}")
            region_means[region_name] = (mean_value, std_value)
    return region_means

def calculate_overall_rbctp_mean(ct_lobe_seg, RBCTPmap):
    mask = ct_lobe_seg[1] > 0
    if RBCTPmap.ndim == 4:
        RBCTPmap = RBCTPmap.squeeze(0)
    non_zero_values = RBCTPmap[mask]
    non_zero_values = non_zero_values[non_zero_values > 0]
    overall_mean = np.mean(non_zero_values)
    overall_std = np.std(non_zero_values)
    return overall_mean, overall_std

def calculate_overall_rbctp_mean_without_mask(RBCTPmap):
    non_zero_values = RBCTPmap[RBCTPmap > 0]
    overall_mean = np.mean(non_zero_values)
    overall_std = np.std(non_zero_values)
    return overall_mean, overall_std



original_id = "001_043"
original_id = "001_016"
original_id = "001_040"
original_id = "004_006"

base_path = ''
ct_path = f''
ct_lobe_seg_path = f''
RBCTPmap_path = f''
ddf_ct_path = f''
ddf_xe_path = f''


# tesing the label
# ct_lobe_seg_path = 'Dataset\highresCT\isotropic_label'
# visualize_lobe_segmentation(ct_lobe_seg_path)

# Load data
ct_img = preprocess_image(ct_path)
ct_lobe_seg = preprocess_image(ct_lobe_seg_path, if_label=True)
RBCTPmap = preprocess_image(RBCTPmap_path, if_RBCTPmap=True)
ddf_ct = sitk.GetArrayFromImage(sitk.ReadImage(ddf_ct_path))
ddf_xe = sitk.GetArrayFromImage(sitk.ReadImage(ddf_xe_path))

warp_layer = Warp().cuda()
warp_label_layer = Warp(mode="nearest").cuda()


overall_mean_before, overall_std_before = calculate_overall_rbctp_mean_without_mask(RBCTPmap)

ct_after_registration = apply_ddf(ct_img, ddf_ct, warp_layer)
ct_lobe_seg_after_registration = apply_ddf(ct_lobe_seg, ddf_ct, warp_label_layer)

RBCTPmap_after_registration = apply_ddf(RBCTPmap, ddf_xe, warp_layer)

# save_path = f'results\\overlay_{original_id}.png'
# visualize_and_save(ct_after_registration, RBCTPmap_after_registration, ct_lobe_seg_after_registration, "Overlay Results", save_path)

'''calculate RBCTP for lung regions'''

region_means = calculate_region_rbctp_means(ct_lobe_seg_after_registration, RBCTPmap_after_registration)


overall_mean_after, overall_std_after = calculate_overall_rbctp_mean(ct_lobe_seg_after_registration, RBCTPmap_after_registration)



for region, (mean_value, std_value) in region_means.items():
    print(f'{region}: {mean_value:.4f} ± {std_value:.4f}')
# print(f'Overall RBC TP before registration: {overall_mean_before:.4f} ± {overall_std_before:.4f}')
print(f'Overall RBC TP after registration: {overall_mean_after:.4f} ± {overall_std_after:.4f}')

