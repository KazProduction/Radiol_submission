
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
import pandas as pd
import re


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

participant_results = []

def min_max_normalize(image):
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image


def preprocess_image(image_path, pad_size=(128, 128, 128), if_label=False, if_RBCTPmap=False):
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    if if_RBCTPmap:
        image = image.astype(np.float32)
    else:
        image = image.astype(np.float32)
    if if_label:
        image = torch.tensor(image).permute(3, 0, 1, 2)
        transform = Compose([SpatialPad(spatial_size=pad_size, method='symmetric')])
    else:
        transform = Compose([AddChannel(), SpatialPad(spatial_size=pad_size, method='symmetric')])
    image = transform(image)
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


    ct_label_channel_1 = ct_label_after_registration[1, :, :, :]
    ct_label_channel_1_mask = (ct_label_channel_1 == 0).astype(np.uint8)  # 反转，0变为1，1变为0
    ct_label_channel_1_mask = ct_label_channel_1_mask[ct_after_registration.shape[0] // 2, :, :]

    masked_rbctpmap = np.ma.masked_where(ct_label_channel_1_mask == 0, xe_after_registration_slice)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=16)


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
            label = int(label)
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


def calculate_rbctp_value(RBCmap, TPmap):
    total_RBC_intensity = np.sum(RBCmap)
    total_TP_intensity = np.sum(TPmap)
    rbctp_value = total_RBC_intensity / total_TP_intensity if total_TP_intensity != 0 else 0
    return rbctp_value


def calculate_region_rbctp_values(ct_lobe_seg_after_registration, RBCmap_after_registration, TPmap_after_registration):
    region_values = {}
    for channel in range(ct_lobe_seg_after_registration.shape[0]):
        unique_labels = np.unique(ct_lobe_seg_after_registration[channel])
        for label in unique_labels:
            if label == 0:
                continue
            mask = (ct_lobe_seg_after_registration[channel] == label)
            total_RBC_intensity = np.sum(RBCmap_after_registration[mask])
            total_TP_intensity = np.sum(TPmap_after_registration[mask])
            rbctp_value = total_RBC_intensity / total_TP_intensity if total_TP_intensity != 0 else 0
            label = int(label)
            if channel == 0:
                region_name = region_names_channel_0.get(label, f'Unknown Region {label}')
            else:
                region_name = region_names_channel_1.get(label, f'Unknown Region {label}')
            region_values[region_name] = rbctp_value
    return region_values

def calculate_lung_rbctp_means(ct_lobe_seg_after_registration, RBCmap_after_registration, TPmap_after_registration):
    left_lung_mask = (ct_lobe_seg_after_registration[1] == 2)
    right_lung_mask = (ct_lobe_seg_after_registration[1] == 1)

    total_RBC_left = np.sum(RBCmap_after_registration[left_lung_mask])
    total_TP_left = np.sum(TPmap_after_registration[left_lung_mask])
    mean_left_lung_rbctp = total_RBC_left / total_TP_left if total_TP_left != 0 else 0

    total_RBC_right = np.sum(RBCmap_after_registration[right_lung_mask])
    total_TP_right = np.sum(TPmap_after_registration[right_lung_mask])
    mean_right_lung_rbctp = total_RBC_right / total_TP_right if total_TP_right != 0 else 0

    total_RBC_whole_lung = np.sum(RBCmap_after_registration)
    total_TP_whole_lung = np.sum(TPmap_after_registration)
    mean_whole_lung_rbctp = total_RBC_whole_lung / total_TP_whole_lung if total_TP_whole_lung != 0 else 0

    return mean_left_lung_rbctp, mean_right_lung_rbctp, mean_whole_lung_rbctp

def process_group(group_name, group_path):
    group_results = []
    overall_rbctp_values_before = []
    overall_rbctp_values_after = []

    ct_path = os.path.join(group_path, 'ct')
    ct_lobe_seg_path = os.path.join(group_path, 'ct_lobe_mask_reorientation')
    rbc_path = os.path.join(group_path, 'rbc')
    tp_path = os.path.join(group_path, 'tp')
    ddf_ct_path = r''
    ddf_xe_path = r''

    patient_ids = []
    pattern = re.compile(r'exp(\d{3}_\d{3})_0_ct\.nii\.gz')

    for f in os.listdir(ct_path):
        match = pattern.match(f)
        if match:
            patient_ids.append(match.group(1))

    for patient_id in patient_ids:
        ct_file = os.path.join(ct_path, f'')
        ct_lobe_seg_file = os.path.join(ct_lobe_seg_path, f'')
        RBC_file = os.path.join(rbc_path, f'')
        TP_file = os.path.join(tp_path, f'')
        ddf_ct_file = os.path.join(ddf_ct_path, f'')
        ddf_xe_file = os.path.join(ddf_xe_path, f'')

        paths = {
            'CT file': ct_file,
            'CT lobe segmentation file': ct_lobe_seg_file,
            'RBC file': RBC_file,
            'TP file': TP_file,
            'DDF CT file': ddf_ct_file,
            'DDF XE file': ddf_xe_file
        }

        missing_paths = [name for name, path in paths.items() if not os.path.exists(path)]
        if missing_paths:
            print(f"Missing data for patient {patient_id} in group {group_name}: {', '.join(missing_paths)}")
            continue

        ct_lobe_seg = preprocess_image(ct_lobe_seg_file, if_label=True)
        RBCmap = preprocess_image(RBC_file, if_RBCTPmap=True)
        TPmap = preprocess_image(TP_file, if_RBCTPmap=True)

        # Before registration whole lung RBCTP mean calculation
        overall_rbctp_value_before = calculate_rbctp_value(RBCmap, TPmap)
        overall_rbctp_values_before.append(overall_rbctp_value_before)

        # Apply DDF for after registration
        ddf_ct = sitk.GetArrayFromImage(sitk.ReadImage(ddf_ct_file))
        ddf_xe = sitk.GetArrayFromImage(sitk.ReadImage(ddf_xe_file))

        warp_layer = Warp().cuda()
        warp_label_layer = Warp(mode="nearest").cuda()

        ct_lobe_seg_after_registration = apply_ddf(ct_lobe_seg, ddf_ct, warp_label_layer)
        RBCmap_after_registration = apply_ddf(RBCmap, ddf_xe, warp_layer)
        TPmap_after_registration = apply_ddf(TPmap, ddf_xe, warp_layer)

        # After registration whole lung RBCTP mean calculation
        overall_rbctp_value_after = calculate_rbctp_value(RBCmap_after_registration, TPmap_after_registration)
        overall_rbctp_values_after.append(overall_rbctp_value_after)

        # Region-wise RBCTP values (after registration)
        region_rbctp_values = calculate_region_rbctp_values(ct_lobe_seg_after_registration, RBCmap_after_registration,
                                                            TPmap_after_registration)


        participant_results.append({
            'Patient ID': patient_id,
            'Group': group_name,
            'Whole Lung Mean RBCTP Before': overall_rbctp_value_before,  # 添加before registration的mean RBCTP
            'Whole Lung Mean RBCTP After': overall_rbctp_value_after,    # 保留after registration的mean RBCTP
            **region_rbctp_values  #
        })

        group_results.append(region_rbctp_values)

    return group_results, participant_results


def aggregate_results(group_results, overall_before, overall_after):
    aggregated = {}
    for result in group_results:
        for region, value in result.items():
            if region not in aggregated:
                aggregated[region] = []
            aggregated[region].append(value)

    aggregated_mean_std = {}
    for region, values in aggregated.items():
        mean_value = np.mean(values)
        std_value = np.std(values)
        aggregated_mean_std[region] = (mean_value, std_value)

    overall_mean_before = np.mean(overall_before)
    overall_std_before = np.std(overall_before)
    overall_mean_after = np.mean(overall_after)
    overall_std_after = np.std(overall_after)
    aggregated_mean_std['Overall before registration'] = (overall_mean_before, overall_std_before)
    aggregated_mean_std['Overall after registration'] = (overall_mean_after, overall_std_after)

    return aggregated_mean_std




# base_path = r''
#
# main(base_path)

def main(base_path):
    all_results = {}
    groups = ['Healthy', 'noSOB+COVID', 'SOB+COVID']

    for group_name in groups:
        group_path = os.path.join(base_path, group_name)
        group_results, participant_results = process_group(group_name, group_path)
        all_results[group_name] = aggregate_results(group_results, [], [])

    df = pd.DataFrame.from_dict(all_results, orient='index')
    print(df)
    # df.to_csv('rbctp_values.csv')


    participant_df = pd.DataFrame(participant_results)
    participant_df.to_csv('participant_rbctp_values_before_after_updated.csv', index=False)



base_path = r''

main(base_path)

