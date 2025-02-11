import os
from lungmask import *
import SimpleITK as sitk
import numpy as np
# print(dir(lungmask.lungmask))

inferer = LMInferer(modelname="LTRCLobes")


img_path = ''
save_path = ''

img_lst = img_lst = os.listdir(img_path)


for img in img_lst:
    input_image = sitk.ReadImage(os.path.join(img_path, img))
    seg = inferer.apply(input_image)

    # Create a new array for the output with an extra channel
    output_seg = np.zeros((seg.shape[0], seg.shape[1], seg.shape[2], 2), dtype=seg.dtype)

    # Channel 0: Original labels (0-5)
    output_seg[..., 0] = seg

    # Channel 1: Combined labels
    background = seg == 0
    right_lung = (seg == 3) | (seg == 4) | (seg == 5)
    left_lung = (seg == 1) | (seg == 2)

    output_seg[..., 1][background] = 0
    output_seg[..., 1][right_lung] = 1
    output_seg[..., 1][left_lung] = 2

    # Create a new SimpleITK image object
    seg_image = sitk.GetImageFromArray(output_seg)

    # Copy spatial information from the original image to the segmented result
    seg_image.SetSpacing(input_image.GetSpacing())
    seg_image.SetOrigin(input_image.GetOrigin())
    seg_image.SetDirection(input_image.GetDirection())

    # Modify the file name to add "_seg"
    base_name = img.split('.')[0]
    ext = '.'.join(img.split('.')[1:])
    seg_save_name = base_name + "_seg." + ext

    # Save the segmented result with spatial information
    sitk.WriteImage(seg_image, os.path.join(save_path, seg_save_name))



