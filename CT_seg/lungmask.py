import os
from lungmask import *
import SimpleITK as sitk
import numpy as np


inferer = LMInferer(modelname="R231CovidWeb")

img_path = ''
save_path = ''

img_lst = img_lst = os.listdir(img_path)

for img in img_lst:
    input_image = sitk.ReadImage(os.path.join(img_path, img))
    seg = inferer.apply(input_image)

    seg[seg > 0] = 1

    seg_image = sitk.GetImageFromArray(seg)


    seg_image.SetSpacing(input_image.GetSpacing())
    seg_image.SetOrigin(input_image.GetOrigin())
    seg_image.SetDirection(input_image.GetDirection())


    base_name = img.split('.')[0]
    ext = '.'.join(img.split('.')[1:])
    seg_save_name = base_name + "_seg." + ext


    sitk.WriteImage(seg_image, os.path.join(save_path, seg_save_name))



