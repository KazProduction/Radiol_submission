import os
import SimpleITK as sitk
import torch
from monai.networks.blocks import Warp
from monai.transforms import Compose, AddChannel, SpatialPad


# Function to normalize the image to the range [0, 1]
def min_max_normalize(image):
    min_val = image.min()
    max_val = image.max()
    return (image - min_val) / (max_val - min_val)


# Function to preprocess the CT image, including normalization and padding
def preprocess_image(image_path, pad_size=(128, 128, 128)):
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    image = min_max_normalize(image)
    transform = Compose([AddChannel(), SpatialPad(spatial_size=pad_size, method='symmetric')])
    image = transform(image)
    return image


# Function to apply the DDF to the CT image
def apply_ddf(moving_image, ddf_path, warp_layer):
    ddf = sitk.GetArrayFromImage(sitk.ReadImage(ddf_path))
    ddf = torch.tensor(ddf, dtype=torch.float32).unsqueeze(0).cuda()  # Add batch dimension
    moving_image_tensor = torch.tensor(moving_image, dtype=torch.float32).unsqueeze(0).cuda()  # Add batch dimension
    warped_image = warp_layer(moving_image_tensor, ddf)
    return warped_image.squeeze().cpu().numpy()


# Main function to iterate over all CT images, preprocess, apply DDF, and save the result
def main():
    # Paths for the original CT images and saved DDFs
    ct_dir = r''
    ddf_dir = r''
    output_dir = r''
    pad_size = (128, 128, 128)  # Desired image size after padding

    # Initialize MONAI's Warp layer
    warp_layer = Warp().cuda()

    # Iterate over all files in the CT directory
    for ct_filename in os.listdir(ct_dir):
        ct_path = os.path.join(ct_dir, ct_filename)
        ddf_filename = ct_filename[:15] + "_ddf.nii.gz"  # Assuming the naming convention for DDF files
        ddf_path = os.path.join(ddf_dir, ddf_filename)

        if os.path.exists(ddf_path):
            # Preprocess the CT image
            moving_image = preprocess_image(ct_path, pad_size)

            # Apply the DDF to the CT image
            warped_image_array = apply_ddf(moving_image, ddf_path, warp_layer)

            # Convert the warped image back to a SimpleITK image
            warped_image = sitk.GetImageFromArray(warped_image_array)
            spacing = (5.0, 5.0, 5.0)  # Set the spacing to (5.0, 5.0, 5.0)
            warped_image.SetSpacing(spacing)

            # Save the warped CT image
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            warped_ct_path = os.path.join(output_dir, ct_filename)
            sitk.WriteImage(warped_image, warped_ct_path)
            print(f"Saved warped CT image to: {warped_ct_path}")
        else:
            print(f"DDF not found for {ct_filename}, skipping...")


if __name__ == '__main__':
    main()
