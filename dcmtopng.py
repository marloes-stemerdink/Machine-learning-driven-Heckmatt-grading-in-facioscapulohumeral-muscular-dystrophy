import os
import png
import pydicom as dicom
import argparse
import numpy as np


def mri_to_png(mri_file, png_file):
    """Convert from a DICOM image to PNG.

    Parameters
    ----------
    mri_file : file-like or str
        Open file object or path to the DICOM file.
    png_file : file-like
        Open binary file object to write the PNG data.
    """

    # Read DICOM and get pixel data as a NumPy array
    ds = dicom.dcmread(mri_file)
    img = ds.pixel_array.astype(np.float32)

    # Handle multi-frame / extra dimensions by squeezing
    img = np.squeeze(img)

    # Normalize to 0–255
    img_min = float(img.min())
    img_max = float(img.max())

    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        # All pixels equal; set to zeros
        img = np.zeros_like(img, dtype=np.float32)

    img = (img * 255.0).astype(np.uint8)

    # Ensure 2D grayscale. If more dims, take first channel/slice.
    if img.ndim > 2:
        img = img[..., 0]

    height, width = img.shape

    # png.Writer expects rows; note width, height order
    w = png.Writer(width, height, greyscale=True)
    w.write(png_file, img.tolist())


def convert_file(mri_file_path, png_file_path):
    """ Function to convert an MRI binary file to a
        PNG image file.

        @param mri_file_path: Full path to the mri file
        @param png_file_path: Fill path to the png file
    """

    # Making sure that the mri file exists
    if not os.path.exists(mri_file_path):
        raise Exception('File "%s" does not exists' % mri_file_path)

    # If the PNG file exists, overwrite it
    if os.path.exists(png_file_path):
        os.remove(png_file_path)

    # Open input (DICOM) and output (PNG) files
    with open(mri_file_path, "rb") as mri_file, open(png_file_path, "wb") as png_file:
        mri_to_png(mri_file, png_file)


def convert_folder(mri_folder, png_folder):
    """ Convert all MRI files in a folder to png files
        in a destination folder
    """

    # Create the folder for the pnd directory structure
    os.makedirs(png_folder)

    # Recursively traverse all sub-folders in the path
    for mri_sub_folder, subdirs, files in os.walk(mri_folder):
        for mri_file in os.listdir(mri_sub_folder):
            mri_file_path = os.path.join(mri_sub_folder, mri_file)

            # Make sure path is an actual file
            if os.path.isfile(mri_file_path):

                # Replicate the original file structure
                rel_path = os.path.relpath(mri_sub_folder, mri_folder)
                png_folder_path = os.path.join(png_folder, rel_path)
                if not os.path.exists(png_folder_path):
                    os.makedirs(png_folder_path)
                png_file_path = os.path.join(png_folder_path, '%s.png' % mri_file)

                try:
                    # Convert the actual file
                    convert_file(mri_file_path, png_file_path)
                    print ('SUCCESS>', mri_file_path, '-->', png_file_path)
                except Exception as e:
                    print ('FAIL>', mri_file_path, '-->', png_file_path, ':', e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a dicom MRI file to png")
    parser.add_argument('-f', action='store_true')
    parser.add_argument('dicom_path', help='Full path to the mri file')
    parser.add_argument('png_path', help='Full path to the generated png file')

    args = parser.parse_args()
    print (args)
    if args.f:
        convert_folder(args.dicom_path, args.png_path)
    else:
        convert_file(args.dicom_path, args.png_path)