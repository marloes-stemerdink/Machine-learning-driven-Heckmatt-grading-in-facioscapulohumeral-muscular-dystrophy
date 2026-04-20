import pydicom
import matplotlib.pyplot as plt
import os
import numpy as np

def load_dicom(file_path):
    """Load DICOM file and return dataset."""
    return pydicom.dcmread(file_path)

def extract_all_info(dicom_dataset):
    """Extract all available information from the DICOM dataset."""
    info = {}
    for elem in dicom_dataset:
        if elem.VR != "SQ":  # Exclude Sequences
            field_name = elem.name.replace(" ", "_")
            info[field_name] = str(elem.value)
    return info

def visualize_image(dicom_dataset):
    """Visualize the DICOM image."""
    plt.imshow(dicom_dataset.pixel_array, cmap=plt.cm.gray)
    plt.title("DICOM Image")
    plt.show()

def visualize_metadata(metadata):
    """Visualize metadata in a simple text format."""
    for key, value in metadata.items():
        print(f"{key}: {value}")

# def save_image_as_png(dicom_dataset, output_file_path):
#     """Save the DICOM image data as a PNG file."""
#     plt.imsave(output_file_path, dicom_dataset.pixel_array, cmap=plt.cm.gray)

# def convert_dicom_to_png(input_dir, output_dir):
#     """Recursively read all DICOM images in a directory and its subdirectories, and save them as PNG images in an output directory."""
#     for root, dirs, files in os.walk(input_dir):
#         for file in files:
#             if file.endswith(".dcm"):
#                 file_path = os.path.join(root, file)
#                 dicom_dataset = pydicom.dcmread(file_path)
#                 output_file_path = os.path.join(output_dir, file.replace(".dcm", ".png"))
#                 plt.imsave(output_file_path, dicom_dataset.pixel_array, cmap=plt.cm.gray)

def crop_image(image, top_crop_px=0, bottom_crop_px=0, left_crop_px=0, right_crop_px=0):
    """
    Crop irrelevant information from all sides of the image.

    Works for:
      - 2D arrays: (H, W)
      - 3D arrays: (H, W, C)  e.g. color images
    """
    if image.ndim == 2:
        h, w = image.shape
    elif image.ndim == 3:
        h, w, _ = image.shape
    else:
        raise ValueError(f"Unsupported image shape {image.shape}; expected 2D or 3D array.")

    top = max(0, top_crop_px)
    bottom = max(0, bottom_crop_px)
    left = max(0, left_crop_px)
    right = max(0, right_crop_px)

    if top + bottom >= h:
        raise ValueError("Cropping too much vertically; adjust top/bottom_crop_px.")
    if left + right >= w:
        raise ValueError("Cropping too much horizontally; adjust left/right_crop_px.")

    if image.ndim == 2:
        return image[top:h - bottom, left:w - right]
    else:  # 3D (H, W, C)
        return image[top:h - bottom, left:w - right, :]

def save_image_as_png(
    dicom_dataset,
    output_file_path,
    top_crop_px=0,
    bottom_crop_px=0,
    left_crop_px=0,
    right_crop_px=0,
):
    """Save the DICOM image data as a PNG file, with optional cropping from all sides."""
    img = dicom_dataset.pixel_array

    if any(v > 0 for v in (top_crop_px, bottom_crop_px, left_crop_px, right_crop_px)):
        img = crop_image(
            img,
            top_crop_px=top_crop_px,
            bottom_crop_px=bottom_crop_px,
            left_crop_px=left_crop_px,
            right_crop_px=right_crop_px,
        )

    plt.imsave(output_file_path, img, cmap=plt.cm.gray)

def convert_dicom_to_png(
    input_dir,
    output_dir,
    top_crop_px=0,
    bottom_crop_px=0,
    left_crop_px=0,
    right_crop_px=0,
):
    """
    Recursively read all DICOM images in a directory and its subdirectories,
    crop from all sides, and save them as PNG images in an output directory.
    """
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".dcm"):
                file_path = os.path.join(root, file)
                dicom_dataset = pydicom.dcmread(file_path)
                output_file_path = os.path.join(output_dir, file.replace(".dcm", ".png"))
                save_image_as_png(
                    dicom_dataset,
                    output_file_path,
                    top_crop_px=top_crop_px,
                    bottom_crop_px=bottom_crop_px,
                    left_crop_px=left_crop_px,
                    right_crop_px=right_crop_px,
                )

# Example Usage
# Only convert all your DICOMs in a folder to PNG (no single-image inspection)
if __name__ == "__main__":
    # # ---- STEP 1: inspect a single DICOM to decide crop values ----
    # # Pick one representative DICOM file:
    # sample_file = "/home/marloes.stemerdink@mydre.org/Documents/DCM_test/1.2.392.200036.9116.6.22.11522156.9942.20250502070844414.2.73.dcm"

    # ds = load_dicom(sample_file)
    # print("Image shape (height, width):", ds.pixel_array.shape)
    # visualize_image(ds)  # hover mouse to read x, y in the status bar

    # # After you run this once and write down:
    # #   - y_start, y_end (top/bottom useful rows)
    # #   - x_start, x_end (left/right useful columns)
    # # compute:
    # #
    # #   H, W = ds.pixel_array.shape
    # #   top_crop_px    = y_start
    # #   bottom_crop_px = H - 1 - y_end
    # #   left_crop_px   = x_start
    # #   right_crop_px  = W - 1 - x_end
    # #
    # # Then comment out the block above and uncomment the batch conversion below.

    # ---- STEP 2 (after you know the four crop values): batch convert ----
    input_dir = "/mnt/data/Visit1_cleaned/"
    output_dir = "/mnt/data/Visit1_PNG/"
    os.makedirs(output_dir, exist_ok=True)
    
    top_crop_px = 148
    bottom_crop_px = 216
    left_crop_px = 238
    right_crop_px = 244
    
    convert_dicom_to_png(
        input_dir,
        output_dir,
        top_crop_px=top_crop_px,
        bottom_crop_px=bottom_crop_px,
        left_crop_px=left_crop_px,
        right_crop_px=right_crop_px,
    )

# # use plotly to visualize the dicom image
# import plotly.express as px
# import plotly.figure_factory as ff
# import plotly.graph_objects as go

# import numpy as np
# from PIL import Image
# import os
# from tqdm import tqdm

# def count_transitions(image_path):
#     # Load the png image
#     image = Image.open(image_path)
#     image = np.array(image)

#     # Take the last column of the image in only one channel
#     image = image[:, :, 0]
#     last_column = image[:, -1] > 100

#     # Find differences between adjacent elements
#     transitions = (last_column[:-1] == True) & (last_column[1:] == False)
#     count = np.sum(transitions)

#     # measure distance between first and second transition
#     first_transition = np.where(transitions)[0][0]
#     second_transition = np.where(transitions)[0][1]
#     distance = second_transition - first_transition
#     cf = 1 / distance

#     return count, distance, cf

# # now make a loop that counts the transitions for all the images in the dataset and stores the results in a dataframe   
# import os
# import pandas as pd

# root = '/media/francesco/DEV001/PROJECT-FSHD/DATA/converted_dicom'
# df = pd.DataFrame(columns=['image', 'transitions', 'distance', 'cf'])

# for file in tqdm(os.listdir(root)):
#     image_path = os.path.join(root, file)
#     transitions, distance, cf = count_transitions(image_path)
#     df.loc[len(df)] = {'image': file, 'transitions': transitions, 'distance': distance, 'cf': cf}

# df.to_csv('/media/francesco/DEV001/PROJECT-FSHD/DATA/TABULAR/conversion_factor.csv', index=False)
# df.to_excel('/media/francesco/DEV001/PROJECT-FSHD/DATA/TABULAR/conversion_factor.xlsx', index=False)

# # plot image with plotly
# import plotly.express as px
# import plotly.figure_factory as ff
# import plotly.graph_objects as go

# import numpy as np
# from PIL import Image
# import os

# image_path = '/media/francesco/DEV001/PROJECT-FSHD/DATA/converted_dicom/anon_2320_Depressor anguli oris_L_3.png'
# image = Image.open(image_path)
# image = np.array(image)

# fig = px.imshow(image)
# fig.show()
