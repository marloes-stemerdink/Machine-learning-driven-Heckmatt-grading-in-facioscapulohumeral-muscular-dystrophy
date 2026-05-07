import numpy as np
from PIL import Image
from skimage import morphology
import os
import matplotlib.colors as mcolors

from tqdm import tqdm

from radiomics import featureextractor
import logging

import pandas as pd
import cv2
from scipy.ndimage import label
from scipy.spatial import ConvexHull
import SimpleITK as sitk  # Import SimpleITK for in-memory image handling

import multiprocessing
from functools import partial

def retain_largest_object(mask):
    labeled, num = label(mask)
    sizes = np.bincount(labeled.ravel())

    if len(sizes) > 1:  # Ensure there are labels other than the background
        max_label = np.argmax(sizes[1:]) + 1  # Ignore background label
        mask = (labeled == max_label)

    return mask


def postProcessNetworkOutput(pred, class_labels, classes, muscle):
    """
    Process network output to clean up segmentation masks.
    
    The muscle argument is passed in directly so class_pred is always
    set correctly, instead of inferring it from pixel values (which
    caused everything to be labelled as Biceps_brachii).
    """

    unique_labels, counts = np.unique(pred, return_counts=True)
    labels = unique_labels[unique_labels > 0]

    if len(labels) == 0:
        return pred, muscle  # no foreground found, return muscle name directly

    # Remove disconnected blobs that don't belong to dominant label
    pred_binary = pred > 0
    labeled_array, num_features = label(pred_binary)

    if num_features > 1:
        sizes = np.bincount(labeled_array.ravel())
        sizes[0] = 0  # Background size
        max_label = np.argmax(sizes)

        biggest_object = (labeled_array == max_label)
        other_objects = pred_binary & (~biggest_object)

        selem = morphology.disk(7)
        dilated_biggest = morphology.dilation(biggest_object, selem)
        overlap = dilated_biggest & other_objects
        overlap_pct = np.sum(overlap) / (np.sum(dilated_biggest) + np.finfo(float).eps)

        if overlap_pct <= 0.1:
            pred[~biggest_object] = 0

    unique_labels, counts = np.unique(pred, return_counts=True)
    labels = unique_labels[unique_labels > 0]

    if len(labels) == 0:
        return pred, muscle

    if len(labels) > 1:
        oh_pred = (np.arange(len(class_labels)+1) == pred[..., None]).astype(int) > 0
        oh_pred_original = oh_pred.copy()

        for i in labels:
            selem = morphology.disk(2)
            oh_pred[..., i] = morphology.dilation(oh_pred[..., i], selem)
            oh_pred[..., i] = morphology.remove_small_holes(oh_pred[..., i], 100000)
            msk_cv = oh_pred[..., i].astype(np.uint8)
            contours, _ = cv2.findContours(msk_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours and len(contours) > 0:
                all_points = np.vstack(contours).squeeze()
                if len(all_points.shape) == 2 and all_points.shape[0] >= 3:
                    hull = ConvexHull(all_points)
                    hull_points = all_points[hull.vertices]
                    blank_mask = np.zeros(oh_pred[..., i].shape, dtype=np.uint8)
                    oh_pred[..., i] = cv2.fillPoly(blank_mask, pts=[hull_points], color=(255)) > 0

        for i in labels:
            mask1 = oh_pred[..., i]
            sum_mask1 = oh_pred_original[..., i].sum()

            for j in labels:
                if j > i:
                    mask2 = oh_pred[..., j]
                    sum_mask2 = oh_pred_original[..., j].sum()
                    intersection = np.logical_and(mask1, mask2).sum()
                    union = np.logical_or(mask1, mask2).sum()
                    iou = intersection / union if union != 0 else 0

                    if iou > 0.6:
                        if sum_mask1 > sum_mask2:
                            pred[pred == j] = i
                        else:
                            pred[pred == i] = j

        # Re-compute after merging labels
        oh_pred = (np.arange(len(class_labels)+1) == pred[..., None]).astype(int) > 0
        unique_labels, counts = np.unique(pred, return_counts=True)
        labels = unique_labels[unique_labels > 0]

        for i in labels:
            selem = morphology.disk(2)
            oh_pred[..., i] = morphology.dilation(oh_pred[..., i], selem)
            oh_pred[..., i] = morphology.remove_small_holes(oh_pred[..., i], 100000)
            oh_pred[..., i] = retain_largest_object(oh_pred[..., i])
            oh_pred[..., i] = morphology.erosion(oh_pred[..., i], selem)

        pred_out = np.argmax(oh_pred, axis=-1)

    else:
        dominant_label = labels[0]
        seg_map = (pred > 0).astype(np.uint8)
        selem = morphology.disk(3)
        seg_map = morphology.dilation(morphology.erosion(seg_map, selem), selem)
        seg_map = morphology.opening(seg_map, selem)
        seg_map = morphology.closing(seg_map, selem)
        seg_map = retain_largest_object(seg_map)
        pred_out = seg_map * dominant_label

    # Always use the muscle argument for class_pred, not pixel values
    class_pred = muscle

    return pred_out, class_pred


def process_file(file, fold, pred_fold, img_fold, muscle, classes, class_labels):
    import numpy as np
    import cv2
    from scipy.ndimage import label
    import SimpleITK as sitk
    from radiomics import featureextractor
    import os
    from PIL import Image

    # ------------------------------------------------------------------ #
    # Skip thickness ("t") images — only echogenicity images 1, 2, 3 are used  #
    # ------------------------------------------------------------------ #
    slice_id = file.replace('.png', '').split('_')[3] if len(file.replace('.png', '').split('_')) >= 4 else ''
    if slice_id.lower() == 't':
        return []

    temp = dict()
    temp['Fold'] = fold
    temp['Muscle'] = muscle  # Add current muscle to the summary

    # Load image and prediction only
    img_PIL = Image.open(os.path.join(img_fold, file))
    pred_PIL = Image.open(os.path.join(pred_fold, file))

    img = np.array(img_PIL)

    # if img has 3 channels, keep only the first channel
    if len(img.shape) == 3:
        img = img[..., 0]

    pred = np.array(pred_PIL)

    # Ensure all arrays have the same shape
    if img.shape != pred.shape:
        min_height = min(img.shape[0], pred.shape[0])
        min_width = min(img.shape[1], pred.shape[1])
        img = img[:min_height, :min_width]
        pred = pred[:min_height, :min_width]

    # Pass muscle so class_pred is set correctly inside the function
    pred_out, class_pred = postProcessNetworkOutput(pred, class_labels, classes, muscle)

    # Check number of labels in pred_out
    unique_labels_pred, counts_pred = np.unique(pred_out, return_counts=True)

        # Initialize SimpleITK images
    if len(img.shape) == 3:
        img = img[:, :, 0]
    sitk_image = sitk.GetImageFromArray(img)
    sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)

    # Set consistent spacing, origin, and direction
    spacing = [1.0, 1.0]
    origin = [0.0, 0.0]
    direction = [1.0, 0.0, 0.0, 1.0]  # 2D identity matrix flattened

    sitk_image.SetSpacing(spacing)
    sitk_image.SetOrigin(origin)
    sitk_image.SetDirection(direction)

    # Create feature extractor inside the function to avoid pickling issues
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName('Original')
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('glszm')
    extractor.enableFeatureClassByName('gldm')
    extractor.enableFeatureClassByName('ngtdm')
    extractor.settings['additionalInfo'] = False

    # For each label in the prediction
    results = []
    for i in unique_labels_pred:
        if i == 0:
            continue  # skip background

                # Store values
        temp_copy = temp.copy()
        temp_copy['File'] = file
        temp_copy['slice'] = slice_id          # stored explicitly for easy filtering later
        temp_copy['subject'] = file.split('_')[0]
        temp_copy['muscle_code'] = file.split('_')[1]
        temp_copy['side'] = file.split('_')[2]
        temp_copy['class_pred'] = class_pred       # kept for reference

                # Prepare predicted mask
        pred_out_copy = pred_out.copy()
        pred_out_copy[pred_out_copy != i] = 0
        pred_bw = pred_out_copy > 0
        pred_mask = (pred_bw * 255).astype(np.uint8)

        sitk_pred_mask = sitk.GetImageFromArray(pred_mask)
        sitk_pred_mask.SetSpacing(spacing)
        sitk_pred_mask.SetOrigin(origin)
        sitk_pred_mask.SetDirection(direction)

        # Features inside predicted mask
        try:
            features = extractor.execute(sitk_image, sitk_pred_mask, label=255)
            temp_copy['features_img_pred'] = {str(key): str(value) for key, value in features.items()}
        except Exception as e:
            temp_copy['features_img_pred'] = 'mask not found'

        # Features outside predicted mask (negativeregion)
        try:
            kernel_size = 20
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            # Prepare negative predicted mask
            dilated_mask_pred = cv2.dilate(pred.astype(np.uint8), kernel, iterations=1)
            neg_seg_map_pred = np.logical_not(dilated_mask_pred).astype(np.uint8) * 255

            border_width = int(0.15 * min(pred.shape[:2]))
            offset = int(0.10 * min(pred.shape[:2]))
            neg_seg_map_pred[:border_width, :] = 0
            neg_seg_map_pred[-border_width:, :] = 0
            neg_seg_map_pred[:, :border_width] = 0
            neg_seg_map_pred[:, -border_width:] = 0

            labeled_array_pred, num_features_pred = label(pred_bw)
            if labeled_array_pred.size == 0 or num_features_pred == 0:
                object_centroid_pred = [0, 0]
            else:
                coords_pred = np.column_stack(np.where(labeled_array_pred > 0))
                if coords_pred.size == 0:
                    object_centroid_pred = [0, 0]
                else:
                    object_centroid_pred = coords_pred.mean(axis=0)

            neg_seg_map_pred[:int(object_centroid_pred[0] + offset), :] = 0

            sitk_neg_pred_mask = sitk.GetImageFromArray(neg_seg_map_pred)
            sitk_neg_pred_mask.SetSpacing(spacing)
            sitk_neg_pred_mask.SetOrigin(origin)
            sitk_neg_pred_mask.SetDirection(direction)

            features = extractor.execute(sitk_image, sitk_neg_pred_mask, label=255)
            temp_copy['features_img_pred_not'] = {str(key): str(value) for key, value in features.items()}
        except Exception:
            temp_copy['features_img_pred_not'] = 'mask not found'

        results.append(temp_copy)
    return results


# Prepare parameters for feature extraction
logger = logging.getLogger("radiomics")
logger.setLevel(logging.ERROR)

# Define base preds_dirs with a placeholder for muscle name
base_preds_dir_template = "/mnt/data/Visit1_segmentation/musclespecific/{muscle}/pred"
image_dir = "/mnt/data/Visit1_PNG/"

net = 'knet_swin_mod'
experiment = 'muscle_specific'

# Define class and palette for better visualization
classes = [
    'background',
    'Biceps_brachii',
    'Deltoideus',
    'Depressor_anguli_oris',
    'Digastricus',
    'Gastrocnemius_medial_head',
    'Geniohyoideus',
    'Masseter',
    'Mentalis',
    'Orbicularis_oris',
    'Rectus_abdominis',
    'Rectus_femoris',
    'Temporalis',
    'Tibialis_anterior',
    'Trapezius',
    'Vastus_lateralis',
    'Zygomaticus'
]

# Exclude 'background' from the list
muscle_names = classes[1:]
# Convert the list into a single string, separated by comma and space
muscle_names_str = ', '.join(muscle_names)
print("Muscles to be processed:", muscle_names_str)

class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

color_names = [
    'black', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white',
    'purple', 'lime', 'teal', 'navy', 'grey', 'maroon', 'olive', 'brown',
    'coral'
]

cmap_muscle = mcolors.ListedColormap(color_names)

# Initialize summary list to collect results from all muscles
summary = []

# Loop over each muscle
for muscle in muscle_names:

    print(f"\nProcessing muscle: {muscle}\n")

    # Update preds_dirs for the current muscle by formatting the template paths
    pred_fold = base_preds_dir_template.format(muscle=muscle)

    if not os.path.exists(pred_fold):
        print(f"No pred folder found for {muscle}, skipping.")
        continue

    filenames = os.listdir(pred_fold)

    # Log how many t-files will be skipped
    t_files = [f for f in filenames if len(f.replace('.png','').split('_')) >= 4
               and f.replace('.png','').split('_')[3].lower() == 't']
    print(f"  Skipping {len(t_files)} thickness (t) files out of {len(filenames)} total.")

    # Prepare the partial function with fixed arguments
    partial_process_file = partial(
        process_file,
        fold=0,
        pred_fold=pred_fold,
        img_fold=image_dir,
        muscle=muscle,
        classes=classes,
        class_labels=class_labels
    )

    # Use multiprocessing Pool
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(partial_process_file, filenames), total=len(filenames), desc=f"Processing {muscle}"))

    # Flatten the list of lists
    for res in results:
        summary.extend(res)

# ------------------------------------------------------------------ #
# Save outputs                                                        #
# ------------------------------------------------------------------ #

df = pd.DataFrame().from_dict(summary)

# Save the DataFrame to a single Excel file
output_excel_path = f'/home/marloes.stemerdink@mydre.org/Documents/analysis/results/feature_extraction_output/segmentation_summary_{net}_{experiment}.xlsx'
df.to_excel(output_excel_path, index=False)
print(f"\nSummary Excel file saved to: {output_excel_path}")

# Optionally, save the DataFrame to a JSON file as well
output_json_path = f'/home/marloes.stemerdink@mydre.org/Documents/analysis/results/feature_extraction_output/segmentation_summary_{net}_{experiment}.json'
df.to_json(output_json_path, indent=4)
print(f"Summary JSON file saved to: {output_json_path}")