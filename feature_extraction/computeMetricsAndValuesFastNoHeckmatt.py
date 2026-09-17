import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
import gc
from matplotlib.colors import ListedColormap  # Import for custom colormap

# Avoid plotting plt figures to screen
plt.ioff()

# Define base directories
# TODO: update this to point at your own DATA/RESULTS folders
RESULTS_DIR = Path('/mnt/data/dataset_training/subset_1/results_inference_2/')
testing_round = 'base_model'

def load_data(experiment):
    """Loads segmentation summary data.

    Note: HeckMap loading was removed since none of the downstream steps
    (IoU by fold, confusion matrices, boxplots, summary tables) use the
    sex/age/bmi/manual_h_score columns it provided. If you later get
    hold of 'heckMapPlusCharacteristics.xlsx', you can reinstate it and
    pass it into process_data.
    """
    segmentation_summary_file = RESULTS_DIR / f'segmentation_summary_{experiment}.json'
    with open(segmentation_summary_file, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame.from_dict(data)
    del data
    gc.collect()
    return df


def process_data(df):
    """Processes and cleans the segmentation dataframe.

    HeckMap-derived columns (sex, age, bmi, manual_h_score) have been
    dropped since nothing downstream in this script uses them.
    """
    df['Slice'] = df['File'].apply(lambda x: x.split('_')[-1].split('.')[0])
    df = df[df['Slice'].astype(int) <= 90]

    # Map 'class_gt' to 'muscle_code'
    class_to_code = {
        'Biceps_brachii': '001',
        'Deltoideus': '002',
        'Depressor_anguli_oris': '003',
        'Digastricus': '004',
        'Extensor_digitorum_brevis': '005',
        'Flexor_carpi_radialis': '006',
        'Flexor_digitorum_profundus': '007',
        'Gastrocnemius_medial_head': '008',
        'Geniohyoideus': '009',
        'Levator_labii_superior': '010',
        'Masseter': '011',
        'Mentalis': '012',
        'Orbicularis_oris': '013',
        'Peroneus_tertius': '014',
        'Rectus_abdominis': '015',
        'Rectus_femoris': '016',
        'Temporalis': '017',
        'Tibialis_anterior': '018',
        'Trapezius': '019',
        'Vastus_lateralis': '020',
        'Zygomaticus': '021'
    }
    df['muscle_code'] = df['Muscle'].map(class_to_code)

    # Extract 'gt_mean' and 'pred_mean'
    df['gt_mean'] = df['features_img_gt'].apply(
        lambda x: x.get('original_firstorder_Mean') if isinstance(x, dict) else np.nan)
    df['pred_mean'] = df['features_img_pred'].apply(
        lambda x: x.get('original_firstorder_Mean') if isinstance(x, dict) else np.nan)

    # Create 'muscle_side' column (kept in case you want it later; doesn't require HeckMap)
    df['muscle_side'] = df['muscle_code'].astype(str) + '_' + df['side'].astype(str)

    return df


def compute_mean_iou_per_fold(df):
    """Calculates mean IoU grouped by folds."""
    mean_iou = df.groupby('Fold')['iou'].mean().reset_index()
    std_iou = df.groupby('Fold')['iou'].std().reset_index()
    return mean_iou, std_iou


def plot_boxplots(df, experiment):
    """Creates and saves boxplots for segmentation metrics."""
    boxplot_dir = RESULTS_DIR / 'BOXPLOT'
    boxplot_dir.mkdir(parents=True, exist_ok=True)
    metrics = ['iou', 'prec', 'rec']

    # Create a single boxplot for the whole dataset
    for metric in metrics:
        plt.figure(figsize=(16, 8))
        sns.boxplot(x='Muscle', y=metric, data=df, palette='tab10')  # Using 'tab10' palette
        plt.title(f'Boxplot of {metric} grouped by Muscle')
        plt.xticks(rotation=90)
        plt.savefig(boxplot_dir / f'{testing_round}_boxplot_{metric}_total.png', dpi=300, bbox_inches='tight')
        plt.close()


def compute_and_save_summary_tables(df, classes, experiment):
    """Computes summary statistics and saves them to Excel."""
    excel_dir = RESULTS_DIR / 'EXCEL'
    excel_dir.mkdir(parents=True, exist_ok=True)
    output_file = excel_dir / f'{experiment}_summary_tables.xlsx'

    # Calculate mean and std dev grouped by 'Muscle'
    mean_metrics = df.groupby('Muscle')[['iou', 'prec', 'rec']].mean()
    std_metrics = df.groupby('Muscle')[['iou', 'prec', 'rec']].std()

    # Compute confusion matrix
    cm = confusion_matrix(df['Muscle'], df['Muscle'], labels=classes)

    # Calculate the percentage of correct classifications for each class
    correct_percentage = pd.DataFrame({
        'Correct Percentage': (cm.diagonal() / cm.sum(axis=1)) * 100
    }, index=classes).round(2)

    # Calculate the percentage of entries with 'iou' lower than 0.2 for each class
    low_iou_percentage = pd.DataFrame({
        'Low IoU Percentage': df.groupby('Muscle')['iou'].apply(lambda x: (x < 0.2).mean() * 100)
    }, index=classes).round(2)

    # Concatenate the DataFrames
    summary_table = pd.concat([correct_percentage, low_iou_percentage, mean_metrics, std_metrics], axis=1)
    summary_table = summary_table.round(2)

    # Create a DataFrame with mean and std in the format 'mean +/- std'
    metrics_formatted = mean_metrics.copy()
    for col in metrics_formatted.columns:
        metrics_formatted[col] = mean_metrics[col].map('{:.2f}'.format) + ' +/- ' + std_metrics[col].map('{:.2f}'.format)

    # Save to Excel
    with pd.ExcelWriter(output_file) as writer:
        summary_table.to_excel(writer, sheet_name='Summary', index_label='Muscle')
        metrics_formatted.to_excel(writer, sheet_name='Metrics Formatted', index_label='Muscle')
    print(f"Summary tables saved to {output_file}")


def main():
    experiment = 'knet_swin_mod_muscle_specific'

    # Load data
    df = load_data(experiment)

    # Process data
    df_processed = process_data(df)

    # Compute mean IoU per fold
    mean_iou_by_fold, std_iou_by_fold = compute_mean_iou_per_fold(df_processed)
    print("Mean IoU by Fold:")
    print(mean_iou_by_fold)
    print("Std IoU by Fold:")
    print(std_iou_by_fold)

    # Save mean and std IoU per fold to Excel in the same file
    excel_dir = RESULTS_DIR / 'EXCEL'
    excel_dir.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(excel_dir / f'{experiment}_iou_by_fold.xlsx') as writer:
        mean_iou_by_fold.to_excel(writer, sheet_name='mean_iou', index=False)
        std_iou_by_fold.to_excel(writer, sheet_name='std_iou', index=False)

    classes = [cl for cl in df_processed['Muscle'].unique() if cl != 'background']

    # Plot boxplots
    plot_boxplots(df_processed, experiment)

    # Compute and save summary tables
    compute_and_save_summary_tables(df_processed, classes, experiment)


if __name__ == "__main__":
    main()