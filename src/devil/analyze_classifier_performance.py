import argparse
import os

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from ..common_util.misc import makedirs, pr_to_ipr_info, data_to_dataframe

_SCORE_TYPE_NAMES = ['camera_motion_score',
                     'fg_motion_score',
                     'bg_scene_motion_score',
                     'fg_displacement_score',
                     'fg_size_score']


def main(scores_path, labels_path, output_dir, delimiter, num_skip_rows, score_types):
    # Create root directory to store all output files
    makedirs(os.path.join(output_dir, 'plots'))
    makedirs(os.path.join(output_dir, 'tables'))

    # Read in scores and labels
    scores = pd.read_csv(scores_path, sep=delimiter, skiprows=num_skip_rows)
    labels = pd.read_csv(labels_path, sep=delimiter, skiprows=num_skip_rows)
    # Only keep labels for videos included in the results file
    labels = labels[labels['video_name'].isin(scores['video_name'])]
    # Replace "H" with True and "L" with False
    labels.replace({'H': True, 'L': False, 'N': False}, inplace=True)

    # Iterate through each score type
    for score_type_name in score_types:
        # Get rows where the current score type is not NaN
        non_nan_rows = ~scores[score_type_name].isnull()
        # Get a copy of the non-NaN scores for the current score type
        cur_scores = scores[score_type_name][non_nan_rows]
        # Replace inf with a dummy maximum value
        max_value_non_inf = cur_scores[cur_scores < np.inf].max()
        cur_scores_no_inf = cur_scores.replace(np.inf, max_value_non_inf + 1)
        # Normalize the scores between [0, 1] to represent probabilities (required to call precision-recall function)
        cur_scores_no_inf_prob = cur_scores_no_inf / cur_scores_no_inf.max()
        # Get labels for current score type
        cur_labels = labels[score_type_name][non_nan_rows]

        # Produce precision-recall curve data
        precision, recall, prc_thresholds_prob = precision_recall_curve(cur_labels, cur_scores_no_inf_prob)
        prc_auc = auc(recall, precision)
        i_precision, ipr_auc = pr_to_ipr_info(precision, recall)
        # Recover thresholds by de-normalizing
        prc_thresholds = prc_thresholds_prob * (max_value_non_inf + 1)
        # Restore inf
        prc_thresholds[prc_thresholds > max_value_non_inf] = np.inf

        # Produce ROC curve data
        fpr, tpr, roc_thresholds_prob = roc_curve(cur_labels, cur_scores_no_inf_prob)
        roc_auc = auc(fpr, tpr)
        # Recover thresholds by de-normalizing
        roc_thresholds = roc_thresholds_prob * (max_value_non_inf + 1)
        # Restore inf
        roc_thresholds[roc_thresholds > max_value_non_inf] = np.inf

        # Save PR and ROC curve plots to disk
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), tight_layout=True)
        plot_precision_recall_curve(axs[0], precision, recall)
        plot_roc_curve(axs[1], fpr, tpr)
        fig.suptitle(score_type_name)
        fig.savefig(os.path.join(output_dir, 'plots', f'{score_type_name}.pdf'))

        # Store PR and ROC curve information in a tab-separated value (TSV) file
        roc_data = {
            'precision': precision,
            'i_precision': i_precision,
            'recall': recall,
            'prc_thresholds': prc_thresholds,
            'prc_auc': prc_auc,
            'ipr_auc': ipr_auc,
            'fpr': fpr,
            'tpr': tpr,
            'roc_thresholds': roc_thresholds,
            'roc_auc': roc_auc
        }
        output_table = data_to_dataframe(roc_data)
        table_str = output_table.to_csv(index=False).replace(',', '\t')
        with open(os.path.join(output_dir, 'tables', f'{score_type_name}.tsv'), 'w') as f:
            f.write(table_str)


def plot_precision_recall_curve(axes, precision, recall):
    """Produces a figure containing the given precision-recall curve.

    :param axes: The axes to draw the plot on (e.g., from plt.subplot())
    :param recall: Descending list of recall values from `precision_recall_curve` (N float NumPy array)
    :param precision: Ascending list of precision values from `precision_recall_curve` (N float NumPy array)
    :return: plt.Figure
    """
    i_precision, ipr_auc = pr_to_ipr_info(precision, recall)

    # Plot "interpolated" precision-recall curve
    axes.plot(recall, i_precision, linestyle='--', color='#dddddd')

    # Plot actual data
    axes.plot(recall, precision)

    # Compute AUC (trapezoidal rule) for normal precision-recall curve
    pr_auc = auc(recall, precision)

    # Format and label the plot
    axes.set_xlabel('Recall')
    axes.set_ylabel('Precision')
    axes.set_xlim([-0.02, 1.02])
    axes.set_ylim([-0.02, 1.02])
    axes.set_title('Precision-Recall Curve (AUC={:.04f}, iAUC={:.04f})'.format(pr_auc, ipr_auc))


def plot_roc_curve(axes, fpr, tpr):
    """Produces a figure containing the given ROC curve.

    :param axes: The axes to draw the plot on (e.g., from plt.subplot())
    :param fpr: An ascending list of false positive rates from `roc_curve` (N float NumPy array)
    :param tpr: An ascending list of true positive rates from `roc_curve` (N float NumPy array)
    :return: plt.Figure
    """
    # Draw curve for a random classifier (i.e., one that evenly returns T/F for any Precision@k => FPR == TPR)
    axes.plot([0, 1], [0, 1], linestyle='--', color='#dddddd')

    # Plot actual data
    axes.plot(fpr, tpr)

    # Compute AUC (trapezoidal rule)
    roc_auc = auc(fpr, tpr)

    # Format and label the plot
    axes.set_xlabel('False Positive Rate')
    axes.set_ylabel('True Positive Rate')
    axes.set_xlim([-0.02, 1.02])
    axes.set_ylim([-0.02, 1.02])
    axes.set_title('ROC Curve (AUC={:.04f})'.format(roc_auc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scores_path', type=str, default='scores/scores.tsv',
                        help='Path to the predicted scores')
    parser.add_argument('-l', '--labels_path', type=str, default='labels/davis.tsv',
                        help='Path to the gold-standard labels')
    parser.add_argument('-o', '--output_dir', type=str, default='analysis-results/default',
                        help='Directory where plots and tables will be generated')
    parser.add_argument('-d', '--delimiter', type=str, default='\t',
                        help='The separation character that delimits cells in a row')
    parser.add_argument('--num_skip_rows', type=int, default=2,
                        help='The number of rows to skip when reading the results and labels files')
    parser.add_argument('-t', '--score_types', type=str, nargs='+', default=_SCORE_TYPE_NAMES,
                        help='Which score types to analyze')
    args = parser.parse_args()

    for score_type in args.score_types:
        assert score_type in _SCORE_TYPE_NAMES, f'"{score_type}" is not a supported score type'

    main(**vars(args))
