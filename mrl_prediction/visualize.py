import os
import pickle
from matplotlib import patches, gridspec
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm

from config import results_dir, images_dir
from utils.model_definitions import MODELS

COLORMAP = 'viridis'
GRIDSIZE = 180

def load_model_results(model_name, show_framepool_50, show_framepool_100, show_optimus_50, show_optimus_100):
    valuated_models = []
    result_folder = os.path.join(results_dir, "mrl_predictions")
    if show_framepool_50:
        with open(os.path.join(result_folder, "framepool50.pkl"), "rb") as f:
            framepool50 = pickle.load(f)
        valuated_models.append(("Framepool50", framepool50))

    if show_framepool_100:
        with open(os.path.join(result_folder, "framepool100.pkl"), "rb") as f:
            framepool100 = pickle.load(f)
        valuated_models.append(("Framepool100", framepool100))

    if show_optimus_50:
        with open(os.path.join(result_folder, "optimus50.pkl"), "rb") as f:
            optimus50 = pickle.load(f)
        valuated_models.append(("Optimus50", optimus50))

    if show_optimus_100:
        with open(os.path.join(result_folder, "optimus100.pkl"), "rb") as f:
            optimus100 = pickle.load(f)
        valuated_models.append(("Optimus100", optimus100))

    if len(valuated_models) == 0:
        return []

    result_file = os.path.join(result_folder, f"{model_name}.pkl")
    if not os.path.exists(result_file):
        print(f"File {result_file} does not exist")
        return []

    with open(result_file, "rb") as f:
        model_results = pickle.load(f)

    valuated_models.append((model_name, model_results))

    return valuated_models

def visualize_mrl(model_name, show_framepool_50=False, show_framepool_100=False, show_optimus_50=False, show_optimus_100=False):
    class_dir = os.path.join(images_dir, 'mrl_prediction')
    os.makedirs(class_dir, exist_ok=True)
    figure_path = os.path.join(class_dir, f"{model_name}.pdf")

    models = load_model_results(model_name, show_framepool_50, show_framepool_100, show_optimus_50, show_optimus_100)

    num_models = len(models)
    if num_models == 0:
        return

    figsize = (18, 4 * num_models)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(num_models, 4, width_ratios=[1, 1, 1, 1], wspace=0.05, hspace=0.05)

    for i, (name, results) in enumerate(models):
        ax_random_fixed = fig.add_subplot(gs[i, 0])
        ax_random_var = fig.add_subplot(gs[i, 1])
        ax_human_fixed = fig.add_subplot(gs[i, 2])
        ax_human_var = fig.add_subplot(gs[i, 3])

        if i == 0:
            rand_fixed_rect = patches.FancyBboxPatch(
                (0.0, 1.0),
                1.0, 0.1,
                boxstyle="round,pad=0.00",
                transform=ax_random_fixed.transAxes,
                linewidth=0.5,
                edgecolor='black',
                facecolor='white',
                clip_on=False
            )
            ax_random_fixed.text(
                0.5, 1.05, "Random (50bp) MPRA",
                transform=ax_random_fixed.transAxes,
                ha='center',
                va='center',
                fontsize=14
            )
            ax_random_fixed.add_patch(rand_fixed_rect)

            rand_var_rect = patches.FancyBboxPatch(
                (0.0, 1.0),
                1.0, 0.1,
                boxstyle="round,pad=0.00",
                transform=ax_random_var.transAxes,
                linewidth=0.5,
                edgecolor='black',
                facecolor='white',
                clip_on=False
            )
            ax_random_var.text(
                0.5, 1.05, "Random (25-100bp) MPRA",
                transform=ax_random_var.transAxes,
                ha='center',
                va='center',
                fontsize=14
            )
            ax_random_var.add_patch(rand_var_rect)

            human_fixed_rect = patches.FancyBboxPatch(
                (0.0, 1.0),
                1.0, 0.1,
                boxstyle="round,pad=0.00",
                transform=ax_human_fixed.transAxes,
                linewidth=0.5,
                edgecolor='black',
                facecolor='white',
                clip_on=False
            )
            ax_human_fixed.text(
                0.5, 1.05, "Human (50bp) MPRA",
                transform=ax_human_fixed.transAxes,
                ha='center',
                va='center',
                fontsize=14
            )
            ax_human_fixed.add_patch(human_fixed_rect)

            human_var_rect = patches.FancyBboxPatch(
                (0.0, 1.0),
                1.0, 0.1,
                boxstyle="round,pad=0.00",
                transform=ax_human_var.transAxes,
                linewidth=0.5,
                edgecolor='black',
                facecolor='white',
                clip_on=False
            )
            ax_human_var.text(
                0.5, 1.05, "Human (25-100bp) MPRA",
                transform=ax_human_var.transAxes,
                ha='center',
                va='center',
                fontsize=14
            )
            ax_human_var.add_patch(human_var_rect)

        """
        Random with Fixed Sequence Lengths
        """
        y_true_random_fixed = results["y_true_random_fixed"]
        y_pred_random_fixed = results["y_pred_random_fixed"]

        hb = ax_random_fixed.hexbin(
            y_true_random_fixed,
            y_pred_random_fixed,
            gridsize=GRIDSIZE,
            cmap=COLORMAP,
            bins='log'
        )
        R_random_fixed = pearsonr(y_true_random_fixed, y_pred_random_fixed)[0]
        ax_random_fixed.text(
            0.65, 0.1, f"R = {R_random_fixed:.3f}",
            transform=ax_random_fixed.transAxes,
            fontsize=14,
            fontstyle='italic',
            verticalalignment='top'
        )

        if i < num_models - 1:
            ax_random_fixed.set_xticks([])


        """
        Random with Variable Sequence Lengths
        """
        y_true_random_var = results["y_true_random_var"]
        y_pred_random_var = results["y_pred_random_var"]

        ax_random_var.hexbin(
            y_true_random_var,
            y_pred_random_var,
            gridsize=GRIDSIZE,
            cmap=COLORMAP,
            bins='log',
        )
        R_random_var = pearsonr(y_true_random_var, y_pred_random_var)[0]
        ax_random_var.text(
            0.65, 0.1, f"R = {R_random_var:.3f}",
            transform=ax_random_var.transAxes,
            fontsize=14,
            fontstyle='italic',
            verticalalignment='top'
        )

        ax_random_var.set_yticks([])
        if i < num_models - 1:
            ax_random_var.set_xticks([])

        """
        Human with Fixed Sequence Lengths
        """
        y_true_human_fixed = results["y_true_human_fixed"]
        y_pred_human_fixed = results["y_pred_human_fixed"]

        ax_human_fixed.hexbin(
            y_true_human_fixed,
            y_pred_human_fixed,
            gridsize=GRIDSIZE,
            cmap=COLORMAP,
            bins='log'
        )
        R_human_fixed = pearsonr(y_true_human_fixed, y_pred_human_fixed)[0]
        ax_human_fixed.text(
            0.65, 0.1, f"R = {R_human_fixed:.3f}",
            transform=ax_human_fixed.transAxes,
            fontsize=14,
            fontstyle='italic',
            verticalalignment='top'
        )

        ax_human_fixed.set_yticks([])
        if i < num_models - 1:
            ax_human_fixed.set_xticks([])

        """
        Human with Variable Sequence Lengths
        """
        y_true_human_var = results["y_true_human_var"]
        y_pred_human_var = results["y_pred_human_var"]

        ax_human_var.hexbin(
            y_true_human_var,
            y_pred_human_var,
            gridsize=GRIDSIZE,
            cmap=COLORMAP,
            bins='log'
        )
        R_human_var = pearsonr(y_true_human_var, y_pred_human_var)[0]
        ax_human_var.text(
            0.65, 0.1, f"R = {R_human_var:.3f}",
            transform=ax_human_var.transAxes,
            fontsize=14,
            fontstyle='italic',
            verticalalignment='top'
        )

        ax_human_var.set_yticks([])
        if i < num_models - 1:
            ax_human_var.set_xticks([])

        ax_human_var.text(1.01, 0.5, f"{name}", transform=ax_human_var.transAxes,
                      rotation=270, va='center', ha='left', fontsize=14)

        rect_model_name = patches.FancyBboxPatch(
            (1.00, -0.002),
            0.1, 1.0,
            transform=ax_human_var.transAxes,
            boxstyle="round,pad=0.00",
            linewidth=0.5,
            edgecolor='black',
            facecolor='white',
            clip_on=False
        )
        ax_human_var.add_patch(rect_model_name)
        print(f"{name}: R_random_fixed = {R_random_fixed:.3f}    R_random_var = {R_random_var:.3f}    R_human_fixed = {R_human_fixed:.3f}    R_human_var = {R_human_var:.3f}")

    cbar_ax = fig.add_axes([0.94, 0.3, 0.02, 0.4])  # [left, bottom, width, height]
    cbar = fig.colorbar(hb, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_title("count", fontsize=14, pad=10, loc='center')

    if num_models == 2:
        bottom_pad = 0.08
    elif num_models == 3:
        bottom_pad = 0.06
    elif num_models == 4:
        bottom_pad = 0.055
    elif num_models == 5:
        bottom_pad = 0.048
    else:
        bottom_pad = 0.048

    fig.text(0.01, 0.5, 'Predicted MRL', va='center', rotation='vertical', fontsize=16)
    fig.text(0.5, 0.02, 'Observed MRL', ha='center', fontsize=16)
    plt.subplots_adjust(left=0.04, right=0.90, top=0.96, bottom=bottom_pad)
    plt.tight_layout()
    plt.savefig(figure_path)

if __name__ == "__main__":
    show_framepool_50 = True
    show_framepool_100 = False
    show_optimus_50 = False
    show_optimus_100 = False
    for model_name in tqdm(MODELS):
        visualize_mrl(
            model_name,
            show_framepool_50,
            show_framepool_100,
            show_optimus_50,
            show_optimus_100
        )