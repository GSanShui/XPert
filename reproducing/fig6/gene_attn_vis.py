import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import cosine

PiYG_cmap = cm.get_cmap('PiYG')
target_green = PiYG_cmap(1.0) 
colorMap = LinearSegmentedColormap.from_list('WhiteToGreen', ['white', target_green], N=256)


def plot_attention_heatmap(attention_matrix, xticklabels=None, yticklabels=None, 
                           title="Gene Attention across Samples", figsize=(20, 8), cmap=colorMap,
                           annot=True, fmt=".3f", save_path=None, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    sns.heatmap(
        attention_matrix,
        ax=ax,
        cmap=cmap,
        annot=annot, 
        fmt=fmt,
        cbar=True,
        linewidths=.5
    )

    # Set cbar font size
    cbar_ax = ax.figure.axes[-1]
    cbar_ax.tick_params(labelsize=16)

    ax.set_title(title, fontsize=22, pad=20)
    ax.set_xlabel("Gene Index", fontsize=22)
    ax.set_ylabel("Samples", fontsize=22)
    
    xticklabels=[ i for i in range(attention_matrix.shape[1]+1)] if xticklabels is None else xticklabels
    yticklabels=[ i for i in range(attention_matrix.shape[0]+1)] if yticklabels is None else yticklabels
    # ax.set_xticks([i+0.5 for i in xticklabels])
    # ax.set_xticklabels(xticklabels, rotation=0, fontsize=10)
    if len(xticklabels) > 30:
        ax.set_xticks([i + 0.5 for i in range(attention_matrix.shape[1]+1) if i % 20 == 0])
        ax.set_xticklabels([str(i) for i in range(attention_matrix.shape[1]) if i % 20 == 0], rotation=0, fontsize=12)

    if len(yticklabels) > 30:
        ax.set_yticks([i + 0.5 for i in range(attention_matrix.shape[0]+1) if i % 5 == 0])
        ax.set_yticklabels([str(i) for i in range(attention_matrix.shape[0]) if i % 5 == 0], rotation=0, fontsize=12)
    else:
        ax.set_yticks([i+0.5 for i in xticklabels])
        ax.set_yticklabels(xticklabels, rotation=0, fontsize=10)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, format="svg", dpi=300, bbox_inches='tight')

    # plt.show()

    return ax.get_figure(), ax



def plot_attention_boxplot(attention_matrix, xticklabels=None,
                           title="Distribution of Gene Attention across Samples", figsize=(20, 8),
                           save_path=None, ax=None, palette="Spectral"):
    """
    Draw boxplots for attention scores, each column of the matrix corresponds to a boxplot.

    Args:
    - attention_matrix (np.array): 2D attention matrix (samples, genes).
    - xticklabels (list, optional): X-axis labels. Defaults to column indices.
    - title (str, optional): Chart title.
    - figsize (tuple, optional): Figure size.
    - save_path (str, optional): Path to save the image.
    - ax (matplotlib.axes.Axes, optional): Existing matplotlib axis object for plotting.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Seaborn's boxplot function is suitable for this case
    # It automatically treats each DataFrame column as a separate distribution for boxplot
    sns.boxplot(data=pd.DataFrame(attention_matrix), ax=ax, showfliers=False, palette="Spectral", saturation=0.6) # showfliers=False hides outliers

    # --- Set chart title and axis labels ---
    ax.set_title(title, fontsize=22, pad=20)
    ax.set_xlabel("Gene Index", fontsize=22)
    ax.set_ylabel("Attention Score", fontsize=22) # Y-axis is now attention value

    # --- Adjust axis ticks ---
    ax.tick_params(axis='y', labelsize=16) # Set Y-axis tick font size

    num_genes = attention_matrix.shape[1]
    # If xticklabels not provided, use default indices
    if xticklabels is None:
        xticklabels = [str(i) for i in range(num_genes)]

    # If there are too many genes (columns), show only some X-axis labels to avoid overlap
    if num_genes > 30:
        # Show a tick label every 20 genes
        tick_positions = [i for i in range(num_genes) if i % 20 == 0]
        tick_labels = [xticklabels[i] for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=0, fontsize=14)
    else:
        # If not many genes, show all labels
        ax.set_xticks(range(num_genes))
        ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=14)

    ax.set_xlim([-1, num_genes+1])

    plt.tight_layout()

    # --- Save image ---
    if save_path is not None:
        plt.savefig(save_path, format="svg", dpi=300, bbox_inches='tight')

    return fig, ax




def plot_attention_scatter(attention_vectors, labels=None, title="Mean Gene Attention across Folds", figsize=(16, 6),
                           cmap='Purples', save_path=None, ax=None):
    """
    Plot scatter of mean attention vectors across different seeds, and calculate cosine similarity and coefficient of variation, annotated on the plot.

    Args:
        attention_vectors (np.ndarray): 2D array of attention vectors for different seeds,
                                        shape (num_seeds, num_atoms).
        labels (list, optional): Label for each vector, used for legend. Defaults to 'Seed 1', 'Seed 2', ...
        title (str, optional): Chart title.
        figsize (tuple, optional): Figure size.
        cmap (str, optional): Color map for different colored points.
        save_path (str, optional): Path to save the chart. If None, do not save.
    """
    num_seeds, num_atoms = attention_vectors.shape
    
    # Color settings
    colors = sns.color_palette(cmap, n_colors=num_seeds)
    if labels is None:
        labels = [f"Fold {i+1}" for i in range(num_seeds)]

    # 1. Calculate cosine similarity matrix
    cosine_sim_matrix = np.zeros((num_seeds, num_seeds))
    for i in range(num_seeds):
        for j in range(num_seeds):
            cosine_sim_matrix[i, j] = 1 - cosine(attention_vectors[i], attention_vectors[j])
    
    np.fill_diagonal(cosine_sim_matrix, np.nan)
    mean_cosine_sim = np.nanmean(cosine_sim_matrix)

    # 2. Calculate coefficient of variation (CV)
    mean_attention = np.mean(attention_vectors, axis=0)
    std_attention = np.std(attention_vectors, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = np.where(mean_attention != 0, std_attention / mean_attention, 0)
    mean_cv = np.mean(cv)

    # Draw scatter plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    for i in range(num_seeds):
        ax.scatter(range(num_atoms), attention_vectors[i], label=labels[i],
                   color=colors[i], alpha=1, s=13)

    # 1. Annotate Cosine Sim and CV
    ax.text(0.02, 0.95, f"Mean Cosine Similarity: {mean_cosine_sim:.3f}", 
            transform=ax.transAxes, fontsize=18, verticalalignment='top')
    ax.text(0.02, 0.88, f"Mean CV: {mean_cv:.3f}", 
            transform=ax.transAxes, fontsize=18, verticalalignment='top')

    # Set chart title and labels
    ax.set_title(title, fontsize=22, pad=20)
    ax.set_xlabel("Gene Index", fontsize=22)
    ax.set_ylabel("Attention Score", fontsize=22)
    
    # Set x-axis ticks
    ax.set_xticks([i + 0.5 for i in range(attention_vectors.shape[1]) if i % 20 == 0])
    ax.set_xticklabels([str(i) for i in range(attention_vectors.shape[1]) if i % 20 == 0], rotation=0, fontsize=14)

    ax.tick_params(axis='y', labelsize=16)

    # Set y-axis range
    ax.set_ylim([0, 0.013])
    ax.set_xlim([-1, num_atoms+1])
    
    # 3. Remove horizontal grid lines, keep vertical grid lines, and set at each xtick position
    # ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Remove legend border
    # ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12)
    ax.legend(loc='upper right', fontsize=16)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format="svg", dpi=300, bbox_inches='tight')

    return ax.get_figure(), ax
