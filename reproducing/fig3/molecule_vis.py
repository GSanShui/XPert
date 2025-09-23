import io

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from scipy.spatial.distance import cosine

PiYG_cmap = cm.get_cmap('PiYG')
target_green = PiYG_cmap(1.0) 
colorMap = LinearSegmentedColormap.from_list('WhiteToGreen', ['white', target_green], N=256)

def visualize_molecule_attention(smiles, attn_weights, norm=True, cmap=colorMap, figsize=(6, 6), radius=60, color_factor=0.3, save_path=None, ax=None):
    """
    Visualize molecular structure and atom attention weights

    Args:
    smiles: SMILES representation of the molecule
    attn_weights: Attention weights for each atom, shape (num_atoms,)
    """
    # Create molecule object
    mol = Chem.MolFromSmiles(smiles)
    
    # Generate 2D coordinates
    AllChem.Compute2DCoords(mol)
    
    # Normalize attention weights to [0,1]
    if norm:
        norm_attn = (attn_weights - attn_weights.min()) / (attn_weights.max() - attn_weights.min())
    else:
        norm_attn = attn_weights
    
    # Use RDKit's built-in rendering
    size = (800, 800)
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    drawer.drawOptions().useBWAtomPalette = False
    drawer.drawOptions().bondLineWidth = 3
    drawer.drawOptions().atomLabelFontSize = 20
    
    # Draw molecule and get atom coordinates
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    
    # Get base molecule image
    base_img_data = drawer.GetDrawingText()
    base_img = Image.open(io.BytesIO(base_img_data)).convert('RGBA')
    
    # Create a new image for gradient overlay
    overlay = Image.new('RGBA', base_img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Create a new drawer to get atom positions
    coords_drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    coords_drawer.ClearDrawing()
    
    # Copy drawing options for consistency
    for opt in dir(drawer.drawOptions()):
        if opt.startswith('_') or opt == 'extraMetadata':
            continue
        try:
            value = getattr(drawer.drawOptions(), opt)
            if hasattr(coords_drawer.drawOptions(), opt):
                setattr(coords_drawer.drawOptions(), opt, value)
        except:
            pass
    
    # Prepare for drawing but do not finish, so we can get atom coordinates
    rdMolDraw2D.PrepareAndDrawMolecule(coords_drawer, mol)
    
    # Get atom coordinates
    atom_coords = {}
    for i in range(mol.GetNumAtoms()):
        atom_coords[i] = coords_drawer.GetDrawCoords(i)

    # For each atom, create a gradient circle
    for i in range(mol.GetNumAtoms()):
        # Get atom coordinates in the image
        x, y = atom_coords[i]
        
        # Convert to pixel coordinates
        x = int(x)
        y = int(y)
        
        rgb = cmap(norm_attn[i]+ color_factor)
        color = (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        
        radius = radius 
        max_alpha_factor = 0.9
        power = 0.9
        for r in range(radius, 0, -1):
            normalized_r = r / radius
            gradient_effect = 1 - (normalized_r ** power) 
            alpha = int(255 * gradient_effect * (norm_attn[i]+0.1) * max_alpha_factor)
            alpha = max(0, min(255, alpha))
            circle_color = color + (alpha,)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=circle_color)


    # Overlay the gradient layer onto the base image
    final_img = Image.alpha_composite(base_img, overlay)
    
    # Show image
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.imshow(final_img)
    ax.axis('off')
    # plt.title('attention weight', fontsize=18, pad=20)
    
    # Add colorbar
    if norm:
        cbar_norm = plt.Normalize(attn_weights.min(), attn_weights.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=cbar_norm)
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.1)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Atom Attention Score',fontdict={'size': 16}, labelpad=12) # labelpad is the distance between label and colorbar
    cbar.outline.set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, format="svg", dpi=300, bbox_inches='tight')

    # plt.show()

    return fig, ax


def plot_attention_heatmap(attention_matrix, xticklabels=None, yticklabels=None, 
                           title="Atom Attention across Samples", figsize=(7, 6), cmap=colorMap,
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

    # Set colorbar font size
    cbar_ax = ax.figure.axes[-1]
    cbar_ax.tick_params(labelsize=16)

    ax.set_title(title, fontsize=22, pad=20)
    ax.set_xlabel("Atom Index", fontsize=22)
    ax.set_ylabel("Samples", fontsize=22)
    
    xticklabels=[ i for i in range(attention_matrix.shape[1])] if xticklabels is None else xticklabels
    yticklabels=[ i for i in range(attention_matrix.shape[0])] if yticklabels is None else yticklabels
    ax.set_xticks([i+0.5 for i in xticklabels])
    ax.set_xticklabels(xticklabels, rotation=0, fontsize=10)
    if len(yticklabels) > 30:
        ax.set_yticks([i + 0.5 for i in range(attention_matrix.shape[0]) if i % 4 == 0])
        ax.set_yticklabels([str(i) for i in range(attention_matrix.shape[0]) if i % 4 == 0], rotation=0, fontsize=10)
    else:
        ax.set_yticks([i+0.5 for i in yticklabels])
        ax.set_yticklabels(yticklabels, rotation=0, fontsize=10)


    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, format="svg", dpi=300, bbox_inches='tight')

    # plt.show()

    return ax.get_figure(), ax



def plot_attention_scatter(attention_vectors, labels=None, title="Mean Atom Attention across Seeds", figsize=(8, 6),
                           cmap='Purples', save_path=None, ax=None):
    """
    Plot scatter of mean attention vectors across different seeds, and compute cosine similarity and coefficient of variation, annotated on the plot.

    Args:
        attention_vectors (np.ndarray): 2D array of attention vectors for different seeds,
                                        shape (num_seeds, num_atoms).
        labels (list, optional): Label for each vector, used in legend. Defaults to 'Seed 1', 'Seed 2', ....
        title (str, optional): Plot title.
        figsize (tuple, optional): Figure size.
        cmap (str, optional): Color map for different points.
        save_path (str, optional): Path to save the plot. If None, do not save.
    """
    num_seeds, num_atoms = attention_vectors.shape
    
    # Color settings
    colors = sns.color_palette(cmap, n_colors=num_seeds)
    if labels is None:
        labels = [f"Seed {i+1}" for i in range(num_seeds)]

    # 1. Compute cosine similarity matrix
    cosine_sim_matrix = np.zeros((num_seeds, num_seeds))
    for i in range(num_seeds):
        for j in range(num_seeds):
            cosine_sim_matrix[i, j] = 1 - cosine(attention_vectors[i], attention_vectors[j])
    
    np.fill_diagonal(cosine_sim_matrix, np.nan)
    mean_cosine_sim = np.nanmean(cosine_sim_matrix)

    # 2. Compute coefficient of variation (CV)
    mean_attention = np.mean(attention_vectors, axis=0)
    std_attention = np.std(attention_vectors, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = np.where(mean_attention != 0, std_attention / mean_attention, 0)
    mean_cv = np.mean(cv)

    # Plot scatter
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    for i in range(num_seeds):
        ax.scatter(range(num_atoms), attention_vectors[i], label=labels[i],
                   color=colors[i], alpha=0.7, s=50)

    # 1. Annotate Cosine Sim and CV
    ax.text(0.02, 0.95, f"Mean Cosine Similarity: {mean_cosine_sim:.3f}", 
            transform=ax.transAxes, fontsize=18, verticalalignment='top')
    ax.text(0.02, 0.88, f"Mean CV: {mean_cv:.3f}", 
            transform=ax.transAxes, fontsize=18, verticalalignment='top')

    # Set plot title and labels
    ax.set_title(title, fontsize=22, pad=20)
    ax.set_xlabel("Atom Index", fontsize=22)
    ax.set_ylabel("Attention Score", fontsize=22)
    
    # Set x-axis ticks
    ax.set_xticks(range(num_atoms))
    ax.set_xticklabels([i for i in range(num_atoms)], rotation=0, fontsize=10)
    ax.tick_params(axis='y', labelsize=14)
    
    # 3. Remove horizontal grid lines, keep vertical grid lines at each xtick
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Remove legend border
    # ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12)
    ax.legend(loc='upper right', fontsize=16)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format="svg", dpi=300, bbox_inches='tight')


    return ax.get_figure(), ax
