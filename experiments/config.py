"""
SubstanceNet v4 — Experiment Configuration
==========================================
Single source of truth for all experiment parameters.
Ensures reproducibility across runs and machines.
"""
import os
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime

# === Paths ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'results')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# === Reproducibility ===
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=SEED):
    """Fix all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# === Plot Style ===
COLORS = {
    'primary': '#2563EB',      # Blue — SubstanceNet main
    'secondary': '#7C3AED',    # Purple — consciousness/wave
    'success': '#059669',      # Green — innate/biological
    'warning': '#D97706',      # Orange — acquired/Hebbian
    'danger': '#DC2626',       # Red — baselines/saturation
    'gray': '#6B7280',         # Gray — references
    'light': '#E5E7EB',        # Light gray — grid/background
}

def setup_plot_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'figure.figsize': (8, 5),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'axes.grid': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'legend.fontsize': 10,
        'legend.framealpha': 0.9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })

setup_plot_style()

# === Helpers ===
def save_results(name, data):
    """Save experiment results as JSON."""
    path = os.path.join(RESULTS_DIR, f'{name}.json')
    data['_meta'] = {
        'timestamp': datetime.now().isoformat(),
        'seed': SEED,
        'device': str(DEVICE),
        'torch_version': torch.__version__,
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f'Results saved: {path}')
    return path

def save_figure(name, fig=None):
    """Save figure as PNG and PDF."""
    if fig is None:
        fig = plt.gcf()
    png_path = os.path.join(FIGURES_DIR, f'{name}.png')
    pdf_path = os.path.join(FIGURES_DIR, f'{name}.pdf')
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f'Figure saved: {png_path}')
    return png_path

def create_model(**kwargs):
    """Create SubstanceNet with Hebbian learning disabled for deterministic tests."""
    from src.model.substance_net import SubstanceNet
    model = SubstanceNet(**kwargs).to(DEVICE)
    # Disable Hebbian for reproducible forward passes
    if hasattr(model.v3, 'output_proj') and hasattr(model.v3.output_proj, 'set_learning'):
        model.v3.output_proj.set_learning(False)
    for attr in ['compress_in', 'compress_out']:
        if hasattr(model.v4, attr):
            getattr(model.v4, attr).set_learning(False)
    return model

def enable_hebbian(model):
    """Re-enable Hebbian learning for maturation experiments."""
    if hasattr(model.v3, 'output_proj') and hasattr(model.v3.output_proj, 'set_learning'):
        model.v3.output_proj.set_learning(True)
    for attr in ['compress_in', 'compress_out']:
        if hasattr(model.v4, attr):
            getattr(model.v4, attr).set_learning(True)

def print_header(title):
    """Print formatted experiment header."""
    print(f'\n{"="*60}')
    print(f'  {title}')
    print(f'  Device: {DEVICE} | Seed: {SEED}')
    print(f'{"="*60}\n')

# === Data paths ===
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
