"""
=========================================================
config.py
Project Configuration
Works on:
1. Windows (VS Code)
2. Google Colab
3. Kaggle
=========================================================
"""

from pathlib import Path
import os

# =========================================================
# Image Configuration
# =========================================================

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

NUM_CLASSES = 4
# =========================================================
# Data Distribution
# =========================================================

# DATA_DISTRIBUTION = "non_iid"      # "iid" or "non_iid"
# -------------------------------

NUM_CLIENTS = 25
# Non-IID Configuration
# -------------------------------

USE_NON_IID = True

DIRICHLET_ALPHA = 0.3

# =========================================================
# Training Configuration
# =========================================================

BATCH_SIZE = 32

LEARNING_RATE = 0.001

LOCAL_EPOCHS = 10
GLOBAL_ROUNDS = 20
GLOBAL_EARLY_STOPPING = True
GLOBAL_PATIENCE = 5
MIN_DELTA = 0.0001
RANDOM_SEED = 42

# =========================================================
# Federated Learning
    # =========================================================

NUM_CLIENTS = 25

CLIENTS_PER_ROUND = 5

# =========================================================
# Project Directory
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

# =========================================================
# Detect Environment
# =========================================================

# Google Colab
if os.path.exists("/content"):

    print("Running on Google Colab")

    DATASET_PATH = Path("/content/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-and-Stone")
   # DATASET_PATH = Path("/content/drive/MyDrive/KidneyCT")

    MODELS_PATH = Path("/content/drive/MyDrive/FederatedLearning/models")

    RESULTS_PATH = Path("/content/drive/MyDrive/FederatedLearning/results")

    CHECKPOINT_PATH = Path("/content/drive/MyDrive/FederatedLearning/checkpoints")

# Kaggle
elif os.path.exists("/kaggle"):

    print("Running on Kaggle")

    DATASET_PATH = Path("/kaggle/input/kidneyct/KidneyCT")

    MODELS_PATH = PROJECT_DIR / "models"

    RESULTS_PATH = PROJECT_DIR / "results"

    CHECKPOINT_PATH = PROJECT_DIR / "checkpoints"

# Windows / Linux Local
else:

    print("Running on Local Machine")

    DATASET_PATH = PROJECT_DIR / "KidneyCT"

    MODELS_PATH = PROJECT_DIR / "models"

    RESULTS_PATH = PROJECT_DIR / "results"

    CHECKPOINT_PATH = PROJECT_DIR / "checkpoints"

# =========================================================
# Create Required Directories
# =========================================================

MODELS_PATH.mkdir(parents=True, exist_ok=True)

RESULTS_PATH.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)


