# ===========================
# Dataset
# ===========================

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

NUM_CLASSES = 4

# ===========================
# Training
# ===========================

BATCH_SIZE = 32
LEARNING_RATE = 0.001
LOCAL_EPOCHS = 1
GLOBAL_ROUNDS = 10
CLIENTS_PER_ROUND = 5
# This line is required
RANDOM_SEED = 42

# ===========================
# Federated Learning
# ===========================

NUM_CLIENTS = 25

# ===========================
# Paths
# ===========================

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_DIR / "KidneyCT"
MODELS_PATH = PROJECT_DIR / "models"
RESULTS_PATH = PROJECT_DIR / "results"
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints"