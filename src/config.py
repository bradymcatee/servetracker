from pathlib import Path

#Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT.parent / "Dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

# Model parameters
INPUT_HEIGHT = 360
INPUT_WIDTH = 640
BATCH_SIZE = 2
LEARNING_RATE = 1.0
NUM_EPOCHS = 500
STEPS_PER_EPOCH = 200

# Training parameters
TRAIN_VAL_SPLIT = 0.8
RANDOM_SEED = 42
