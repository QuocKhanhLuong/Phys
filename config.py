import os


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DATASET_ROOT = os.path.join(PROJECT_ROOT, 'BraTS21')
TRAIN_DATA_PATH = os.path.join(BASE_DATASET_ROOT, 'training')
TEST_DATA_PATH = os.path.join(BASE_DATASET_ROOT, 'testing')

NUM_EPOCHS = 250
NUM_CLASSES = 4
LEARNING_RATE = 1e-3
IMG_SIZE = 224
BATCH_SIZE = 24
NUM_SLICES = 5
EARLY_STOP_PATIENCE = 30

BRATS_CLASS_MAP = {
    0: "Background",
    1: "NCR/NET",
    2: "Edema",
    3: "ET"
}

