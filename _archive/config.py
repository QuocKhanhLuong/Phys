import os


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

USE_MONAI_PIPELINE = True
MONAI_SAMPLES_PER_PATIENT = None  
DATA_NUM_WORKERS = 8  
DATA_PREFETCH_FACTOR = 2  

NPY_DIR = os.path.join(PROJECT_ROOT, 'BraTS21_preprocessed')
MONAI_NPY_DIR = os.path.join(PROJECT_ROOT, 'BraTS21_preprocessed_monai')
MONAI_CACHE_DIR = os.path.join(PROJECT_ROOT, 'monai_cache')

BRATS_RAW_DIR = os.path.join(PROJECT_ROOT, 'BraTS21')


NUM_EPOCHS = 250
NUM_CLASSES = 4
LEARNING_RATE = 5e-5  # Reduced from 3e-4 for BraTS21
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

BRATS_REGION_NAMES = {
    'ET': 'Enhancing Tumor',
    'TC': 'Tumor Core',
    'WT': 'Whole Tumor'
}

