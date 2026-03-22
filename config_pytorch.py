"""
Configuration file for Dog Breed Classification with PyTorch
"""
import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset', 'Images')
ANNOTATION_DIR = os.path.join(BASE_DIR, 'dataset', 'Annotation')
MODEL_DIR = os.path.join(BASE_DIR, 'models_pytorch')
LOGS_DIR = os.path.join(BASE_DIR, 'logs_pytorch')
RESULTS_DIR = os.path.join(BASE_DIR, 'results_pytorch')

# Create directories if they don't exist
for directory in [MODEL_DIR, LOGS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Device Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4  # For DataLoader
PIN_MEMORY = True if torch.cuda.is_available() else False

# Model Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
NUM_CLASSES = 120
BATCH_SIZE = 32  # GPU có thể handle batch size lớn hơn
EPOCHS = 30
LEARNING_RATE = 0.001

# Data Split
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Data Augmentation (using torchvision transforms)
AUGMENTATION = True
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
NORMALIZE_STD = [0.229, 0.224, 0.225]   # ImageNet std

# Model Architecture
MODEL_ARCHITECTURE = 'resnet50'  # Options: 'resnet50', 'efficientnet_b0', 'mobilenet_v3', 'vit_b_16'
USE_PRETRAINED = True
FREEZE_BACKBONE = True  # Freeze early layers for transfer learning
DROPOUT_RATE = 0.5

# Training Configuration
EARLY_STOPPING_PATIENCE = 7
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.1
MIN_LR = 1e-7
WEIGHT_DECAY = 1e-4

# Optimizer
OPTIMIZER = 'adam'  # Options: 'adam', 'sgd', 'adamw'
SGD_MOMENTUM = 0.9

# Loss Function
LABEL_SMOOTHING = 0.1  # Label smoothing for better generalization

# Callbacks
SAVE_BEST_ONLY = True
MONITOR_METRIC = 'val_accuracy'
MODE = 'max'

# Mixed Precision Training (faster on GPU)
USE_AMP = True  # Automatic Mixed Precision

# Random Seed
RANDOM_SEED = 42

# Logging
LOG_INTERVAL = 10  # Log every N batches
SAVE_CHECKPOINT_EVERY = 5  # Save checkpoint every N epochs

print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
