DATA_PATH = "/home/ubuntu/brats_2018_on_intel/data/processed/Task01_BrainTumour/"
DATA_DIR = "/home/ubuntu/brats_2018_on_intel/data/processed/"
DATASET = "Task01_BrainTumour/"

# DATA_PATH="../data/decathlon/Task01_BrainTumour/"

TRAIN_TEST_SPLIT = 0.80
VALIDATE_TEST_SPLIT = 0.50

BATCH_SIZE_TRAIN = 8
BATCH_SIZE_VALIDATE = 4
BATCH_SIZE_TEST = 1

TILE_HEIGHT = 144
TILE_WIDTH = 144
TILE_DEPTH = 144
NUMBER_INPUT_CHANNELS = 1

CROP_DIM = (144,144,144,1)

NUMBER_OUTPUT_CLASSES=1


MODEL_DIR = "/home/ubuntu/brats_2018_on_intel/models"
SAVED_MODEL_NAME="3d_unet_decathlon"

FILTERS=16
NUM_EPOCHS=40

PRINT_MODEL=False
USE_UPSAMPLING=False


OV_OUTPUT_DIR = "/home/ubuntu/brats_2018_on_intel/models/openvino"
IR_MODEL_PRECISION = "FP32"


RANDOM_SEED=64
