import tensorflow as tf
import os
import glob

AUTO_TUNE = tf.data.experimental.AUTOTUNE
CASSAVA_LEAF_PATH = os.path.join(os.path.dirname(__file__), 'cassava-leaf-disease-classification-data')
BATCH_SIZE = 32
IMAGE_SIZE = [512, 512]

FILE_NAMES = glob.glob(CASSAVA_LEAF_PATH + "/train_tfrecords/*.tfrec")

split_ind = int(0.8 * len(FILE_NAMES))

TRAINING_FILENAMES, VALID_FILENAMES = FILE_NAMES[:split_ind], FILE_NAMES[split_ind:]

LABEL_IDX_NAME_MAPPING = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy"
}

LABEL_IDX_SHORT_NAME_MAPPING = {
    0: "CBB",
    1: "CBSD",
    2: "CGM",
    3: "CMD",
    4: "Healthy"
}