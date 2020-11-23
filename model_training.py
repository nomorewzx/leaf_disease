"""
Train keras model on TFRecord files: https://keras.io/examples/keras_recipes/tfrecord/
"""

import tensorflow as tf

from model_constants import TRAINING_FILENAMES, VALID_FILENAMES, LABEL_IDX_NAME_MAPPING, LABEL_IDX_SHORT_NAME_MAPPING
from tfrecord_util import get_dataset
import matplotlib.pyplot as plt

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()

print('Number of replicas:', strategy.num_replicas_in_sync)
print('TRAIN tf records files', len(TRAINING_FILENAMES))
print('VALIDATION tf records files', len(VALID_FILENAMES))

train_dataset = get_dataset(TRAINING_FILENAMES)
valid_dataset = get_dataset(VALID_FILENAMES)

image_batch, label_batch = next(iter(train_dataset))


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5,5, n+1)
        plt.imshow(image_batch[n]/255.0)
        if label_batch[n] is not None:
            plt.title(f"{LABEL_IDX_SHORT_NAME_MAPPING[label_batch[n]]}")
        else:
            plt.title('No Label')
        plt.axis("off")
    plt.show()


show_batch(image_batch.numpy(), label_batch.numpy())