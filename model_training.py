"""
Train keras model on TFRecord files: https://keras.io/examples/keras_recipes/tfrecord/
"""

import tensorflow as tf

from model import make_model, checkpoint_cb, early_stopping_cb, augment, tensorboard_cb
from model_constants import TRAINING_FILENAMES, VALID_FILENAMES, FILE_NAMES, BATCH_SIZE
from tfrecord_util import get_dataset, get_splited_data

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

all_dataset = get_dataset(FILE_NAMES)
train_dataset, val_dataset, test_dataset = get_splited_data(all_dataset)

train_dataset = train_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)

with strategy.scope():
    model = make_model()

AUGMENT = False

if AUGMENT:
    train_dataset = augment(train_dataset)
    valid_dataset = augment(val_dataset)

history = model.fit(train_dataset,
                    epochs=2,
                    validation_data=val_dataset,
                    callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb])