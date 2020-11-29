import math
import os
from functools import partial

import pandas as pd
import tensorflow as tf

from model_constants import IMAGE_SIZE, AUTO_TUNE


def decode_img(image):
    image = tf.image.decode_image(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image


def read_tfrecord(example, labeled):
    tfrecord_format = (
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
            "image_name": tf.io.FixedLenFeature([], tf.string)
        }
    )

    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_img(example["image"])
    label_id = tf.cast(example["target"], tf.int32)
    label = tf.one_hot(label_id, 5)
    return image, label


def load_dataset(filenames, labeled=True) -> tf.data.TFRecordDataset:
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTO_TUNE)
    return dataset


def get_dataset(filenames, labeled=True) -> tf.data.TFRecordDataset:
    dataset = load_dataset(filenames, labeled)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTO_TUNE)
    return dataset


def get_train_val_test_size(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    train_df = pd.read_csv(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'cassava-leaf-disease-classification-data/train.csv'))
    dataset_size = train_df.image_id.drop_duplicates().shape[0]
    assert train_ratio + val_ratio + test_ratio <= 1, 'Adjust the train, val, test split ratio'

    train_size = math.ceil(dataset_size * train_ratio)
    val_size = math.ceil(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size

    return train_size, val_size, test_size


def get_splited_data(all_dataset: tf.data.TFRecordDataset):
    train_size, val_size, test_size = get_train_val_test_size()
    train_dataset = all_dataset.take(train_size)
    remaining_dataset = all_dataset.skip(train_size)
    val_dataset = remaining_dataset.take(val_size)
    test_dataset = remaining_dataset.skip(val_size)
    return train_dataset, val_dataset, test_dataset
