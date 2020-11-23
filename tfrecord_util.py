from functools import partial
import tensorflow as tf
from model_constants import IMAGE_SIZE, AUTO_TUNE, BATCH_SIZE


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
        }
        if labeled
        else {"image": tf.io.FixedLenFeature([], tf.string),}
    )

    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_img(example["image"])
    if labeled:
        label = tf.cast(example["target"], tf.int32)
        return image, label

    return image


def load_dataset(filenames, labeled=True):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTO_TUNE)
    return dataset


def get_dataset(filenames, labeled=True):
    dataset = load_dataset(filenames, labeled)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTO_TUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset