import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from model_constants import LABEL_IDX_SHORT_NAME_MAPPING, FILE_NAMES
from tfrecord_util import get_dataset


def test_read_tf_record():
    tf_path = './resources/ld_test00-1.tfrec'
    dataset = tf.data.TFRecordDataset(tf_path)
    for raw_record in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)


def test_load_dataset_and_show():
    all_dataset = get_dataset(FILE_NAMES)
    image_batch, label_batch = next(iter(all_dataset))

    show_batch(image_batch.numpy(), label_batch.numpy())


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5,5, n+1)
        plt.imshow(image_batch[n]/255.0)
        if label_batch[n] is not None:
            label_id = np.argmax(label_batch[n])
            plt.title(f"{LABEL_IDX_SHORT_NAME_MAPPING[label_id]}")
        else:
            plt.title('No Label')
        plt.axis("off")
    plt.show()

