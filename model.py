import tensorflow as tf
from datetime import datetime
from model_constants import IMAGE_SIZE, AUTO_TUNE

initial_training_lr = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_training_lr,
                                                             decay_steps=20, decay_rate=0.96, staircase=True)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("cassava_leaf.h5", save_best_only=True)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir)


def augment(ds):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=[0.2, 0.3], width_factor=[0.2, 0.3])
    ])

    ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTO_TUNE)

    return ds.prefetch(buffer_size=AUTO_TUNE)


def make_model():
    base_model = tf.keras.applications.ResNet101V2(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights=None)

    base_model.trainable = False

    inputs = tf.keras.layers.Input([*IMAGE_SIZE, 3])

    x = tf.keras.applications.xception.preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.7)(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss="categorical_crossentropy",
                  metrics=tf.keras.metrics.AUC(name="auc"))

    return model