import tensorflow as tf

from model_constants import IMAGE_SIZE

initial_training_lr = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_training_lr,
                                                             decay_steps=20, decay_rate=0.96, staircase=True)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("melanoma_model.h5", save_best_only=True)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)


def make_model():
    base_model = tf.keras.applications.Xception(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet")

    base_model.trainable = False

    inputs = tf.keras.layers.Input([*IMAGE_SIZE, 3])

    x = tf.keras.applications.xception.preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.7)(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss="categorical_crossentropy",
                  metrics=tf.keras.metrics.AUC(name="auc"))

    return model