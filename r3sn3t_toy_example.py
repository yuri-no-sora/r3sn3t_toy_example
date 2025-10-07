# r3sn3t_toy_example.py
"""
Toy ResNet Example (r3sn3t_toy_example)
======================================
Implements a minimal residual network (ResNet-like) on MNIST dataset.

H(x) = F(x, {W_i}) + x
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ------------------------------
# Residual Block Definition
# ------------------------------
def residual_block(x, filters, kernel_size=3):
    """
    A simple residual block with two conv layers and a skip connection.
    """
    shortcut = x  # identity path

    # Convolution path F(x)
    x = layers.Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)

    # Add skip connection H(x) = F(x) + x
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


# ------------------------------
# Build Toy ResNet Model
# ------------------------------
def build_toy_resnet(input_shape=(28, 28, 1), num_classes=10):
    inputs = keras.Input(shape=input_shape)

    # Stem
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(2)(x)

    # Residual blocks
    x = residual_block(x, 32)
    x = residual_block(x, 32)

    # Classifier head
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name="toy_resnet")
    return model


# ------------------------------
# Train & Evaluate
# ------------------------------
def main():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # Build model
    model = build_toy_resnet()
    model.summary()

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Train
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=5,
        batch_size=128,
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    # Plot training results
    plt.figure(figsize=(6,4))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Toy ResNet Training Accuracy')
    plt.show()


if __name__ == "__main__":
    main()
