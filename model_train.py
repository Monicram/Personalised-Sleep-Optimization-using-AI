import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load dataset (Ensure dataset is in a structured folder format: "sleep_disorder" and "no_sleep_disorder")
datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    "eye_images/",
    target_size=(64, 64),
    color_mode="grayscale",
    batch_size=32,
    class_mode="binary",
    subset="training",
)

val_generator = datagen.flow_from_directory(
    "eye_images/",
    target_size=(64, 64),
    color_mode="grayscale",
    batch_size=32,
    class_mode="binary",
    subset="validation",
)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Save the model
model.save("sleep_disorder_model.h5")
print("Model saved successfully!")
