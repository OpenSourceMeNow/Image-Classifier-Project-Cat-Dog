import os
import numpy as np
import scipy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

def load_and_preprocess_data(train_dir, test_dir, img_size=(150, 150)):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=32,
        class_mode='binary',
        subset='training')
    
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=32,
        class_mode='binary',
        subset='validation')
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=32,
        class_mode=None,  # No labels for test data
        shuffle=False)
    
    return train_generator, validation_generator, test_generator

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),  # L2 regularization
        Dropout(0.5),  # Dropout
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

train_dir = '/mnt/e/Cats&Dogs/train/train'
test_dir = '/mnt/e/Cats&Dogs/test1'

train_generator, validation_generator, test_generator = load_and_preprocess_data(train_dir, test_dir)

model = build_model()

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10,  # You can adjust this
    callbacks=[early_stopping]  # Early Stopping
)

# Evaluate on validation set
val_loss, val_acc = model.evaluate(validation_generator)
print(f'Validation loss: {val_loss}, Validation accuracy: {val_acc}')

# Save the model
model_save_path = '/mnt/e/Cats&Dogs/my_model.h5'
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Predict on test set (optional)
predictions = model.predict(test_generator)
# Do something with predictions, e.g., save them or analyze them further