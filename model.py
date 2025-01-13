import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize
import shap

# 1. Define dataset path and parameters
dataset_path = 'cats-and-dogs'
subdirs = ['train', 'val']
categories = ['cat', 'dog']
img_size = (128, 128)

# visualize_sample_images(dataset_path, categories)

# 2. Data visualization function
def visualize_sample_images(dataset_path, categories, subdir='train'):
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for i, category in enumerate(categories):
        sample_img_path = os.path.join(dataset_path, subdir, category, os.listdir(os.path.join(dataset_path, subdir, category))[0])
        img = plt.imread(sample_img_path)
        axes[i].imshow(img)
        axes[i].set_title(category)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

visualize_sample_images(dataset_path, categories)

# 3. Load and preprocess the data
data = []
for subdir in subdirs:
    for category in categories:
        path = os.path.join(dataset_path, subdir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_array = plt.imread(img_path)
                img_resized = resize(img_array, img_size, anti_aliasing=True)
                data.append([img_resized, class_num])
            except Exception as e:
                print(f"Failed to load image {img} in {path}: {e}")

X, y = zip(*data)
X = np.array(X)
y = np.array(y)

# 4. Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Loaded {len(data)} images")
print(f"Training set: {len(X_train)} images")
print(f"Validation set: {len(X_val)} images")

# 5. Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# 6. Build the CNN model
model = Sequential([
    # Convolutional base
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    # Flatten and fully connected layers
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer=SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()

# 7. Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=10,
    steps_per_epoch=len(X_train) // 32
)

# 8. Plot training history
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history)

# 9. Save the model
model.save('cats_dogs_cnn.keras')

# 10. Explain predictions using SHAP
def explain_with_shap(model, X_sample, num_samples=5):
    """
    Explains the predictions of the model using SHAP.
    :param model: Trained model
    :param X_sample: Sample input data to explain
    :param num_samples: Number of data samples to explain
    """
    # Select a few samples for explanation
    sample_inputs = X_sample[:num_samples]

    # Initialize the SHAP explainer with a TensorFlow model
    explainer = shap.DeepExplainer(model, sample_inputs)

    # Compute SHAP values
    shap_values = explainer.shap_values(sample_inputs)

    # Plot SHAP values for the first sample
    for i, sample in enumerate(sample_inputs):
        shap.image_plot(shap_values, np.expand_dims(sample, axis=0), show=True)

# Load the saved model and explain predictions for validation samples
new_model = load_model('cats_dogs_cnn.keras')
explain_with_shap(new_model, X_val, num_samples=5)
