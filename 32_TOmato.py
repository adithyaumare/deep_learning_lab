# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %%
base_dir = r"D:\chrome download\DL_LAB_EXAM\DL_LAB_EXAM\Datasets\Plant_data\Tomato"
IMG_SIZE = (128, 128)
EPOCHS = 5
LEARNING_RATE = 0.0001

# %%
def build_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    opt = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# %%
def load_data(batch_size):
    datagen = ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_directory(os.path.join(base_dir, 'train'), target_size=IMG_SIZE, batch_size=batch_size, class_mode='categorical')
    val_gen = datagen.flow_from_directory(os.path.join(base_dir, 'val'), target_size=IMG_SIZE, batch_size=batch_size, class_mode='categorical')
    test_gen = datagen.flow_from_directory(os.path.join(base_dir, 'test'), target_size=IMG_SIZE, batch_size=batch_size, class_mode='categorical', shuffle=False)
    return train_gen, val_gen, test_gen

# %%
def plot_history(history, batch_size):
    plt.figure(figsize=(12, 5))
    plt.suptitle(f'Batch Size: {batch_size}', fontsize=14)

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


# %%
def evaluate_model(model, generator, name):
    loss, acc = model.evaluate(generator)
    print(f"📌 {name} Accuracy: {acc:.4f}, Loss: {loss:.4f}")
    return acc, loss

# %%
def plot_confusion_matrix(model, generator):
    y_true = generator.classes
    y_pred = model.predict(generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    class_names = list(generator.class_indices.keys())

    cm = confusion_matrix(y_true, y_pred_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# %%
for batch_size in [32, 64]:
    print(f"\n🔧 Training with batch size: {batch_size}\n{'-'*50}")
    train_gen, val_gen, test_gen = load_data(batch_size)
    model = build_model(num_classes=len(train_gen.class_indices))
    
    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# %%
plot_history(history, batch_size)

# %%
    evaluate_model(model, train_gen, "Training")
    evaluate_model(model, val_gen, "Validation")
    evaluate_model(model, test_gen, "Test")

# %%
 plot_confusion_matrix(model, test_gen)


