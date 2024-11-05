# Speech Command Recognition using CNN
# Author: Claude
# Date: November 2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import librosa
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import warnings
warnings.filterwarnings('ignore')

# 1. Data Loading and Preprocessing Functions

def load_audio_file(file_path, duration=1.0, sr=16000):
    """
    Load and preprocess audio file
    """
    try:
        # Load audio file with a fixed duration
        y, sr = librosa.load(file_path, duration=duration, sr=sr)
        
        # Pad or truncate to ensure fixed length
        if len(y) > sr:
            y = y[:sr]
        else:
            y = np.pad(y, (0, max(0, sr - len(y))))
            
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None, None

def extract_mfcc(y, sr, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Extract MFCC features from audio signal
    """
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, 
                                   hop_length=hop_length)
        # Normalize MFCC
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
        return mfcc
    except:
        return None

def augment_audio(y, sr):
    """
    Apply random augmentation to audio signal
    """
    # Random time shift
    shift = np.random.randint(-sr//10, sr//10)
    y_shifted = np.roll(y, shift)
    if shift > 0:
        y_shifted[:shift] = 0
    else:
        y_shifted[shift:] = 0
    
    # Random pitch shift
    n_steps = np.random.randint(-4, 4)
    y_pitched = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    
    # Add random noise
    noise_factor = np.random.uniform(0, 0.01)
    noise = np.random.normal(0, 1, len(y))
    y_noisy = y + noise_factor * noise
    
    # Randomly choose one augmentation
    augmented = np.random.choice([y_shifted, y_pitched, y_noisy])
    return augmented

def prepare_dataset(data_path, classes):
    """
    Prepare dataset with features and labels
    """
    features = []
    labels = []
    
    for idx, class_name in enumerate(classes):
        class_path = os.path.join(data_path, class_name)
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            
            # Load and preprocess audio
            y, sr = load_audio_file(file_path)
            if y is None:
                continue
                
            # Extract MFCC
            mfcc = extract_mfcc(y, sr)
            if mfcc is None:
                continue
                
            features.append(mfcc)
            labels.append(idx)
            
            # Add augmented version
            if np.random.random() < 0.3:  # 30% chance of augmentation
                y_aug = augment_audio(y, sr)
                mfcc_aug = extract_mfcc(y_aug, sr)
                if mfcc_aug is not None:
                    features.append(mfcc_aug)
                    labels.append(idx)
    
    features = np.array(features)
    labels = np.array(labels)
    
    # Resize features to target shape (64, 64)
    features_resized = []
    for feature in features:
        resized = tf.image.resize(feature[..., np.newaxis], [64, 64]).numpy()
        features_resized.append(resized)
    
    features = np.array(features_resized)
    
    return features, labels

# 2. Model Definition

def create_cnn_model(input_shape, num_classes):
    """
    Create CNN model for speech recognition
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Learning rate scheduler
def lr_schedule(epoch):
    initial_lr = 0.001
    if epoch < 10:
        return initial_lr
    else:
        return initial_lr * tf.math.exp(-0.1 * (epoch - 10))

# 3. Training and Evaluation Functions

def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
    """
    Train the model with early stopping and learning rate scheduling
    """
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        LearningRateScheduler(lr_schedule)
    ]
    
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    
    return history

def plot_training_history(history):
    """
    Plot training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'])
    
    # Plot loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'])
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# 4. Main Execution

def main():
    # Define parameters
    data_path = "path_to_dataset"  # Update with actual path
    classes = ['zero', 'one']  # Update with actual classes
    
    # Prepare dataset
    print("Preparing dataset...")
    features, labels = prepare_dataset(data_path, classes)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)
    
    # Create and compile model
    print("Creating model...")
    model = create_cnn_model(input_shape=(64, 64, 1), num_classes=len(classes))
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Train model
    print("Training model...")
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Generate predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    # Plot results
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred, classes)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))

if __name__ == "__main__":
    main()
