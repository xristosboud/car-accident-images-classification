import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class CarOrientationClassifier:
    def __init__(self, data_dir, target_size=(224, 224), num_classes=4):
        self.data_dir = data_dir
        self.target_size = target_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.class_names = None

    def load_data(self):
        """Load and preprocess the data."""
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        val_datagen = ImageDataGenerator(rescale=1.0 / 255)

        # Load training and validation datasets
        self.train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'train'),
            target_size=self.target_size,
            batch_size=32,
            class_mode='categorical'
        )

        self.val_generator = val_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'val'),
            target_size=self.target_size,
            batch_size=32,
            class_mode='categorical'
        )

        self.class_names = self.train_generator.class_indices
        print(f"Classes: {self.class_names}")

    def create_model(self, learning_rate=0.001):
        """Create and compile the Keras model."""
        base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False

        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(self.num_classes, activation='softmax')  # 4 classes
        ])

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def fit(self, epochs=25, batch_size=32):
        """Fit the model to the training data."""
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            batch_size=batch_size,
            verbose=1
        )

    def plot_performance(self, save_path=None):
        """Plot training and validation accuracy and loss over epochs."""
        if self.history is None:
            print("No training history found. Train the model first.")
            return
        
        # Plot accuracy
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")

        plt.show()

    def plot_confusion_matrix(self, save_path=None):
        """Evaluate model on validation data and plot confusion matrix."""
        # Get true labels and predictions
        y_true = self.val_generator.classes
        y_pred = np.argmax(self.model.predict(self.val_generator), axis=-1)

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        
        # Plot the confusion matrix
        plt.figure(figsize=(8, 8))
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix")

        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()


    def manual_grid_search(self, learning_rates, batch_sizes, epochs=25):
        """Perform manual grid search for hyperparameter tuning."""
        best_val_acc = 0
        best_params = {}

        for lr in learning_rates:
            for bs in batch_sizes:
                print(f"Training model with learning rate: {lr} and batch size: {bs}")
                
                self.create_model(learning_rate=lr)
                self.fit(epochs=epochs, batch_size=bs)
                
                # Get the best validation accuracy
                val_acc = max(self.history.history['val_accuracy'])
                print(f"Validation Accuracy: {val_acc}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_params = {'learning_rate': lr, 'batch_size': bs}

        print("Best Result:")
        print(f"Learning Rate: {best_params['learning_rate']}, Batch Size: {best_params['batch_size']}, Validation Accuracy: {best_val_acc}")
        return best_params

    def save_model(self, model_path=None):
        """Save the trained model."""
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
