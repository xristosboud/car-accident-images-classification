import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

class CarOrientationPredictor:
    def __init__(self, model_path, class_names):
        """Initialize the predictor with the model path and class names."""
        self.model = load_model(model_path)
        self.class_names = class_names

    def preprocess_image(self, img_path, target_size=(224, 224)):
        """Load and preprocess the image."""
        img = load_img(img_path, target_size=target_size)  # Load image
        img_array = img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array

    def predict(self, img_array):
        """Make a prediction for a single image."""
        predictions = self.model.predict(img_array)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]  # Get the class index
        return self.class_names[predicted_class_idx]

    def predict_directory(self, dir_path):
        """Predict orientations for all images in the specified directory."""
        img_path_lst = []
        predicted_classes = []

        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(root, file)
                    img_path_lst.append(img_path)

                    img_array = self.preprocess_image(img_path)
                    predicted_class = self.predict(img_array)
                    predicted_classes.append(predicted_class)

        return img_path_lst, predicted_classes

    def save_predictions(self, img_paths, predictions, output_path):
        """Save predictions to an Excel file."""
        results_df = pd.DataFrame({
            'path': img_paths,
            'orientation': predictions
        })
        results_df.to_excel(output_path, index=False)
        print(f"Predictions saved to {output_path}")