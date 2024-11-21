import os
import argparse
from predictor import CarOrientationPredictor

def main(model_path, image_dir, output_path):
    """Main function to predict car orientations."""
    class_names = ['front', 'back', 'side', 'inside']  # Define your class names here
    predictor = CarOrientationPredictor(model_path, class_names)

    img_paths, predictions = predictor.predict_directory(image_dir)
    predictor.save_predictions(img_paths, predictions, output_path)

if __name__ == "__main__":
    # Define paths directly in the code
    model_path = 'outputs/best_car_orientation_classifier.h5'  # Change this to your model path
    image_dir = 'data/claims'      # Change this to your image directory
    output_path = 'outputs/predictions.xlsx'  # Change this to your output path

    main(model_path, image_dir, output_path)
