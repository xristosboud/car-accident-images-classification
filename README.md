# Car Orientation Classification

This project is designed to classify car images based on their orientation (front, back, side, inside) using a deep learning model based on ResNet. The workflow includes training a model, predicting orientations from images, and analyzing the results.

## Project Structure

- **source/classifier.py**: Contains the `CarOrientationClassifier` class for training the model.
- **source/predictor.py**: Contains the `CarOrientationPredictor` class for predicting orientations from images.
- **source/train.py**: Contains the code for executing the training of the model
- **source/predict.py**: Contains the code for executing predictions using the trained model
- **source/ImageDataAnalysis.py**: Script to analyze predictions and generate summary statistics.
- **data/**: Directory to store the training and validation data.
- **outputs/**: Directory for saving output files, such as predictions and statistics.

## Install the required packages
pip install -r requirements.txt