from classifier import CarOrientationClassifier

def main():
    data_dir = 'data'  # Data directory
    classifier = CarOrientationClassifier(data_dir)
    
    classifier.load_data()

    # Define hyperparameters for grid search
    learning_rates = [0.0001, 0.001, 0.01]
    batch_sizes = [16, 32, 64]

    best_params = classifier.manual_grid_search(learning_rates, batch_sizes, epochs=25)

    # Final training with best parameters
    classifier.create_model(learning_rate=best_params['learning_rate'])
    classifier.fit(epochs=25, batch_size=best_params['batch_size'])
    classifier.plot_performance(save_path='outputs')
    classifier.plot_confusion_matrix(save_path='outputs/confusion_matrix')
    classifier.save_model('outputs/best_car_orientation_classifier.h5')

if __name__ == "__main__":
    main()
