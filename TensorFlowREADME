# Breast Cancer Wisconsin (Diagnostic) Neural Network Classification

This project utilizes the Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository to build and evaluate a neural network model for classifying breast cancer as either malignant or benign.

## Dataset Overview

- **Name**: Breast Cancer Wisconsin (Diagnostic)
- **UCI Repository ID**: 17
- **Number of Instances**: 569
- **Number of Features**: 30 (excluding the ID and target variables)
- **Target Variable**: Diagnosis (M = Malignant, B = Benign)
- **Feature Types**: Real-valued features computed from digitized images of breast mass cell nuclei

## Project Structure

- **Data Preprocessing**: The dataset is preprocessed by:
  - Scaling features using `StandardScaler`.
  - Label encoding the target variable (`Diagnosis`), converting 'M' to 1 and 'B' to 0.

- **Model Building**: A neural network model is constructed with the following architecture:
  - Input Layer: 128 neurons, ReLU activation
  - Hidden Layers: 64 neurons, ReLU activation; 16 neurons, ReLU activation
  - Output Layer: 1 neuron, Sigmoid activation

- **Model Training**: The neural network is trained with:
  - Early stopping to prevent overfitting
  - Learning rate reduction based on validation loss
  - Epochs: 10, Batch Size: 32

- **Model Evaluation**: The model is evaluated using:
  - Validation Loss: 0.0396
  - Validation Accuracy: 0.9902

## Results Summary

- **Validation Loss**: 0.0396
- **Validation Accuracy**: 0.9902

## Dependencies

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- UCIMLRepo

## Installation

To run the project, ensure that you have the necessary dependencies installed. You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow ucimlrepo
