# Breast Cancer Wisconsin (Diagnostic) Classification Project

This project uses the Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository to build and evaluate machine learning models and a neural network for the classification of breast cancer as either malignant or benign.

## Dataset Overview

- **Name:** Breast Cancer Wisconsin (Diagnostic)
- **UCI Repository ID:** 17
- **Number of Instances:** 569
- **Number of Features:** 30 (excluding the ID and target variables)
- **Target Variable:** Diagnosis (M = Malignant, B = Benign)
- **Feature Types:** Real-valued features computed from digitized images of breast mass cell nuclei

## Project Structure

### Data Preprocessing
- **Scaling:** Features are scaled using `StandardScaler`.
- **Target Encoding:** The target variable (Diagnosis) is label-encoded, converting 'M' to 1 (Malignant) and 'B' to 0 (Benign).

### Model Evaluation

#### 1. **Machine Learning Models:**
   The following models are evaluated for classification:
   - **Logistic Regression**
   - **K-Nearest Neighbors (KNN)**
   - **Support Vector Machine (SVM)**
   - **Decision Tree**
   - **Random Forest**
  
   **Evaluation Metrics:**  
   The models are evaluated based on the following metrics:
   - **Accuracy**
   - **Confusion Matrix**
   - **Classification Report (Precision, Recall, F1-Score)**


## Results Summary

| Model                  | Accuracy | Precision | Recall  | F1-Score |
|------------------------|----------|-----------|---------|----------|
| Logistic Regression     | 1.0000   | 1.00      | 1.00    | 1.00     |
| K-Nearest Neighbors     | 0.9737   | 0.97      | 0.97    | 0.97     |
| Support Vector Machine  | 1.0000   | 1.00      | 1.00    | 1.00     |
| Decision Tree           | 0.9912   | 0.99      | 0.99    | 0.99     |
| Random Forest           | 1.0000   | 1.00      | 1.00    | 1.00     |

### Best Model:
- **Logistic Regression**, **Support Vector Machine** and **Random Forest** achieved perfect accuracy (1.0000) on the test set.

#### 2. **Neural Network Model:**
   A neural network model is built for classification, with the following architecture:
   - **Input Layer:** 128 neurons, Tanh activation
   - **Hidden Layers:**
       64 neurons, Tanh activation;
       16 neurons, Tanh activation
   - **Output Layer:** 1 neuron, Sigmoid activation
   
   The model is trained using:
   - Early stopping to prevent overfitting
   - Learning rate reduction based on validation loss
   - **Epochs:** 8
   - **Batch Size:** 16

   **Evaluation Metrics:**
- **Validation Loss:** 0.0297
- **Validation Accuracy:** 1.0000
  
- **Training Loss:** 0.0538
- **Training Accuracy:** 0.9890

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
