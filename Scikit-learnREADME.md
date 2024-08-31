# Breast Cancer Wisconsin (Diagnostic) Classification

This project uses the Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository to build and evaluate several machine learning models for the classification of breast cancer as either malignant or benign.

## Dataset Overview

- **Name**: Breast Cancer Wisconsin (Diagnostic)
- **UCI Repository ID**: 17
- **Number of Instances**: 569
- **Number of Features**: 30 (excluding the ID and target variables)
- **Target Variable**: Diagnosis (M = Malignant, B = Benign)
- **Feature Types**: Real-valued features computed from digitized images of breast mass cell nuclei

## Project Structure

- **Data Preprocessing**: The dataset is preprocessed by scaling the features using `StandardScaler`. The target variable (`Diagnosis`) is label-encoded, converting 'M' to 1 and 'B' to 0.

- **Model Evaluation**: Five different machine learning models are evaluated:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Decision Tree
  - Random Forest

- **Metrics Used**: The models are evaluated based on the following metrics:
  - Accuracy
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-Score)

## Results Summary

| Model                 | Accuracy | Precision | Recall | F1-Score |
|-----------------------|----------|-----------|--------|----------|
| Logistic Regression   | 1.0000   | 1.00      | 1.00   | 1.00     |
| K-Nearest Neighbors   | 0.9825   | 0.98      | 0.98   | 0.98     |
| Support Vector Machine| 0.9912   | 0.99      | 0.99   | 0.99     |
| Decision Tree         | 0.9649   | 0.96      | 0.96   | 0.96     |
| Random Forest         | 0.9912   | 0.99      | 0.99   | 0.99     |

- **Best Model**: Logistic Regression achieved perfect accuracy on the test set, followed closely by Support Vector Machine and Random Forest.

## Dependencies

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- UCIMLRepo

## Installation

To run the project, ensure that you have the necessary dependencies installed. You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ucimlrepo
