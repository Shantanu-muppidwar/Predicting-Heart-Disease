# Neural Networks Capstone: Predicting Heart Disease

## Overview
This project leverages neural networks to predict the likelihood of heart disease based on a diverse range of medical features. It encompasses data preprocessing, model development, and evaluation techniques, addressing a binary classification problem in the healthcare domain. The goal is to provide meaningful insights and optimize predictive accuracy through advanced deep learning methodologies.

## Dataset
The dataset comprises the following features:
- **age**: Age in years
- **sex**: Gender (1 = male, 0 = female)
- **cp**: Chest pain type (categorical: 0-3)
- **trestbps**: Resting blood pressure (in mm Hg)
- **chol**: Serum cholesterol in mg/dl
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg**: Resting electrocardiographic results (categorical: 0-2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise-induced angina (1 = yes, 0 = no)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: The slope of the peak exercise ST segment
- **ca**: Number of major vessels (0-3) colored by fluoroscopy
- **thal**: Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)
- **target**: Presence of heart disease (1 = yes, 0 = no)

## Key Steps
1. **Data Analysis and Preprocessing**: Conduct exploratory analysis, handle missing values, and prepare the dataset for training.
2. **Model Development**: Design and train a feedforward neural network using TensorFlow and Keras.
3. **Performance Evaluation**: Assess the model using metrics such as accuracy, precision, recall, and F1 score.
4. **Optimization**: Apply hyperparameter tuning and network adjustments to enhance model performance.

## Tools and Libraries
- **Python**: Primary programming language for implementation.
- **Pandas & NumPy**: Libraries for data manipulation and preprocessing.
- **Matplotlib & Seaborn**: Tools for visualization and exploratory data analysis.
- **TensorFlow/Keras**: Frameworks for building and training neural networks.
- **Scikit-learn**: Utilities for evaluation metrics and dataset splitting.

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd neural-networks-capstone
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch the notebook:
   ```bash
   jupyter notebook Neural_Networks_Capstone.ipynb
   ```

## Usage
- Ensure the dataset (`heart.csv`) is available in the project directory.
- Open `Neural_Networks_Capstone.ipynb` in Jupyter Notebook.
- Follow the step-by-step implementation to preprocess the data, train the model, and evaluate results.
- Modify hyperparameters or the network architecture to experiment with different configurations.

## Results
- **Model Accuracy**: 0.7582417582417582
- **Confusion Matrix**: Illustrates true positives, true negatives, and misclassifications.
- **Precision, Recall, and F1 Score**: Scores are 0.79, 0.76 and 0.78 respectively. Provides detailed insights into the model’s predictive performance.


