# Student Performance Predictor

## Project Overview

This project is a machine learning-based Student Performance Predictor that predicts a student's expected final marks based on academic factors such as study hours, attendance, previous marks, assignment score, and test score.

The project covers an end-to-end machine learning workflow, from data generation and exploration to model training, evaluation, model saving, and deployment through a Streamlit web application.

## Features

- Predicts expected final marks from student academic data
- Uses Linear Regression and Random Forest Regression
- Evaluates models using MAE, MSE, RMSE, and R²
- Saves the trained Random Forest model using Joblib
- Provides an interactive Streamlit web interface
- Validates user inputs with appropriate ranges

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

## Machine Learning Models

### Linear Regression

Linear Regression was used to understand the relationship between the input features and final marks.

The model performed very well on this dataset because the synthetic target values were generated using a linear formula.

### Random Forest Regression

Random Forest Regression was used to learn relationships using multiple decision trees.

The trained Random Forest model was saved using Joblib and integrated into the Streamlit application for making predictions on new student data.

## Dataset

The dataset contains 100 student records.

### Input Features

| Feature | Description |
|---|---|
| study_hours | Number of hours studied |
| attendance | Student attendance percentage |
| previous_marks | Previous academic marks |
| assignment_score | Assignment score |
| test_score | Test score |

### Target

`final_marks` — expected final marks of the student.

## Model Evaluation

The models were evaluated using:

- **MAE (Mean Absolute Error)** — measures the average absolute prediction error.
- **MSE (Mean Squared Error)** — gives more importance to larger errors.
- **RMSE (Root Mean Squared Error)** — represents prediction error in the same unit as the target.
- **R² Score** — measures how much variation in the target is explained by the model.

### Evaluation Results

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | ~0 | ~0 | ~0 | 1.00 |
| Random Forest | 3.89 | 23.79 | 4.88 | 0.746 |

> Note: The Linear Regression model achieved nearly perfect results because the dataset was synthetically generated using an exact linear formula. These results should not be interpreted as representative of real-world student performance prediction.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/annu18coder/Student-performance-predictor.git

## Live Demo

[Open Student Performance Predictor](https://student-performance-predictor-xwbqwyvaddlsmidnbgpyc6.streamlit.app/)