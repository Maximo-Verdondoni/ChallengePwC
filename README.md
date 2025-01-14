# Predictive Salary Model

This repository contains a machine learning model designed to predict salaries based on various employee features such as gender, education level, and years of experience. The model is built using **scikit-learn** and exposes a REST API through **FastAPI** to make predictions for new data.

## Table of Contents
- [Model Overview](#model-overview)
- [Data Preprocessing](#data-preprocessing)
- [Model Development](#model-development)
- [API Usage](#api-usage)
- [Installation](#installation)
- [Testing](#testing)
- [License](#license)

## Model Overview

The model predicts the salary of an employee based on the following features:
- **Gender**: The gender of the employee (Male or Female)
- **Education Level**: The highest level of education attained (e.g., Bachelor's, Master's, PhD)
- **Years of Experience**: The number of years the employee has been working in their field

The model was trained using the provided data and is designed to provide salary predictions.

## Data Preprocessing

Before training the model, the data underwent several preprocessing steps:
1. **Handling Missing Data**: Any missing values were imputed or removed, depending on the context.
2. **Feature Encoding**: Categorical features such as Gender and Education Level were mapped to numerical values.
3. **Feature Scaling**: The features were scaled using a **StandardScaler** to standardize the range of feature values and improve model performance.
4. **Train-Test Split**: The data was split into training and testing sets to evaluate model performance effectively (80% train & 20% test).

## Model Development

The following steps were involved in developing the predictive model:
1. **Model Selection**: A **Linear Regression** model was chosen due to its simplicity and effectiveness for this type of problem.
2. **Model Training**: The model was trained using the preprocessed data.
3. **Model Evaluation**: Performance was assessed using various metrics, including including analysis of assumptions to be met by the model.

## API Usage

The model is exposed via a REST API, built using **FastAPI**, that allows users to input employee data and receive salary predictions.

### How to Use the API
1. **Run the API**:
   - Clone the repository and navigate to the `api` directory.
   - Install the required dependencies from `requirements.txt`.
   - Start the server using the command:
     ```bash
     uvicorn api.main:app --reload
     ```

2. **Make a Prediction**:
   - Send a `POST` request to `http://127.0.0.1:8000/predict/` with the following JSON body:
     ```json
     {
       "gender": "Male",
       "education_level": "Master's",
       "years_of_experience": 5
     }
     ```

   - Example using **requests** in Python:
     ```python
     import requests

     url = "http://127.0.0.1:8000/predict/"

     data = {
         "gender": "Male",
         "education_level": "Master's",
         "years_of_experience": 5
     }

     response = requests.post(url, json=data)

     if response.status_code == 200:
         print("Predicted Salary:", response.json())
     else:
         print(f"Error: {response.status_code}")
         print(response.text)
     ```

### Possible Errors:
- **Invalid Data**: If the input contains invalid values (e.g., a non-existent gender), the API will raise an error with a detailed message.

## Installation

To run the model and the API locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ChallengePwC.git
   ```

2. Navigate to the project directory and install the dependencies:
   ```bash
   cd ChallengePwC
   pip install -r requirements.txt
   ```

3. Run the FastAPI server:
   ```bash
   uvicorn api.main:app --reload
   ```

## Funcionality
The work is divided into 2 .ipynb (EDA.ipynb and Model.ipynb) to see the analysis, cleaning and the work done to the dataset to see the model, its metrics and API use Model ipynb
You can check the source code of the api in the "api" folder and you can check the source code of the model in "LinearRegression" folder

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
