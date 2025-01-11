import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    #Dropping Age because of its high correlation with "Year of experience" (multicollinearity)
    #Dropping Job Title because of its high number of unique values in the dataset
    df = df.drop(columns=['Age', "Job Title"])

    #Convert the categorics variables to numericals
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['Education Level'] = df['Education Level'].map({
        "Bachelor's": 1,
        "Master's": 2,
        "PhD": 3
    })

    #Setting independents and dependent variables
    X = df.drop(columns=['Salary'])
    y = df['Salary']

    # Splitting the dataset into train and test dataset (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Standardizing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test


    