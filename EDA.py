import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px

# Function to load data
@st.cache
def load_data(file):
    data = pd.read_csv(file)
    return data

# Function to display summary statistics
def display_summary(data):
    st.write("Summary Statistics:")
    st.write(data.describe())

# Function to handle missing values
def handle_missing_values(data):
    st.write("Handling Missing Values:")
    missing_counts = data.isnull().sum()
    if missing_counts.sum() == 0:
        st.write("No missing values found.")
    else:
        st.write("Missing Values Counts:")
        st.write(missing_counts)
        if st.button("Drop Missing Values"):
            data = data.dropna()
            st.write("Missing values dropped.")
    return data

# Function for encoding categorical variables
def encode_categorical(data):
    st.write("Encoding Categorical Variables:")
    categorical_columns = data.select_dtypes(include=['object']).columns
    if len(categorical_columns) == 0:
        st.write("No categorical columns found.")
    else:
        for col in categorical_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
        st.write("Categorical variables encoded.")
    return data

# Function for scaling numerical variables
def scale_numerical(data):
    st.write("Scaling Numerical Variables:")
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    if len(numerical_columns) == 0:
        st.write("No numerical columns found.")
    else:
        scaler = StandardScaler()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        st.write("Numerical variables scaled.")
    return data

# Function for univariate analysis
def univariate_analysis(data):
    st.write("Univariate Analysis:")
    st.subheader("Distribution of Numerical Columns")
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_columns:
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, ax=ax)
        st.pyplot(fig)
        plt.clf()

# Function for bivariate analysis
def bivariate_analysis(data):
    st.write("Bivariate Analysis:")
    st.subheader("Correlation Heatmap of Numerical Columns")
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = data[numerical_columns].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    plt.clf()

# Function for machine learning model training and evaluation
def train_and_evaluate_model(data):
    st.write("Machine Learning Model Training and Evaluation:")
    st.subheader("Select Target Variable")
    target_variable = st.selectbox("Select the target variable", data.columns)
    if target_variable in data.columns:
        X = data.drop(columns=[target_variable])
        y = data[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write("Training Random Forest Classifier...")
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        st.write("Model Evaluation Results:")
        st.write(classification_report(y_test, y_pred))
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', ax=ax)
        st.pyplot(fig)
        plt.clf()
        st.subheader("Feature Importances")
        feature_importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.write(feature_importances)

# Streamlit UI
def main():
    st.title("EDA and Machine Learning Integration App")

    st.write("# Upload your dataset (CSV format, max 200MB)")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", accept_multiple_files=False, max_upload_size=200*1024*1024)
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)

        st.write("## Data Overview")
        st.write(df.head())

        st.write("## Data Cleaning and Preprocessing")
        
        st.write("### Summary Statistics")
        display_summary(df)

        st.write("### Handling Missing Values")
        df = handle_missing_values(df)

        st.write("### Encoding Categorical Variables")
        df = encode_categorical(df)

        st.write("### Scaling Numerical Variables")
        df = scale_numerical(df)

        st.write("## Exploratory Data Analysis")
        
        st.write("### Univariate Analysis")
        univariate_analysis(df)

        st.write("### Bivariate Analysis")
        bivariate_analysis(df)

        st.write("## Machine Learning Model Training and Evaluation")
        train_and_evaluate_model(df)

if _name_ == "_main_":
    main()
