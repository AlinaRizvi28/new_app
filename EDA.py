import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load data
@st.cache
def load_data(file):
    data = pd.read_csv(file)
    return data

# Function to display the summary statistics
def display_summary(data):
    st.write(data.describe())

# Function to handle missing values
def handle_missing_values(data):
    st.write("Missing Values Summary:")
    st.write(data.isnull().sum())
    if st.button("Drop Missing Values"):
        data = data.dropna()
        st.write("Missing values dropped.")
    return data

# Function for Univariate Analysis
def univariate_analysis(data, column):
    st.write(f"Distribution of {column}")
    plt.figure(figsize=(10, 4))
    sns.histplot(data[column], kde=True)
    st.pyplot(plt)
    plt.clf()

# Function for Bivariate Analysis
def bivariate_analysis(data, col1, col2):
    st.write(f"Bivariate Analysis between {col1} and {col2}")
    plt.figure(figsize=(10, 4))
    sns.scatterplot(x=data[col1], y=data[col2])
    st.pyplot(plt)
    plt.clf()

# Function for Multivariate Analysis
def multivariate_analysis(data):
    st.write("Pairplot for Multivariate Analysis")
    sns.pairplot(data)
    st.pyplot(plt)

# Function for Outlier Detection and Handling
def handle_outliers(data, column):
    st.write(f"Boxplot for {column}")
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=data[column])
    st.pyplot(plt)
    plt.clf()
    if st.button(f"Remove Outliers in {column}"):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
        st.write(f"Outliers in {column} removed.")
    return data

# Streamlit UI
st.title("Automated EDA Web App")
st.write("Upload your dataset (CSV format) to start the analysis.")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("Data loaded successfully!")
    
    st.subheader("Data Overview")
    st.write(df.head())

    st.subheader("Summary Statistics")
    display_summary(df)

    st.subheader("Handle Missing Values")
    df = handle_missing_values(df)

    st.subheader("Univariate Analysis")
    columns = df.select_dtypes(include=['float64', 'int64']).columns
    column = st.selectbox("Choose a column for univariate analysis", columns)
    if column:
        univariate_analysis(df, column)
    
    st.subheader("Bivariate Analysis")
    col1 = st.selectbox("Choose column 1", columns)
    col2 = st.selectbox("Choose column 2", columns)
    if col1 and col2:
        bivariate_analysis(df, col1, col2)

    st.subheader("Multivariate Analysis")
    if st.button("Generate Pairplot"):
        multivariate_analysis(df)

    st.subheader("Outlier Detection and Handling")
    outlier_column = st.selectbox("Choose a column for outlier detection", columns)
    if outlier_column:
        df = handle_outliers(df, outlier_column)

st.write("Thank you for using the Automated EDA Web App!")
