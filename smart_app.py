# Smart Data Preprocessing and Imbalance Detection Web App

# Step 1: Import Required Libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Step 2: App Title and Description
st.set_page_config(page_title="Smart Data Preprocessor", layout="wide")
st.title("🧠 Smart Data Preprocessing Web App")
st.markdown("""
This app allows you to:
- Upload a CSV file
- Analyze missing values and class imbalance
- Apply imputation and resampling
- Download cleaned dataset
""")

# Step 3: File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # Step 4: Summary
    st.markdown("---")
    st.subheader("📈 Dataset Summary")
    st.write("Shape:", df.shape)
    st.write("Data Types:")
    st.write(df.dtypes)

    # Step 5: Missing Value Analysis
    st.markdown("---")
    st.subheader("🧩 Missing Values")
    missing_counts = df.isnull().sum()
    st.write(missing_counts[missing_counts > 0])

    if missing_counts.sum() > 0:
        st.markdown("### Choose Imputation Method")
        method = st.selectbox("Select imputation strategy", ["Mean", "Median", "Mode", "Drop Rows"])

        if st.button("Apply Imputation"):
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if method == "Mean":
                        imp = SimpleImputer(strategy="mean")
                    elif method == "Median":
                        imp = SimpleImputer(strategy="median")
                    elif method == "Mode":
                        imp = SimpleImputer(strategy="most_frequent")
                    elif method == "Drop Rows":
                        df.dropna(inplace=True)
                        break
                    df[[col]] = imp.fit_transform(df[[col]])
            st.success("Imputation applied successfully!")

    # Step 6: Class Imbalance Handling
    st.markdown("---")
    st.subheader("⚖️ Class Imbalance Detection")
    target_col = st.selectbox("Select Target Column (Categorical)", df.columns)
    class_counts = df[target_col].value_counts()
    st.bar_chart(class_counts)

    if st.checkbox("Fix Class Imbalance"):
        technique = st.radio("Choose resampling method", ["SMOTE", "Random Oversample", "Random Undersample"])
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        try:
            X = pd.get_dummies(X)  # One-hot encode categorical features
            if technique == "SMOTE":
                sampler = SMOTE()
            elif technique == "Random Oversample":
                sampler = RandomOverSampler()
            elif technique == "Random Undersample":
                sampler = RandomUnderSampler()
            X_res, y_res = sampler.fit_resample(X, y)
            df = pd.DataFrame(X_res)
            df[target_col] = y_res.values
            st.success("Class imbalance handled successfully!")
            st.bar_chart(y_res.value_counts())
        except Exception as e:
            st.error(f"Error: {e}")

    # Step 7: Download Cleaned Dataset
    st.markdown("---")
    st.subheader("💾 Download Cleaned Dataset")
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(df)
    st.download_button("Download Processed CSV", csv, "cleaned_data.csv", "text/csv")

else:
    st.warning("Please upload a CSV file to proceed.")

