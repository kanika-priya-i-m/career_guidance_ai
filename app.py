
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import warnings

# Suppress all warnings to clean up the Streamlit output
warnings.filterwarnings('ignore')

# --- 1. Data Loading --- #
@st.cache_resource
def load_data():
    df = pd.read_csv('student_career_guidance_dataset_5000.csv')
    df.columns = df.columns.str.strip() # Ensure column names are clean
    return df

# --- 2. KMeans Model Training and Preprocessing --- #
@st.cache_resource
def train_kmeans_model(df):
    numeric_cols_kmeans = [
        "Math", "Physics", "Chemistry", "Biology", "English",
        "Computer_Science", "History", "Attendance_pct", "Average_Marks", "Age"
    ]
    categorical_cols_kmeans = ["Gender", "Learning_Style", "Personality", "Budget"]

    preprocess_kmeans = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols_kmeans),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols_kmeans)
    ])

    # Fit preprocessor and transform data
    # Using a copy to avoid modifying the original dataframe during fitting
    X_kmeans = preprocess_kmeans.fit_transform(df[numeric_cols_kmeans + categorical_cols_kmeans].copy())
    kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10) # n_init=10 to suppress warning
    kmeans_model.fit(X_kmeans)
    return preprocess_kmeans, kmeans_model

# --- 3. RandomForest Model Training and Preprocessing --- #
@st.cache_resource
def train_rf_model(df):
    # Define features and target for RandomForest
    features_rf = [
        'Age','Gender','Interests','Strengths','Skills','Math','Physics',
        'Chemistry','Biology','English','Computer_Science','History',
        'Attendance_pct','Learning_Style','Average_Marks','Personality',
        'Goals','Budget'
    ]
    target_rf = 'Suggested_Career'

    X_rf = df[features_rf].copy()
    y_rf = df[target_rf].copy()

    # Label Encode categorical features for RandomForest
    rf_encoders = {}
    cat_cols_rf = X_rf.select_dtypes(include='object').columns
    for col in cat_cols_rf:
        enc = LabelEncoder()
        # Fit encoder on all unique values from the training data for the column
        X_rf[col] = enc.fit_transform(X_rf[col])
        rf_encoders[col] = enc
    
    # Label Encode target variable
    career_encoder = LabelEncoder()
    y_rf_encoded = career_encoder.fit_transform(y_rf)

    # Train RandomForest model
    rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
    rf_model.fit(X_rf, y_rf_encoded)

    return rf_model, rf_encoders, career_encoder, features_rf

# --- 4. Prediction Functions --- #
def recommend_career_kmeans(cluster_id):
    # Placeholder for actual career recommendations based on KMeans cluster
    # This should ideally map to careers found during EDA and clustering analysis
    cluster_careers = {
        0: ["IT Support", "Administrative Assistant", "Technical Support"],
        1: ["Software Engineer", "Data Scientist", "AI/ML Engineer"],
        2: ["Content Writer", "Media Designer", "Law"],
        3: ["Software Developer", "Product Manager", "UX Designer", "Researcher"]
    }
    return cluster_careers.get(cluster_id, ["General Career Path"])

def predict_career_rf(student_data_dict, rf_model, rf_encoders, career_encoder, features_rf):
    df_student = pd.DataFrame([student_data_dict])

    # Apply the same label encoding as during training for RF model
    for col in rf_encoders.keys():
        if col in df_student.columns:
            # Handle unseen labels: if a value is not in the encoder's classes,
            # assign it the label for an 'unknown' category if we had one, or a default.
            # For simplicity, we check if the label exists, if not, we use a placeholder
            # or re-fit the encoder with the new class (which is not ideal for deployment).
            # A better approach for deployment might be to use OneHotEncoder or a custom transformer
            # with handle_unknown='ignore' or a designated 'unknown' category.

            # For demonstration, we'll try to transform. If it fails, we'll use a default.
            try:
                df_student[col] = rf_encoders[col].transform(df_student[col])
            except ValueError:
                # If an unseen label is encountered, we'll try to extend the encoder classes.
                # This is a workaround for demonstration, in production, unseen data should be handled carefully.
                st.warning(f"Unseen label for column '{col}': '{df_student[col].iloc[0]}'. Attempting to add to encoder.")
                # Extend classes with the unseen label
                new_classes = np.append(rf_encoders[col].classes_, df_student[col].iloc[0])
                rf_encoders[col].classes_ = np.sort(np.unique(new_classes))
                df_student[col] = rf_encoders[col].transform(df_student[col])

    # Ensure the order of columns matches the training data
    df_student = df_student[features_rf]

    # Make prediction
    prediction_encoded = rf_model.predict(df_student)

    # Inverse transform the prediction to get the career name
    predicted_career = career_encoder.inverse_transform(prediction_encoded)

    return predicted_career[0]

# --- Streamlit UI --- #
st.set_page_config(layout="wide", page_title="Career Guidance System")
st.title("Career Guidance System")
st.markdown("Fill in the details below to get personalized career recommendations!")

# Load data and train models
df = load_data()
preprocess_kmeans, kmeans_model = train_kmeans_model(df.copy()) # Pass a copy to avoid modification
rf_model, rf_encoders, career_encoder, features_rf = train_rf_model(df.copy()) # Pass a copy to avoid modification

# Collect user inputs
with st.form("career_form"):
    st.header("Student Information")
    age = st.number_input("Age", min_value=14, max_value=25, value=18)
    gender = st.selectbox("Gender", df['Gender'].unique())
    # For multi-value categorical inputs like Interests, Strengths, Skills
    # We'll use text input and rely on the model's LabelEncoder to handle new combinations
    interests = st.text_input("Interests (e.g., Technology, Arts)", "Technology, Problem-solving")
    strengths = st.text_input("Strengths (e.g., Logical Thinking, Creativity)", "Logical Thinking")
    skills = st.text_input("Skills (e.g., Python;SQL;Marketing)", "Python;SQL") # User enters combined skills
    
    st.subheader("Academic Performance (Scores out of 100)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        math_score = st.number_input("Math Score", min_value=0, max_value=100, value=75)
        biology_score = st.number_input("Biology Score", min_value=0, max_value=100, value=70)
    with col2:
        physics_score = st.number_input("Physics Score", min_value=0, max_value=100, value=70)
        english_score = st.number_input("English Score", min_value=0, max_value=100, value=70)
    with col3:
        chemistry_score = st.number_input("Chemistry Score", min_value=0, max_value=100, value=65)
        cs_score = st.number_input("Computer Science Score", min_value=0, max_value=100, value=80)
    with col4:
        history_score = st.number_input("History Score", min_value=0, max_value=100, value=55)

    attendance_pct = st.slider("Attendance Percentage", min_value=0.0, max_value=100.0, value=85.0)
    average_marks = st.number_input("Average Marks", min_value=0.0, max_value=100.0, value=72.0)

    st.subheader("Personal Attributes")
    learning_style = st.selectbox("Learning Style", df['Learning_Style'].unique())
    personality = st.selectbox("Personality", df['Personality'].unique())
    goals = st.selectbox("Goals", df['Goals'].unique())
    budget = st.selectbox("Budget", df['Budget'].unique())

    submitted = st.form_submit_button("Get Recommendations")

    if submitted:
        student_input = {
            'Age': age,
            'Gender': gender,
            'Interests': interests,
            'Strengths': strengths,
            'Skills': skills, # Pass raw string for RF model
            'Math': math_score,
            'Physics': physics_score,
            'Chemistry': chemistry_score,
            'Biology': biology_score,
            'English': english_score,
            'Computer_Science': cs_score,
            'History': history_score,
            'Attendance_pct': attendance_pct,
            'Learning_Style': learning_style,
            'Average_Marks': average_marks,
            'Personality': personality,
            'Goals': goals,
            'Budget': budget
        }

        # --- KMeans Prediction --- 
        # Filter student_input for only the columns expected by the KMeans preprocessor
        kmeans_numeric_cols = ["Math", "Physics", "Chemistry", "Biology", "English", "Computer_Science", "History", "Attendance_pct", "Average_Marks", "Age"]
        kmeans_categorical_cols = ["Gender", "Learning_Style", "Personality", "Budget"]
        
        # Create a DataFrame for KMeans input, ensuring all expected columns are present
        # and in the correct order as per the preprocessor's training
        kmeans_input_data = {col: [student_input[col]] for col in kmeans_numeric_cols + kmeans_categorical_cols}
        kmeans_input_df = pd.DataFrame(kmeans_input_data)

        X_new_kmeans = preprocess_kmeans.transform(kmeans_input_df)
        predicted_cluster = kmeans_model.predict(X_new_kmeans)[0]
        kmeans_recommendations = recommend_career_kmeans(predicted_cluster)

        st.subheader("Recommendations based on Clustering (KMeans)")
        st.write(f"Based on your profile, you are in **Cluster {predicted_cluster}**.")
        st.write(f"Suggested career paths from this cluster: {', '.join(kmeans_recommendations)}")

        # --- RandomForest Prediction ---
        rf_recommendation = predict_career_rf(student_input, rf_model, rf_encoders, career_encoder, features_rf)

        st.subheader("Personalized Career Recommendation (Random Forest)")
        st.write(f"Based on our trained model, your personalized career recommendation is: **{rf_recommendation}**")

st.markdown("---Disclaimer: This system provides recommendations based on a trained model and may not always be perfectly accurate. Always consult with a career counselor. ---")
