imimport streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Page styling with CSS
st.markdown("""
    <style>
        .main {
            background-color: #F0F0F0;
        }
        .stButton button {
            background-color: #FF6F61;
            color: white;
        }
        .stSidebar {
            background-color: #D3D3D3;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🌡️ Heart Disease Prediction App")
st.image("images2.png", width=500)
st.write("This app builds a machine learning model to predict heart disease! 💓")

# Load dataset
data = pd.read_csv("data.csv")

# Show the shape of the dataset
st.write("Shape of the dataset:", data.shape)

# Convert categorical variables to numeric
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Sidebar menu with updated design
menu = st.sidebar.radio("📊 Menu", ["Home", "Prediction Details"])

if menu == "Home":
    st.image("image3.png", width=550)
    st.header("📋 Tabular Data of Heart Disease Features")
    if st.checkbox("Show Tabular Data"):
        st.table(data.head(150))

    st.header("📊 Statistical Summary of the Dataframe")
    if st.checkbox("Show Statistics"):
        st.table(data.describe())

    st.header("📈 Correlation Graph")
    if st.checkbox("Show Correlation Graph"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap="viridis", ax=ax)  # Updated color palette
        st.pyplot(fig)

    st.header("📉 Visualizations")

    # Select graph type
    graph = st.selectbox("Choose the type of graph", ["Scatter Plot", "Bar Graph", "Histogram"])

    if graph == "Scatter Plot":
        x_col = st.selectbox("Select x-axis column", data.select_dtypes(include=[np.number]).columns)
        y_col = st.selectbox("Select y-axis column", data.select_dtypes(include=[np.number]).columns)
        hue_col = st.selectbox("Select hue column (optional)", ["None"] + list(data.select_dtypes(include=[object]).columns))
        hue_col = None if hue_col == "None" else hue_col

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue_col, ax=ax, palette="coolwarm")
        ax.set_title(f"Scatter Plot of {x_col} vs {y_col}")
        st.pyplot(fig)

    elif graph == "Bar Graph":
        categorical_columns = data.select_dtypes(include=[object]).columns
        
        if len(categorical_columns) > 0:
            column_to_plot = st.selectbox("Select column to plot", categorical_columns)
            
            if column_to_plot:
                fig, ax = plt.subplots(figsize=(12, 6))
                counts = data[column_to_plot].value_counts().reset_index()
                counts.columns = [column_to_plot, 'Count']

                sns.barplot(x=column_to_plot, y='Count', data=counts, ax=ax, palette="muted")
                ax.set_title(f'Count of Occurrences for {column_to_plot}')
                st.pyplot(fig)
        else:
            st.write("No categorical columns available for plotting.")

    elif graph == "Histogram":
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 0:
            column_to_plot = st.selectbox("Select column to plot", numeric_columns)
            
            if column_to_plot:
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.histplot(data[column_to_plot], kde=True, ax=ax, color="darkgreen")
                ax.set_title(f'Histogram of {column_to_plot}')
                ax.set_xlabel(column_to_plot)
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
        else:
            st.write("No numeric columns available for plotting.")

if menu == "Prediction Details":
    st.title("🧠 Heart Disease Prediction")
    
    # Prepare data for training
    features = ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Diabetes', 'Family History', 
                'Smoking', 'Obesity', 'Alcohol Consumption', 'Exercise Hours Per Week', 'Diet', 
                'Previous Heart Problems', 'Medication Use', 'Stress Level', 'Sedentary Hours Per Day', 
                'Income', 'BMI', 'Triglycerides', 'Physical Activity Days Per Week', 'Sleep Hours Per Day']
    
    X = data[features]
    y = data['Heart Attack Risk']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)
    st.write("Model Accuracy: ", accuracy_score(y_test, y_pred))
    st.write("Classification Report: ")
    st.text(classification_report(y_test, y_pred))

    # User input for prediction
    st.header("📝 Enter Details for Prediction")
    
    # Collect user input with updated layout
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    cholesterol = st.number_input("Cholesterol", min_value=0, max_value=500, value=200)
    blood_pressure = st.text_input("Blood Pressure (e.g., 120/80)")
    heart_rate = st.number_input("Heart Rate", min_value=0, max_value=200, value=70)
    diabetes = st.selectbox("Diabetes (0 = No, 1 = Yes)", [0, 1])
    family_history = st.selectbox("Family History (0 = No, 1 = Yes)", [0, 1])
    smoking = st.selectbox("Smoking (0 = No, 1 = Yes)", [0, 1])
    obesity = st.selectbox("Obesity (0 = No, 1 = Yes)", [0, 1])
    alcohol_consumption = st.selectbox("Alcohol Consumption (0 = No, 1 = Yes)", [0, 1])
    exercise_hours = st.number_input("Exercise Hours Per Week", min_value=0, value=0)
    diet = st.selectbox("Diet (0 = Unhealthy, 1 = Average, 2 = Healthy)", [0, 1, 2])
    previous_heart_problems = st.selectbox("Previous Heart Problems (0 = No, 1 = Yes)", [0, 1])
    medication_use = st.selectbox("Medication Use (0 = No, 1 = Yes)", [0, 1])
    stress_level = st.number_input("Stress Level", min_value=0, value=0)
    sedentary_hours = st.number_input("Sedentary Hours Per Day", min_value=0, value=0)
    income = st.number_input("Income", min_value=0, value=0)
    bmi = st.number_input("BMI", min_value=0.0, value=0.0)
    triglycerides = st.number_input("Triglycerides", min_value=0, value=0)
    physical_activity_days = st.number_input("Physical Activity Days Per Week", min_value=0, value=0)
    sleep_hours = st.number_input("Sleep Hours Per Day", min_value=0, value=0)
    
    # Prepare user input for prediction
    user_input = np.array([[age, cholesterol, heart_rate, diabetes, family_history, smoking, obesity, 
                            alcohol_consumption, exercise_hours, diet, previous_heart_problems, 
                            medication_use, stress_level, sedentary_hours, income, bmi, triglycerides, 
                            physical_activity_days, sleep_hours]])
    
    # Make prediction
    if st.button("Predict Heart Disease Risk"):
        prediction = model.predict(user_input)
        risk = "High" if prediction[0] == 1 else "Low"
        st.write(f"Predicted Risk of Heart Disease: {risk}")
