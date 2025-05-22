import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st
from PIL import Image

df = pd.read_csv('./assets/data/heart_attack_prediction_dataset.csv')

df.drop(columns='Patient ID', inplace=True)

numerical_attributes = df.select_dtypes(include=[np.number]).columns
categorical_attributes = df.select_dtypes(include=['object']).columns

numerical_binaries = []
numerical_nonbinaries = []

for i in numerical_attributes:
    if df[i].nunique() == 2:
        numerical_binaries.append(i)
    else:
        numerical_nonbinaries.append(i)

df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True)
df['Systolic_BP'] = pd.to_numeric(df['Systolic_BP'])
df['Diastolic_BP'] = pd.to_numeric(df['Diastolic_BP'])
df = df.drop('Blood Pressure', axis=1)

X = df.drop(columns=['Heart Attack Risk'])
y = df['Heart Attack Risk']

categorical_attributes = X.select_dtypes(include=['object']).columns
for column in categorical_attributes:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

st.set_page_config(page_title="Heart Attack Prediction", page_icon=":heart:", layout="wide", initial_sidebar_state="expanded")

logo = Image.open('assets/images/logo.png')

st.sidebar.image(logo, use_column_width=True)
st.sidebar.title("Navigation")
nav_selection = st.sidebar.radio("Go to", ["Home", "Predict Heart Attack Risk", "Field Descriptions", "Normal Ranges"])

if nav_selection == "Home":
    st.title("Welcome to the Heart Attack Prediction App")
    st.write("This app uses machine learning to predict the risk of a heart attack based on various health and lifestyle factors.")
    st.write("To get started, select 'Predict Heart Attack Risk' from the navigation menu.")

elif nav_selection == "Predict Heart Attack Risk":
    st.title("Heart Attack Risk Prediction")
    st.subheader("Enter Patient Information")

    with st.form(key="patient_info_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=120, step=1, help="The patient's age in years.")
            cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0.0, max_value=500.0, step=1.0, help="The patient's total cholesterol level in milligrams per deciliter.")
            systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=200, step=1, help="The patient's systolic blood pressure in millimeters of mercury.")
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, step=1, help="The patient's resting heart rate in beats per minute.")
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1, help="Body Mass Index")
        
        with col2:
            diabetes = st.selectbox("Diabetes", ["No", "Yes"], help="Whether the patient has been diagnosed with diabetes.")
            smoking = st.selectbox("Smoking", ["No", "Yes"], help="Whether the patient currently smokes or has a history of smoking.")
            family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"], help="Whether the patient has a family history of heart disease.")
            previous_heart_problems = st.selectbox("Previous Heart Problems", ["No", "Yes"], help="Whether the patient has a history of previous heart problems.")
            physical_activity = st.number_input("Physical Activity Days per Week", min_value=0, max_value=7, step=1)

        predict_button = st.form_submit_button("Predict Heart Attack Risk")

    # After model training, calculate feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': abs(model.coef_[0])
    }).sort_values('importance', ascending=False)
    
    if predict_button:
        # Set default values based on healthy ranges from dataset statistics
        default_values = {
            'sex': 0,  # Female as default
            'diastolic_bp': min(max(70, systolic_bp - 40), 90),  # Normal range
            'obesity': 1 if bmi > 30 else 0,
            'alcohol': 0,
            'exercise': min(physical_activity * 2.0, 10.0),  # More realistic exercise hours
            'diet': 1,  # Healthy diet
            'medication_use': 1 if previous_heart_problems == "Yes" else 0,
            'stress': 0,  # Low stress
            'sedentary_hours': max(24 - (physical_activity * 3), 4),  # More realistic sedentary hours
            'income': 50000,  # Average income
            'triglycerides': min(max(100, cholesterol * 0.6), 200),  # More realistic triglycerides
            'sleep_hours': 8,  # Optimal sleep
            'country': 0,
            'continent': 0,
            'hemisphere': 0
        }
    
    # Create user input with more realistic correlations
    user_input = np.array([age, default_values['sex'], cholesterol, systolic_bp,
                          default_values['diastolic_bp'], heart_rate,
                          1 if diabetes == "Yes" else 0,
                          1 if family_history == "Yes" else 0,
                          1 if smoking == "Yes" else 0,
                          default_values['obesity'], default_values['alcohol'],
                          default_values['exercise'], default_values['diet'],
                          1 if previous_heart_problems == "Yes" else 0,
                          default_values['medication_use'], default_values['stress'],
                          default_values['sedentary_hours'], default_values['income'],
                          bmi, default_values['triglycerides'], physical_activity,
                          default_values['sleep_hours'], default_values['country'],
                          default_values['continent'], default_values['hemisphere']
                          ]).reshape(1, -1)
    
    # Scale the user input
    user_input_scaled = scaler.transform(user_input)
    prediction_proba = model.predict_proba(user_input_scaled)[0]
    
    # Calculate weighted risk score based on medical guidelines
    risk_factors = {
    'age': (age > 60, 2),
    'cholesterol': (cholesterol > 240, 2),
    'blood_pressure': (systolic_bp > 140, 2),
    'heart_rate': (heart_rate > 100 or heart_rate < 60, 1),
    'diabetes': (diabetes == "Yes", 2),
    'family_history': (family_history == "Yes", 1),
    'smoking': (smoking == "Yes", 2),
    'bmi': (bmi > 30, 1),
    'previous_problems': (previous_heart_problems == "Yes", 3),
    'physical_activity': (physical_activity < 3, 1)
    }
    
    # Calculate weighted risk score
    risk_score = sum(weight for condition, weight in risk_factors.values() if condition)
    max_risk_score = sum(weight for _, weight in risk_factors.values())
    risk_factor = risk_score / max_risk_score
    
    # Adjust prediction probability based on risk factors
    adjusted_prob = (prediction_proba[1] * 0.7) + (risk_factor * 0.3)
    risk_percentage = min(max(adjusted_prob * 100, 5), 95)
    
    # Determine risk level with more granular thresholds
    if risk_percentage < 20:
        risk_level = "very low"
        color = "green"
    elif risk_percentage < 40:
        risk_level = "low"
        color = "lightgreen"
    elif risk_percentage < 60:
        risk_level = "moderate"
        color = "orange"
    elif risk_percentage < 80:
        risk_level = "high"
        color = "orangered"
    else:
        risk_level = "very high"
        color = "red"

    # Display prediction result with more context
    st.session_state.prediction_result = f"The patient is at {risk_level} risk of a heart attack (Risk: {risk_percentage:.1f}%)"
    st.session_state.prediction_color = color

    # Add risk factor explanation
    risk_explanation = "\n\nKey risk factors contributing to this prediction:\n"
    high_risk_factors = [factor for factor, (condition, _) in risk_factors.items() if condition]
    if high_risk_factors:
        risk_explanation += ", ".join(high_risk_factors)
    else:
        risk_explanation += "No major risk factors identified"

    st.markdown(
        f"""
        <div class="fixed top-0 left-0 w-screen h-screen bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div class="bg-white rounded-lg shadow-lg p-8" style="max-width: 500px;">
                <h2 class="text-2xl font-bold mb-4" style="color: {st.session_state.prediction_color};">{st.session_state.prediction_result}</h2>
                <p class="mb-4">Based on the provided information, the model has predicted the patient's heart attack risk.{risk_explanation}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

elif nav_selection == "Field Descriptions":
    st.title("Field Descriptions")
    st.write("This section provides information about the purpose of each input field in the 'Predict Heart Attack Risk' section.")

    st.subheader("Age")
    st.write("The patient's age in years.")

    st.subheader("Sex")
    st.write("The patient's biological sex.")

    st.subheader("Cholesterol (mg/dL)")
    st.write("The patient's total cholesterol level in milligrams per deciliter.")

    st.subheader("Systolic Blood Pressure (mmHg)")
    st.write("The patient's systolic blood pressure in millimeters of mercury.")

    st.subheader("Diastolic Blood Pressure (mmHg)")
    st.write("The patient's diastolic blood pressure in millimeters of mercury.")

    st.subheader("Heart Rate (bpm)")
    st.write("The patient's resting heart rate in beats per minute.")

    st.subheader("Diabetes")
    st.write("Whether the patient has been diagnosed with diabetes.")

    st.subheader("Family History of Heart Disease")
    st.write("Whether the patient has a family history of heart disease.")

    st.subheader("Smoking")
    st.write("Whether the patient currently smokes or has a history of smoking.")

    st.subheader("Obesity")
    st.write("Whether the patient is considered obese based on their body mass index (BMI).")

    st.subheader("Alcohol Consumption")
    st.write("Whether the patient regularly consumes alcoholic beverages.")

    st.subheader("Exercise Hours per Week")
    st.write("The number of hours per week the patient engages in physical exercise.")

    st.subheader("Healthy Diet")
    st.write("Whether the patient maintains a healthy, balanced diet.")

    st.subheader("Previous Heart Problems")
    st.write("Whether the patient has a history of previous heart problems or cardiovascular issues.")

    st.subheader("Medication Use")
    st.write("Whether the patient is currently taking any medications, including prescription drugs.")

    st.subheader("Stress Level")
    st.write("The patient's perceived level of stress.")

    st.subheader("Sedentary Hours per Day")
    st.write("The number of hours per day the patient spends in sedentary activities.")

    st.subheader("Income (USD)")
    st.write("The patient's annual household income in US dollars.")

    st.subheader("BMI")
    st.write("The patient's body mass index, calculated as weight in kilograms divided by height in meters squared.")

    st.subheader("Triglycerides (mg/dL)")
    st.write("The patient's triglyceride level in milligrams per deciliter.")

    st.subheader("Physical Activity Days per Week")
    st.write("The number of days per week the patient engages in physical activity.")

    st.subheader("Sleep Hours per Day")
    st.write("The number of hours per day the patient sleeps.")

    st.subheader("Country")
    st.write("The country where the patient resides.")

    st.subheader("Continent")
    st.write("The continent where the patient's country is located.")

    st.subheader("Hemisphere")
    st.write("The hemisphere (Northern or Southern) where the patient's country is located.")

elif nav_selection == "Normal Ranges":
    st.title("Normal Ranges for Health Parameters")
    st.write("This section provides information about the normal/healthy ranges for each input parameter used in heart attack risk prediction.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Vital Signs")
        st.markdown("**Age**\n- Adult range: 18-120 years\n- Higher risk: > 60 years")
        
        st.markdown("**Blood Pressure**\n- Normal: < 120/80 mmHg\n- Elevated: 120-129/< 80 mmHg\n- High: ≥ 130/80 mmHg\n- Crisis: > 180/120 mmHg")
        
        st.markdown("**Heart Rate**\n- Normal resting: 60-100 bpm\n- Athletic: 40-60 bpm\n- Concerning: < 40 or > 100 bpm")
        
        st.markdown("**BMI**\n- Underweight: < 18.5\n- Normal weight: 18.5-24.9\n- Overweight: 25-29.9\n- Obese: ≥ 30")

        st.markdown("**Cholesterol**\n- Total Cholesterol:\n  - Normal: < 200 mg/dL\n  - Borderline high: 200-239 mg/dL\n  - High: ≥ 240 mg/dL")

    with col2:
        st.subheader("Lifestyle Factors")
        st.markdown("**Physical Activity**\n- Minimum recommended: 3-5 days/week\n- Optimal: 5-7 days/week\n- Sedentary: < 3 days/week")
        
        st.markdown("**Sleep**\n- Recommended: 7-9 hours/day\n- Minimum: 6 hours/day\n- Maximum: 10 hours/day")
        
        st.markdown("**Exercise Hours**\n- Minimum: 2.5 hours/week\n- Optimal: 5+ hours/week\n- Maximum: 15 hours/week")
        
        st.markdown("**Sedentary Hours**\n- Recommended: < 6 hours/day\n- Concerning: > 8 hours/day\n- High risk: > 10 hours/day")

    st.subheader("Risk Factors")
    st.markdown("The following factors increase heart attack risk when present:")
    risk_factors = """
    - Diabetes
    - Family History of Heart Disease
    - Smoking
    - Previous Heart Problems
    - High Stress Levels
    - Excessive Alcohol Consumption
    - Poor Diet
    - Obesity (BMI ≥ 30)
    """
    st.markdown(risk_factors)

    st.subheader("Protective Factors")
    st.markdown("The following factors may help reduce heart attack risk:")
    protective_factors = """
    - Regular Physical Activity
    - Healthy Diet
    - Adequate Sleep
    - Stress Management
    - Regular Medical Check-ups
    - Maintaining Healthy Weight
    - Blood Pressure Control
    - Cholesterol Management
    """
    st.markdown(protective_factors)

if __name__ == "__main__":
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy on test set: {accuracy:.2%}")