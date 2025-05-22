import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV # Added GridSearchCV
import streamlit as st
from PIL import Image

# Load and preprocess data
df = pd.read_csv('./assets/data/heart_attack_prediction_dataset.csv')
df.drop(columns='Patient ID', inplace=True)

# Split blood pressure into systolic and diastolic
df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True)
df['Systolic_BP'] = pd.to_numeric(df['Systolic_BP'])
df['Diastolic_BP'] = pd.to_numeric(df['Diastolic_BP'])
df = df.drop('Blood Pressure', axis=1)

# Prepare features and target
X = df.drop(columns=['Heart Attack Risk'])
y = df['Heart Attack Risk']

# Encode categorical variables
categorical_attributes = X.select_dtypes(include=['object']).columns
for column in categorical_attributes:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
# model = LogisticRegression(max_iter=200)
# model.fit(X_train, y_train)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200, 300] # Added max_iter to the grid
}
grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model
model = grid_search.best_estimator_

# Calculate and display model accuracy
accuracy = model.score(X_test, y_test)

# Streamlit UI
st.set_page_config(page_title="Heart Attack Risk Prediction", page_icon=":heart:", layout="wide")

# Sidebar with normal ranges
st.sidebar.title("Normal Ranges Reference")
st.sidebar.markdown(f"**Model Accuracy: {accuracy:.2%}**")

st.sidebar.subheader("Vital Signs")
st.sidebar.markdown("""
- **Blood Pressure**:
  - Normal: < 120/80 mmHg
  - Elevated: 120-129/< 80 mmHg
  - High: ≥ 130/80 mmHg

- **Heart Rate**:
  - Normal: 60-100 bpm
  - Athletic: 40-60 bpm

- **Cholesterol**:
  - Normal: < 200 mg/dL
  - Borderline: 200-239 mg/dL
  - High: ≥ 240 mg/dL

- **BMI**:
  - Normal: 18.5-24.9
  - Overweight: 25-29.9
  - Obese: ≥ 30
""")

st.sidebar.subheader("Lifestyle Factors")
st.sidebar.markdown("""
- **Physical Activity**: 150+ mins/week
- **Sleep**: 7-9 hours/day
- **Exercise**: 3-5 days/week
""")

# Main content
st.title("Heart Attack Risk Prediction")
st.write("Enter patient information to predict heart attack risk.")

with st.form(key="patient_info"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=40,
                            help="Normal range: 18-100 years")
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=500, value=180,
                                    help="Normal: <200 mg/dL")
        systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=200, value=120,
                                    help="Normal: <120 mmHg")
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=75,
                                    help="Normal: 60-100 bpm")
        bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=22.0,
                            help="Normal: 18.5-24.9")
    
    with col2:
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
        previous_heart_problems = st.selectbox("Previous Heart Problems", ["No", "Yes"])
        physical_activity = st.number_input("Physical Activity Days per Week", min_value=0, max_value=7, value=3,
                                        help="Recommended: 3-5 days/week")
    
    with col3:
        sex = st.selectbox("Sex", ["Female", "Male"])
        alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
        diet = st.selectbox("Healthy Diet", ["Yes", "No"])
        stress = st.selectbox("High Stress Level", ["No", "Yes"])
        sleep_hours = st.number_input("Sleep Hours per Day", min_value=4, max_value=12, value=8,
                                    help="Normal: 7-9 hours/day")

    predict_button = st.form_submit_button("Predict Risk")

if predict_button:
    # Process input data
    default_values = {
        'sex': 1 if sex == "Male" else 0,
        'diastolic_bp': min(max(70, systolic_bp - 40), 90),
        'obesity': 1 if bmi > 30 else 0,
        'alcohol': 1 if alcohol == "Yes" else 0,
        'exercise': min(physical_activity * 2.0, 10.0),
        'diet': 0 if diet == "No" else 1,
        'medication_use': 1 if previous_heart_problems == "Yes" else 0,
        'stress': 1 if stress == "Yes" else 0,
        'sedentary_hours': max(24 - (physical_activity * 3), 4),
        'income': 50000,
        'triglycerides': min(max(100, cholesterol * 0.6), 200),
        'country': 0,
        'continent': 0,
        'hemisphere': 0
    }

    # Create input array
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
                          sleep_hours, default_values['country'],  # Use sleep_hours directly
                          default_values['continent'], default_values['hemisphere']
                          ]).reshape(1, -1)

    # Scale input and predict
    user_input_scaled = scaler.transform(user_input)
    risk_prob = model.predict_proba(user_input_scaled)[0][1]
    
    # Calculate risk score
    # Adjust risk factors and weights
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
        'physical_activity': (physical_activity < 3, 1),
        'diet': (diet == "No", 1),
        'stress': (stress == "Yes", 1),
        'sleep': (sleep_hours < 6 or sleep_hours > 9, 1),
        'alcohol': (alcohol == "Yes", 1)
    }
    
    # Calculate weighted risk score with protective factors
    risk_score = sum(weight for condition, weight in risk_factors.values() if condition)
    max_risk_score = sum(weight for _, weight in risk_factors.values())
    
    # Add protective factors
    protective_factors = {
        'healthy_diet': (diet == "Yes", 0.5),
        'good_sleep': (6 <= sleep_hours <= 9, 0.5),
        'active_lifestyle': (physical_activity >= 5, 0.5),
        'normal_bmi': (18.5 <= bmi <= 24.9, 0.5),
        'normal_bp': (systolic_bp < 120, 0.5),
        'normal_cholesterol': (cholesterol < 200, 0.5)
    }
    
    protection_score = sum(weight for condition, weight in protective_factors.values() if condition)
    max_protection_score = sum(weight for _, weight in protective_factors.values())
    
    # Calculate risk factor with protection consideration
    risk_factor = (risk_score / max_risk_score) * 0.7 - (protection_score / max_protection_score) * 0.3
    risk_factor = max(0, min(risk_factor, 1))
    
    # Calculate final risk percentage with more weight on protective factors for normal values
    final_risk = (risk_prob * 0.5 + risk_factor * 0.5) * 100
    final_risk = min(max(final_risk, 0), 95)  # Allow 0% risk for very healthy values
    
    # Determine risk level with adjusted thresholds
    if final_risk < 10:
        risk_level = "No Risk"
        color = "#00ff00"  # Bright green for no risk
    elif final_risk < 25:
        risk_level = "Very Low"
        color = "#90EE90"  # Light green
    elif final_risk < 50:  # Changed threshold to 50 for Low risk
        risk_level = "Low"
        color = "#90EE90"  # Light green for Low risk (below 50%)
    elif final_risk < 75:
        risk_level = "Moderate"
        color = "#FFA500"  # Orange for Moderate risk
    else:
        risk_level = "High"
        color = "#FF0000"  # Red for High risk

    # Display results with enhanced visualization
    st.markdown(f"### Risk Assessment")
    if risk_level == "No Risk":
        st.markdown(f"<h3 style='color: {color}'>Risk Level: {risk_level} (Normal Range)</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color: {color}'>Risk Level: {risk_level} ({final_risk:.1f}%)</h3>", unsafe_allow_html=True)
    
    # Show contributing factors and protective factors
    col1, col2 = st.columns(2)
    
    with col1:
        high_risk_factors = [factor.replace('_', ' ').title() for factor, (condition, _) in risk_factors.items() if condition]
        if high_risk_factors:
            st.markdown("**Risk Factors:**")
            for factor in high_risk_factors:
                st.markdown(f"- {factor}")
        else:
            st.markdown("**No major risk factors identified**")
    
    with col2:
        active_protective_factors = [factor.replace('_', ' ').title() for factor, (condition, _) in protective_factors.items() if condition]
        if active_protective_factors:
            st.markdown("**Protective Factors:**")
            for factor in active_protective_factors:
                st.markdown(f"- {factor}")