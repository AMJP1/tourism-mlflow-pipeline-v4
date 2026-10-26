import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_pred_model_v1.joblib")
model = joblib.load(model_path)

st.title("Prediction App")
st.write("""
This application predicts the likelihood of customer converting.
Enter the sensor and configuration data below to get a prediction.
""")

Age    = st.number_input("Age", min_value=1, max_value=99)
TypeofContact    = st.selectbox("Contact type", ["Self Enquiry", "Company Invited"])
CityTier    = st.selectbox("Tier", ["1", "2", "3"])
DurationOfPitch    = st.number_input("Duration", min_value=1, max_value=200)
Occupation    = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender    = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting    = st.number_input("Number Of Person Visiting", min_value=1, max_value=10)
NumberOfFollowups    = st.number_input("Number Of Followups", min_value=1, max_value=10)
ProductPitched    = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
PreferredPropertyStar    = st.selectbox("Preferred Property Star", ["3", "4", "5"])
MaritalStatus    = st.selectbox("Marital Status", ["Single", "Divorced", "Married", "Unmarried"])
NumberOfTrips = st.number_input("Number Of Trips", min_value=0, step=1, value=1)
Passport    = st.selectbox("Passport", ["0", "1"])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=1, step=1)
OwnCar    = st.selectbox("Owned car", ["0", "1"])
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting", min_value=0, max_value=5, value=0, step=1)
Designation    = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=0.0, step=1.0)


input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": int(CityTier),
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": int(PreferredPropertyStar),
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": int(Passport),
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": int(OwnCar),
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome
}])

if st.button("Predict conversion"):
    prediction = model.predict(input_data)[0]
    result = "Conversion" if prediction == 1 else "Not converted"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
