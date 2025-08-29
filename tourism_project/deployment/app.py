import os, json, joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Wellness Package Predictor", page_icon="🏝️", layout="centered")
st.title("🏝️ Wellness Tourism Package — Purchase Propensity")

MODEL_REPO = os.getenv("MODEL_REPO", "MBG0903/tourism_customer_xgb")
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Load model + metadata
model_file = hf_hub_download(repo_id=MODEL_REPO, filename="model.joblib", use_auth_token=HF_TOKEN)
meta_file  = hf_hub_download(repo_id=MODEL_REPO, filename="metadata.json", use_auth_token=HF_TOKEN)

model = joblib.load(model_file)
with open(meta_file) as f: meta = json.load(f)
st.caption("Model training metrics")
st.json(meta.get("metrics", {}))

# Inputs
st.sidebar.header("Customer Profile")
def i_num(label, value, minv=None, maxv=None, step=1):
    return st.sidebar.number_input(label, value=value, min_value=minv, max_value=maxv, step=step)

inputs = {}
inputs["Age"] = i_num("Age", 32, 18, 90)
inputs["TypeofContact"] = st.sidebar.selectbox("TypeofContact", ["Company Invited","Self Inquiry"])
inputs["CityTier"] = st.sidebar.selectbox("CityTier", ["Tier 1","Tier 2","Tier 3"])
inputs["Occupation"] = st.sidebar.selectbox("Occupation", ["Salaried","Freelancer","Self Employed","Student","Retired"])
inputs["Gender"] = st.sidebar.selectbox("Gender", ["Male","Female"])
inputs["NumberOfPersonVisiting"] = i_num("NumberOfPersonVisiting", 2, 1, 10)
inputs["PreferredPropertyStar"] = i_num("PreferredPropertyStar", 4, 1, 5)
inputs["MaritalStatus"] = st.sidebar.selectbox("MaritalStatus", ["Single","Married","Divorced"])
inputs["NumberOfTrips"] = i_num("NumberOfTrips", 3, 0, 50)
inputs["Passport"] = st.sidebar.selectbox("Passport", [0,1])
inputs["OwnCar"] = st.sidebar.selectbox("OwnCar", [0,1])
inputs["NumberOfChildrenVisiting"] = i_num("NumberOfChildrenVisiting", 0, 0, 10)
inputs["Designation"] = st.sidebar.selectbox("Designation", ["Executive","Manager","Senior Manager","AVP","VP","Director"])
inputs["MonthlyIncome"] = i_num("MonthlyIncome", 70000, 0, 1000000, 1000)
inputs["PitchSatisfactionScore"] = i_num("PitchSatisfactionScore", 4, 1, 5)
inputs["ProductPitched"] = st.sidebar.selectbox("ProductPitched", ["Basic","Deluxe","Super Deluxe","King","Queen"])
inputs["NumberOfFollowups"] = i_num("NumberOfFollowups", 2, 0, 20)
inputs["DurationOfPitch"] = i_num("DurationOfPitch", 15, 0, 120)

df_in = pd.DataFrame([inputs])

if st.button("Predict"):
    proba = model.predict_proba(df_in)[:,1][0]
    pred = int(proba >= 0.5)
    st.metric("Purchase Probability", f"{proba:.3f}")
    st.write("Prediction:", "Will Purchase (1)" if pred==1 else "Will Not Purchase (0)")
