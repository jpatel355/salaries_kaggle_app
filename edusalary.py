import streamlit as st
import pandas as pd
import pickle

@st.cache(allow_output_mutation=True)
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

st.title("📊 Kaggle Survey: Job Role Predictor")
st.write("Select your country and industry to see the model’s predicted job role.")

# You can either hard-code these lists or (better) load them dynamically:
COUNTRIES = [
    "United States", "India", "United Kingdom", "Germany",
    "Canada", "Brazil", "France", "Other"
]
INDUSTRIES = [
    "Online Service/Internet-based Services", "Insurance/Risk Assessment",
    "Government/Public Service", "Manufacturing/Fabrication",
    "Computers/Technology", "Accounting/Finance", "Academics/Education",
    "Non-profit/Service", "Other"
]

country  = st.selectbox("Country of Residence", COUNTRIES)
industry = st.selectbox("Industry", INDUSTRIES)

input_df = pd.DataFrame([{ 'Q4': country, 'Q24': industry }])

st.subheader("Inputs")
st.write(input_df.rename(columns={'Q4':'Country','Q24':'Industry'}))

if st.button("Predict Job Role"):
    pred = model.predict(input_df)[0]
    st.success(f"🧑‍💼 Predicted Job Role: **{pred}**")
