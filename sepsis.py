import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer

st.title("Sepsis Prediction")
st.header("Patient Data Input")

st.subheader("Vital Signs Input")
hour = st.number_input("Hour", value=0)
hr = st.number_input("HR", value=0.0)
o2_sat = st.number_input("O2Sat", value=0.0)
temp = st.number_input("Temp", value=0.0)
map = st.number_input("MAP", value=0.0)
resp = st.number_input("Resp", value=0.0)

st.subheader("Laboratory Values Input")
fio2 = st.number_input("FiO2", value=0.0)
sao2 = st.number_input("SaO2", value=0.0)
ast = st.number_input("AST", value=0.0)
bun = st.number_input("BUN", value=0.0)
chloride = st.number_input("Chloride", value=0.0)
creatinine = st.number_input("Creatinine", value=0.0)
glucose = st.number_input("Glucose", value=0.0)
lactate = st.number_input("Lactate", value=0.0)
bilirubin_total = st.number_input("Bilirubin Total", value=0.0)
troponin_i = st.number_input("Troponin I", value=0.0)
hct = st.number_input("Hct", value=0.0)
hgb = st.number_input("Hgb", value=0.0)
wbc = st.number_input("WBC", value=0.0)
platelets = st.number_input("Platelets", value=0.0)

st.subheader("Demographic Input")
age = st.number_input("Age", value=0.0)
gender = st.selectbox("Gender", ["Male", "Female"])
gender_mapping = {"Male": 1, "Female": 0}
gender_encoded = gender_mapping[gender]

if st.button("Submit"):
    user_data = {'Hour': hour,'HR': hr,'O2Sat': o2_sat,'Temp': temp,'MAP': map,
                 'Resp': resp,'FiO2': fio2,'SaO2': sao2,'AST': ast,'BUN': bun,
                 'Chloride': chloride,'Creatinine': creatinine,'Glucose': glucose,
                 'Lactate': lactate,'Bilirubin_total': bilirubin_total,
                 'TroponinI': troponin_i,'Hct': hct,'Hgb': hgb,'WBC': wbc,
                 'Platelets': platelets,'Age': age,'Gender': gender_encoded}

    user_df = pd.DataFrame(user_data, index=[0])

    model = pickle.load(open('model.pkl', 'rb'))

    prediction = model.predict(user_df)

    if (prediction[0] == 0):
        st.markdown("<p style='font-size:25px;'>Prediction: Non-septic</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='font-size:25px;'>Prediction: Septic</p>", unsafe_allow_html=True)

    training_dataset = pd.read_csv("training_dataset.csv", index_col=0)
    training_data = training_dataset.iloc[:,:22]
    training_labels = training_dataset.iloc[:,22]

    explainer = LimeTabularExplainer(training_data=training_data.to_numpy(),
                                     mode="classification",
                                     training_labels=training_labels.to_numpy(),
                                     feature_names=training_data.columns,
                                     discretize_continuous=True)
    
    explanation = explainer.explain_instance(user_df.iloc[0], model.predict_proba, num_features=22)

    st.write("Explanation for the Prediction:")

    feature_names, weights = zip(*explanation.as_list())
    y_pos = np.arange(len(feature_names))
    colors = ['green' if w > 0 else 'red' for w in weights]

    plt.figure(figsize=(10, 6))
    plt.barh(y_pos, weights, color=colors, align='center', alpha=0.5)
    plt.yticks(y_pos, feature_names)
    plt.xlabel('Weight')
    plt.title('Feature Importances')
    st.pyplot(plt)