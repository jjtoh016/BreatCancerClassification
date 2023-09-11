import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import sklearn

st.title("""
         This page is to predict whether you have **BREAST CANCER** (not 100 accuracy as it is trained machine learning model)
         """)

st.subheader("The model is trained using **Support Vector Machine(SVM)** with accuract **97%**")

radius_mean = st.number_input(label='Tumour Radius Mean', key=0, step=0.1,format="%.2f")
perimeter_mean = st.number_input(label='Tumour Perimeter Mean', key=1, step=1.,format="%.2f")
area_mean = st.number_input(label='Tumour Radius Mean', step=0.1, key=2,format="%.2f")
compactness_mean = st.number_input(label='Tumour Radius Mean', step=0.1, key=3,format="%.2f")
concavity_mean = st.number_input(label='Tumour Radius Mean', step=0.1, key=4,format="%.2f")
concave_points_mean = st.number_input(label='Tumour Concave Points Mean', step=0.1, key=5, format="%.2f")
radius_se = st.number_input(label='Tumour Radius Se', step=0.1, key=6,format="%.2f")
perimeter_se = st.number_input(label='Tumour Perimeter Se', step=0.1, key=7,format="%.2f")
area_se = st.number_input(label='Tumour Area Se', step=0.1, key=8,format="%.2f")
radius_worst = st.number_input(label='Tumour Radius Worst', step=0.1, key=9,format="%.2f")
perimeter_worst = st.number_input(label='Tumour Perimeter Worst', step=0.1, key=10,format="%.2f")
area_worst = st.number_input(label='Tumour Area Worst', step=0.1, key=11,format="%.2f")
concavity_worst = st.number_input(label='Tumour Concavity Worst', step=0.1, key=12,format="%.2f")
concave_points_worst = st.number_input(label='Tumour Concave Points Worst', step=0.1, key=13,format="%.2f")

data = np.array([[radius_mean, perimeter_mean, area_mean, compactness_mean, concavity_mean, concave_points_mean, radius_se, perimeter_se,
            area_se, radius_worst, perimeter_worst, area_worst, concavity_worst, concave_points_worst]])

#load ML model

model_path = 'breast_cancer_SVM.pkl'

with open(model_path, 'rb') as file:
    model = pkl.load(file)
    
def prediction():
    pred = model.predict(data)
    
    st.subheader("Prediction Result")
    
    if pred == 0:
        st.write("Your Breast Tumour is **Not** Cancerous")
    else:
        st.write("Unfortunately, your Breast Tumour might be **Cancerous**")
        

if st.button('Submit'):
    prediction()



