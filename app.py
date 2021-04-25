import streamlit as st
import pandas as pd

import pickle
# import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


st.write("""
# Big Markt price Prediction App
This app predicts the **Big Maket Prices**!
""")

st.write('---')



# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')



def user_input_features():
    Pregnancies=st.sidebar.slider('Pregnancies',0,31)
    Glucose=st.sidebar.slider('Glucose',0,199)
    SkinThickness = st.sidebar.slider('SkinThickness',0,99)
    Insulin = st.sidebar.slider('Insulin',0,846)
    BloodPressure = st.sidebar.slider('BloodPressure',0,122)
    BMI=st.sidebar.slider('BMI',0,122)
    DiabetesPedigreeFunction=st.sidebar.slider('DiabetesPedigreeFunction',0.078000,2.420000)
    Age=st.sidebar.slider('Age',21,81)
       
    data = {'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure,
            'SkinThickness': SkinThickness,
            'Insulin':Insulin,
            'BMI': BMI,
            'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
            'Age':Age,
            }   
    features = pd.DataFrame(data, index=[0])

    return features

df = user_input_features()



scaler = StandardScaler()
scaledd_values=scaler.fit_transform(df)
scaledd_df=pd.DataFrame(scaledd_values,columns=df.columns)




load_clf=pickle.load(open('healthCare.pkl', 'rb'))



prediction_praob = load_clf.predict_proba(df)
prediction=load_clf.predict(df)

            



st.header('Specified Input parameters')
st.write(df)
st.write('---')






st.subheader('Prediction')

st.write(prediction)


st.subheader('Prediction Proability')

st.write(prediction_praob)
