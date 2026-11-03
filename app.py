import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
import pandas as pd
import pickle

#load the train model

model=tf.keras.models.load_model('model.keras')

#Load the encodeers and scaler

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender .pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## streamlit app

st.title('Customer Churn Prediction')

# User input

geography=st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance',value=60000.0)
credit_score=st.number_input('Credit_Score',min_value=350,max_value=850,value=650)
estimated_salary=st.number_input('Estimated_Salary',value=50000.0)
tenure=st.slider('Tenure',0,10,3)
num_of_products=st.slider('Number of Products',1,4,2)
has_cr_card=st.selectbox('Has Credit Card',[0,1],index=1)
is_active_memmber=st.selectbox('Is Active Member',[0,1],index=1)

#Prepare the inputdata
input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_memmber],
    'EstimatedSalary':[estimated_salary]
})


#onehot encoder 'Geography'

geo_encoder=onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoder_df=pd.DataFrame(geo_encoder,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))


#combain  on-hot encoder columns with input data

input_data=pd.concat([input_data.reset_index(drop=True),geo_encoder_df],axis=1)


#Scale the input data

input_data_scaled=scaler.transform(input_data)

#prediciton churn

prediction = model.predict(input_data_scaled, verbose=0)

churn_probability = float(prediction.ravel()[0])
churn_percentage = churn_probability * 100

st.subheader("Customer Churn Prediction")

st.metric(
    "Churn Probability",
    f"{churn_percentage:.2f}%"
)

if churn_probability >= 0.5:
    st.error("Current Prediction: Customer is Likely to Churn")
else:
    st.success("Current Prediction: Customer is Not Likely to Churn")