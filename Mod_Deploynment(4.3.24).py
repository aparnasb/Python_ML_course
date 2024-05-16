#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
import streamlit as st


# In[2]:


model=pickle.load(open('log_model.pkl','rb'))


# In[3]:


st.title('Model Deployment using Logistic Regression')
st.sidebar.subheader('User Input Parameters')


# In[4]:


def user_input_parameters():
    CLMSEX= st.sidebar.selectbox('Gender-Female=0,Male=1',[0,1])
    CLMINSUR=st.sidebar.selectbox('Insurance',[0,1])
    SEATBELT= st.sidebar.selectbox('Seatbelt',[0,1])
    CLMAGE= st.sidebar.number_input('Age')
    LOSS= st.sidebar.number_input('LOSS')
    data= {'CLMSEX':CLMSEX,'CLMINSUR':CLMINSUR,'SEATBELT':SEATBELT,'CLMAGE':CLMAGE,'LOSS':LOSS}
    features= pd.DataFrame(data,index=[0])
    return features
df= user_input_parameters()
st.subheader('User_input_Parameters')
st.write(df)

predict= model.predict(df)
predict_proba= model.predict_proba(df)

st.subheader('Predicted Value')
st.write('Yes'if predict_proba[0][1]>0.5 else 'No')

st.subheader('Predict_Proba')
st.write(predict_proba)


# In[ ]:




