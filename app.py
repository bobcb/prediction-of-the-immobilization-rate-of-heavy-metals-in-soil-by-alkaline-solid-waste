#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from joblib import load
import numpy as np

# Load model and scaler
model = load('ML_model.joblib')
scaler = load('StandardScaler.joblib')

st.markdown("""
    <style>
        /* Adjust spacing and align title with columns */
        .block-container {
            padding-left: 7rem;
            padding-right: 7rem;
        }

        /* Set width for the title */
        .main-title {
            font-size: 24px;
            font-weight: bold;
            width: 1020px; /* Set title width */
            word-break: break-word;
            hyphens: auto;  /* Adds hyphenation for long words */
        }

        /* Style for number_input text in red */
        .custom-red-label {
            color: red;
            font-weight: bold;
        }

        /* Add fixed space for the first and third columns */
        .spacing-col {
            margin-top: 20px; /* Set vertical distance for the first and third columns */
        }

        /* Style for the button */
        .custom-button {
            width: 400px; /* 设置按钮宽度 */
            height: 3em; /* 设置按钮高度，使其与输入框一致 */
        }
    </style>
""", unsafe_allow_html=True)


st.markdown('<div class="title-container"><h1 class="main-title">Prediction of the immobilization rate of<br>heavy metals in soil by alkaline solid waste</h1></div>', unsafe_allow_html=True)

with st.container():

    col1, spacer1, col2, spacer2, col3 = st.columns([1, 0.2, 1, 0.2, 1])

    with col1:
        # Add spacing to the first column
        st.markdown('<div class="spacing-col"></div>', unsafe_allow_html=True)
        feature1 = st.number_input(u'$\mathrm{SiO_2\;(\%)}$', step=0.01, format='%.2f')
        feature2 = st.number_input(u'$\mathrm{CaO\;(\%)}$', step=0.01, format='%.2f')
        feature3 = st.number_input(u'$\mathrm{Al_2O_3\;(\%)}$', step=0.01, format='%.2f')
        feature4 = st.number_input(u'$\mathrm{Soil\;pH}$', step=0.01, format='%.2f')

    with col2:
        feature5 = st.number_input(u'$\mathrm{Soil\;heavy\;metal}$\n$\mathrm{concentration\;(mg/kg)}$', step=0.01, format='%.2f')
        feature6 = st.number_input(u'$\mathrm{Temperature\;(℃)}$', step=0.01, format='%.2f')
        feature7 = st.number_input(u'$\mathrm{Curing\;time\;(d)}$', step=0.01, format='%.2f')
        feature8 = st.number_input(u'$\mathrm{Liquid/Solid}$', step=0.01, format='%.2f')

    with col3:
        st.markdown('<div class="spacing-col"></div>', unsafe_allow_html=True)
        feature9 = st.number_input(u'$\mathrm{Extraction\;agent\;pH}$', step=0.01, format='%.2f')
        feature10 = st.number_input(u'$\mathrm{Electronegativity}$', step=0.01, format='%.2f')
        feature11 = st.number_input(u'$\mathrm{Hydrated\;ion\;radius\;(Å)}$', step=0.01, format='%.2f')
        feature = st.number_input(u'$\mathrm{Experimental\;value\;(\%)}$', step=0.01, format='%.2f')

 

        feature_values = [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11]

if st.button('Predict'):
    input_data = np.array([feature_values])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    residual = abs(float(prediction) - feature)
    st.success(f'Predicted Heavy Metal Immobilization Rate: {prediction[0]:.2f}%')
    if feature != 0:
        st.success(f'Residual: {residual:.2f}%')
    


# In[ ]:




