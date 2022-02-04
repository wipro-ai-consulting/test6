from transformers import pipeline

import streamlit as st

 
st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Wipro_Primary_Logo_Color_RGB.svg/240px-Wipro_Primary_Logo_Color_RGB.svg.png')
st.title('Medical Affairs Use case: Can AI understand special words used by doctors? ')
st.write('Can the AI spot elements like CITY  names in any sentance? Try using the 1st model in the dropdown list in the leftside') 
st.write('Can the AI spot CHEMICAL Names, Diseases Names in any sentance?  Try using the 3rd model ') 

st.header('To try out, Enter any text below and presss Control + Enter on keyboard') 
st.header('Also try out different AI brains. One of models can spot Diseases names! Select a brain on the drop down on left side of this screen.') 




unmasker2 = pipeline("fill-mask", model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
fillinsentance = "adjuvant Vaccines induce [MASK] response"

resultsfillmask = unmasker(fillinsentance, top_k=4)

for i in range(len(resultsfillmask)):
    st.header (resultsfillmask[i]['sequence'])
