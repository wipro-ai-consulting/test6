from transformers import pipeline

import streamlit as st

 
st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Wipro_Primary_Logo_Color_RGB.svg/240px-Wipro_Primary_Logo_Color_RGB.svg.png')
st.title('Medical Affairs Use case: Can AI understand special words used by doctors? ')
st.write('Can the AI spot elements like CITY  names in any sentance? Try using the 1st model in the dropdown list in the leftside') 
st.write('Can the AI spot CHEMICAL Names, Diseases Names in any sentance?  Try using the 3rd model ') 

st.header('To try out, Enter any text below and presss Control + Enter on keyboard') 
st.header('Also try out different AI brains. One of models can spot Diseases names! Select a brain on the drop down on left side of this screen.') 


classifier = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")

result = classifier("I hate you")[0]

st.write(result['label'])


question_answerer = pipeline("question-answering")

context = r""" Extractive Question Answering is the task of extracting an answer from a text given a question. """

result = question_answerer(question="What is extractive question answering?", context=context)
st.write(result['answer'])

 
