from transformers import pipeline

import streamlit as st

from transformers import AutoTokenizer
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer

 
st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Wipro_Primary_Logo_Color_RGB.svg/240px-Wipro_Primary_Logo_Color_RGB.svg.png')
st.title('Medical Affairs Use case: Can AI understand special words used by doctors? ')
st.write('Can the AI spot elements like CITY  names in any sentance? Try using the 1st model in the dropdown list in the leftside') 
st.write('Can the AI spot CHEMICAL Names, Diseases Names in any sentance?  Try using the 3rd model ') 

st.header('To try out, Enter any text below and presss Control + Enter on keyboard') 
st.header('Also try out different AI brains. One of models can spot Diseases names! Select a brain on the drop down on left side of this screen.') 

 

question_answerer = pipeline("question-answering")

context = r""" Extractive Question Answering is the task of extracting an answer from a text given a question. """

result = question_answerer(question="What is extractive question answering?", context=context)
st.write(result['answer'])



checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence, return_tensors="tf")

st.write(model_inputs["input_ids"])




tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-pubmed")

# by default encoder-attention is `block_sparse` with num_random_blocks=3, block_size=64
model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-pubmed")

# decoder attention type can't be changed & will be "original_full"
# you can change `attention_type` (encoder only) to full attention like this:
model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-pubmed", attention_type="original_full")


text = "Amoxicillin/clavulanate (Augmentin) is a broad-spectrum antibacterial that has been available for clinical use in a wide range of indications for over 20 years and is now used primarily in the treatment of community-acquired respiratory tract infections. Amoxicillin/clavulanate was developed to provide a potent broad spectrum of antibacterial activity, coverage of beta-lactamase-producing pathogens and a favourable pharmacokinetic/pharmacodynamic (PK/PD) profile. These factors have contributed to the high bacteriological and clinical efficacy of amoxicillin/clavulanate in respiratory tract infection over more than 20 years. This is against a background of increasing prevalence of antimicrobial resistance, notably the continued spread of beta-lactamase-mediated resistance in Haemophilus influenzae and Moraxella catarrhalis, and penicillin, macrolide and quinolone resistance in Streptococcus pneumoniae. The low propensity of amoxicillin/clavulanate to select resistance mutations as well as a favourable PK/PD profile predictive of high bacteriological efficacy may account for the longevity of this combination in clinical use. However, in certain defined geographical areas, the emergence of S. pneumoniae strains with elevated penicillin MICs has been observed. In order to meet the need to treat drug-resistant S. pneumoniae, two new high-dose amoxicillin/clavulanate formulations have been developed. A pharmacokinetically enhanced tablet dosage form of amoxicillin/clavulanate 2000/125 mg twice daily (available as Augmentin XR in the USA), has been developed for use in adult respiratory tract infection due to drug-resistant pathogens, such as S. pneumoniae with reduced susceptibility to penicillin, as well as beta-lactamase-producing H. influenzae and M. catarrhalis. Amoxicillin/clavulanate 90/6.4 mg/kg/day in two divided doses (Augmentin ES-600) is for paediatric use in persistent or recurrent acute otitis media where there are risk factors for the involvement of beta-lactamase-producing strains or S. pneumoniae with reduced penicillin susceptibility. In addition to high efficacy, amoxicillin/clavulanate has a well known safety and tolerance profile of the two new high-dose formulations are not significantly different from those of conventional formulations. Amoxicillin/clavulanate is included in guidelines and recommendations for the treatment of bacterial sinusitis, acute otitis media, community-acquired pneumonia and acute exacerbations of chronic bronchitis. Amoxicillin/clavulanate continues to be an important agent in the treatment of community-acquired respiratory tract infections, both now and in the future."

inputs = tokenizer(text, return_tensors='tf')
prediction = model.generate(inputs)
prediction = tokenizer.batch_decode(prediction)

st.write(prediction)
