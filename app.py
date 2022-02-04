import spacy_streamlit
from pathlib import Path
import srsly
import importlib
import random
import streamlit as st

import spacy
import en_core_sci_sm
import en_ner_bc5cdr_md
import en_ner_craft_md

MODELS = srsly.read_json(Path(__file__).parent / "models.json")
DEFAULT_MODEL = "en_core_web_sm"
DEFAULT_TEXT = "Myeloid derived suppressor cells (MDSC) are immature  myeloid cells with immunosuppressive activity. They accumulate in tumor-bearing mice and humans with different types of cancer, including hepatocellular carcinoma (HCC). Spinal and bulbar muscular atrophy (SBMA) is an inherited motor neuron disease caused by the expansion of a polyglutamine tract within the androgen receptor (AR). SBMA can be caused by this easily. Amoxicillin/clavulanic acid, also known as co-amoxiclav or amox-clav, is an antibiotic medication used for the treatment of a number of bacterial infections. It is a combination consisting of amoxicillin, a β-lactam antibiotic, and potassium clavulanate, a β-lactamase inhibitor"
DESCRIPTION = """** Customization by Wipro based on OpenAI scispacy**"""

 
st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Wipro_Primary_Logo_Color_RGB.svg/240px-Wipro_Primary_Logo_Color_RGB.svg.png')
st.write('Nandri jES')
st.title('Medical Affairs Use case: Can AI understand special words used by doctors? ')
st.header('Can the AI spot elements like CITY  names in any sentance? Try using the 1st model in the dropdown list in the leftside') 
st.header('Can the AI spot GENE Names, Diseases Names in any sentance?  Try using the 3rd model ') 



def get_default_text(nlp):
    # Check if spaCy has built-in example texts for the language
    try:
        examples = importlib.import_module(f".lang.{nlp.lang}.examples", "spacy")
        return DEFAULT_TEXT
    except (ModuleNotFoundError, ImportError):
        return ""

spacy_streamlit.visualize(
    MODELS,
    default_model=DEFAULT_MODEL,
    visualizers=[ "ner", "parser", "similarity", "tokens"],
    show_visualizer_select=True,
    sidebar_description=DESCRIPTION,
    get_default_text=get_default_text
)
