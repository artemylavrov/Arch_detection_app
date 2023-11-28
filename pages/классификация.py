import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background



st.set_page_config(
page_title="Классификация зданий",
page_icon="🔎",
layout="wide",
initial_sidebar_state="expanded",
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility:hidden;}
            footer {visibility:hidden;}
            .leftbar-text {
                font-size:14px;
            }
            header {visibility:hidden;}
            [data-testid="stSidebarNav"] {
                background-image: url(https://play-lh.googleusercontent.com/FG1HquqP8Ka88CrE_Uh5Q-h8s4RRyCjbNyeUyXG0GQakW9CpATKqF9UROLbaDW1ZO7DW);
                background-size: cover;
                padding-top: 150px;
                background-position: 5px 5px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "Навигация";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title('Детектор зданий по фото')
st.info('Данный модуль помогает узнать есть ли здания на фото', icon="ℹ️")


file2 = st.file_uploader('', type=['jpeg', 'jpg', 'png'])


model = load_model('./model/archi_classif.h5')


with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()


if file2 is not None:
    image = Image.open(file2).convert('RGB')
    st.image(image, use_column_width=True)

    class_name, conf_score = classify(image, model, class_names)

    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))