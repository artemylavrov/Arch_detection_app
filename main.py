import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background, GetBaseModel


set_background('bgs/bd6.jpg')

st.title("Сервис классификации архитекутрных стилей")
st.info("Нажмите на меню левой боковой панели, чтобы перейти к другим страницам")
st.markdown(
    """
    <style>
    textarea {
        font-size: 5rem !important;
    }
    input {
        font-size: 5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
hide_st_style = """
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
st.markdown(hide_st_style, unsafe_allow_html=True)

st.subheader("Контактная информация")
st.info(
    """
    Автор - [Лавров Артём!](https://github.com/artemylavrov) \n\n
    Контактный номер: 89588051441\n
    """
)




