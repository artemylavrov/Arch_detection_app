import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background
from util import visualize

import streamlit as st
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from PIL import Image
import numpy as np

from util import visualize, set_background


st.set_page_config(
page_title="Детекция зданий",
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


file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = './model/model.pth'
cfg.MODEL.DEVICE = 'cpu'

predictor = DefaultPredictor(cfg)

# загружаем
if file:
    image = Image.open(file).convert('RGB')

    image_array = np.asarray(image)

    outputs = predictor(image_array)

    threshold = 0.5

    preds = outputs["instances"].pred_classes.tolist()
    scores = outputs["instances"].scores.tolist()
    bboxes = outputs["instances"].pred_boxes

    bboxes_ = []
    for j, bbox in enumerate(bboxes):
        bbox = bbox.tolist()

        score = scores[j]
        pred = preds[j]

        if score > threshold:
            x1, y1, x2, y2 = [int(i) for i in bbox]
            bboxes_.append([x1, y1, x2, y2])

    visualize(image, bboxes_)
