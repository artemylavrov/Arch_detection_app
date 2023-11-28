import base64
import tensorflow as tf
import keras
import streamlit as st
from PIL import ImageOps, Image
import numpy as np
import plotly.graph_objects as go
import streamlit as st

def set_background(image_file):
    """
    Эта функция устанавливает фон приложения Streamlit на изображение, указанное в данном файле изображения.

    Arg:
        image_file (str): путь к файлу изображения, которое будет использоваться в качестве фона.
    Return:
        None
    """

    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def classify(image, model, class_names):
    """
    Эта функция принимает изображение, модель и список имен классов и возвращает прогнозируемый класс и достоверность.
    оценка изображения.

    Параметры:
         image (PIL.Image.Image): изображение, которое необходимо классифицировать.
         model (tensorflow.keras.Model): обученная модель машинного обучения для классификации изображений.
         class_names (list): список имен классов, соответствующих классам, которые может предсказать модель.

    Возврат:
         Кортеж предсказанного имени класса и оценки достоверности этого предсказания.
    """
    # convert image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # convert image to numpy array
    image_array = np.asarray(image)

    # normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # set model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # make prediction
    prediction = model.predict(data)
    # index = np.argmax(prediction)
    index = 0 if prediction[0][0] > 0.95 else 1
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score


def GetBaseModel():
    layerSize = 320
    filterSize = 2
    model = keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal",
                                                           input_shape=(256,
                                                                        256,
                                                                        1)),
        tf.keras.layers.experimental.preprocessing.RandomRotation(.1),
        tf.keras.layers.experimental.preprocessing.RandomZoom(.1),
        tf.keras.layers.Conv2D(layerSize, (filterSize), activation=tf.nn.leaky_relu),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(layerSize, (filterSize), activation=tf.nn.leaky_relu),
        tf.keras.layers.Dropout(.1),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(layerSize, (filterSize), activation=tf.nn.leaky_relu),
        tf.keras.layers.Dropout(.1),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(layerSize, (filterSize), activation=tf.nn.leaky_relu),
        tf.keras.layers.Dropout(.1),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(layerSize, (filterSize), activation=tf.nn.leaky_relu),
        tf.keras.layers.Dropout(.1),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(layerSize, (filterSize), activation=tf.nn.leaky_relu),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(int(layerSize * 2), activation=tf.nn.leaky_relu),
        tf.keras.layers.Dropout(.3),
        tf.keras.layers.Dense(int(layerSize * 1.5), activation=tf.nn.leaky_relu),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Dense(int(layerSize / 2), activation=tf.nn.leaky_relu),

        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics='accuracy')
    # model.load_weights(savePath+'base_model.h5')
    model.load_weights('./model/75_model_6.h5')
    return model

def visualize(image, bboxes):
    """
         Визуализирует изображение с ограничивающими рамками с помощью Plotly.

         Аргументы:
             image: входное изображение.
             bboxes (список): список ограничивающих рамок в формате [x1, y1, x2, y2].

    """

    width, height = image.size

    shapes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox

        # Convert bounding box coordinates to the format expected by Plotly
        shapes.append(dict(
            type="rect",
            x0=x1,
            y0=height - y2,
            x1=x2,
            y1=height - y1,
            line=dict(color='red', width=6),
        ))

    fig = go.Figure()


    fig.update_layout(
        images=[dict(
            source=image,
            xref="x",
            yref="y",
            x=0,
            y=height,
            sizex=width,
            sizey=height,
            sizing="stretch"
        )]
    )


    fig.update_xaxes(range=[0, width], showticklabels=False)
    fig.update_yaxes(scaleanchor="x",
                     scaleratio=1,
                     range=[0, width], showticklabels=False)

    fig.update_layout(
        height=800,
        updatemenus=[
            dict(
                direction='left',
                pad=dict(r=10, t=10),
                showactive=True,
                x=0.11,
                xanchor="left",
                y=1.1,
                yanchor="top",
                type="buttons",
                buttons=[
                    dict(label="Original",
                         method="relayout",
                         args=["shapes", []]),
                    dict(label="Detections",
                         method="relayout",
                         args=["shapes", shapes])
                     ],
            )
        ]
    )

    st.plotly_chart(fig)
def classify(image, model, class_names):
    """
         Эта функция принимает изображение, модель и список имен классов и возвращает прогнозируемый класс и достоверность.
         оценка изображения.

         Параметры:
             image (PIL.Image.Image): изображение, которое необходимо классифицировать.
             model (tensorflow.keras.Model): обученная модель машинного обучения для классификации изображений.
             class_names (список): список имен классов, соответствующих классам, которые может предсказать модель.

         Возврат:
             Кортеж предсказанного имени класса и оценки достоверности этого предсказания.
         """
    # convert image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # convert image to numpy array
    image_array = np.asarray(image)

    # normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # set model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # make prediction
    prediction = model.predict(data)
    # index = np.argmax(prediction)
    index = 0 if prediction[0][0] > 0.95 else 1
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score