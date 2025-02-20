import streamlit as st
from fastai.vision.all import *


def extract_flower(path):
    parts = str(path).split("/")
    return parts[5]

model = load_learner("flower_model.pkl")

def predict(path):
    img = PILImage.create(path)
    print("path type:", type(path))
    print("path value:", path)
    result = model.predict(path)
    print(result)

    flower = result[0]
    flower_index = result[1]
    accuracy_list = result[2]

    flower, flower_index, accuracy_list = model.predict(img)

    prediction_accuracy = accuracy_list[flower_index] * 100
    if prediction_accuracy < 90:
        image_title = f"I am not sure what flower this is. I am {prediction_accuracy:.3f}% sure this is a {flower} flower"
    else:
        image_title = f"{flower} - {prediction_accuracy:.3f}%"

    img.show(title=image_title)

st.title("Flower Prediction")
st.text("Built by Natalie Huynh")

upload_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])


if upload_file is not None:
    print("upload_file type:", type(upload_file))
    print("upload_file value:", upload_file)
    prediction = predict(upload_file)
    st.image(upload_file, caption=prediction, use_column_width=True)
    prediction = predict(upload_file)
    st.write(prediction)




