# Import libraries
import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model


import matplotlib.pyplot as plt
import time

st.title('SHIP CLASSIFIER')  #Title
fig = plt.figure()
# ----------- Sidebar
#page = st.sidebar.selectbox('Page Navigation', ["Predictor", "Model analysis"])
st.sidebar.write("Created by Amrutha A and Ela Tejeshwar")


#st.sidebar.image("assets/logo.png", width=100)

st.markdown("""---""")

import zipfile
import tempfile

stream = st.file_uploader('TF.Keras model file (.h5py.zip)', type='zip')
if stream is not None:
  myzipfile = zipfile.ZipFile(stream)
  with tempfile.TemporaryDirectory() as tmp_dir:
    myzipfile.extractall(tmp_dir)
    root_folder = myzipfile.namelist()[0] # e.g. "model.h5py"
    model_dir = os.path.join(tmp_dir, root_folder)
    #st.info(f'trying to load model from tmp dir {model_dir}...')
    model = load_model(model_dir)


#Main function to upload image and process prediction

def main():
    st.info("Try guessing the class of the ship:")


    upload_columns = st.columns([1, 1])
    file_upload =  upload_columns[0].expander(label='Upload Image')
    file_uploaded = file_upload.file_uploader("Choose File", type=["png", "jpg", "jpeg"])

    if file_uploaded is not None:
        file_bytes = np.asarray(bytearray(file_uploaded.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 0)
        upload_columns[1].image(image, caption='Uploaded Image', use_column_width=True)
        #st.write(image.size)
    st.sidebar.markdown("""---""")
    option = st.selectbox('Select which ship you think is in the image:'
                          , ['Bulk Carrier', 'Container', 'Tanker']) # User selection

    class_btn = st.button("Classify") # Button to classify
    st.markdown("""---""")

    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                predictions = predict(image)
                if option == predictions:
                    st.write('Wow! You are right')
                    st.balloons()
                else:
                    st.write("You were wrong!")                 
                    st.write("Your prediction:",option)
                    st.write("Model's prediction:",predictions)

                time.sleep(1)


#Load classifier and predict the ship type after
#clicking on 'classify'button

def predict(image):
    #classifier_model = "vgg.h5"

    #model = load_model(classifier_model)
    img_r = cv2.resize(image, (128, 128))
    st.write(img_r.size)
    img_array = np.array(img_r)
    img_batch = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_batch)
    class_names = [
            'Bulk Carrier',
            'Container',
            'Tanker']

    result = f"Ship belongs to class: {class_names[np.argmax(prediction)]}."
    predicted_result = class_names[np.argmax(prediction)]
    st.write(result)
    return predicted_result

if __name__ == "__main__":
    main()
