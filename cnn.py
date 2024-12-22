import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import streamlit as st
from PIL import Image as PILImage

model = load_model('models/building_damage_classifier.h5')

class_labels = ['partially damaged', 'non-damaged', 'collapsed']

def classify_image(img):
    img = img.resize((224, 224))  
    img_array = np.array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0 

   
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]

    return predicted_class

def main():
    st.title("Building Damage Classification")
    st.write("Upload an image of the building to classify its damage.")

    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        img = PILImage.open(uploaded_file)
        
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button('Classify'):
            result = classify_image(img)
            st.write(f"The building is classified as: {result}")

if __name__ == "__main__":
    main()
