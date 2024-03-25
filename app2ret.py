import streamlit as st
#import pandas as pd
from PIL import Image
import numpy as np
#import matplotlib.pyplot as plt
#import plotly.figure_factory as ff
#from sklearn.metrics import accuracy_score
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split
#import seaborn as sns
from streamlit_option_menu import option_menu
import keras
import tensorflow as tf

selected = option_menu(
    menu_title=None,
    options=["Diabetes Retinopathy", "About"],
    icons=["search","book"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)


if selected == "Diabetes Retinopathy":
  # Load your pre-trained model
  model = tf.keras.models.load_model("RetinoCNN.h5")

  # Function to preprocess the input image
  def preprocess_image(image):
      image = image.resize((224, 224))
      image = tf.keras.preprocessing.image.img_to_array(image)
      image = np.expand_dims(image, axis=0)
      image = tf.keras.applications.mobilenet.preprocess_input(image)
      return image

  # Function to make predictions
  def predict_diabetes_retinopathy(image):
      preprocessed_image = preprocess_image(image)
      prediction = model.predict(preprocessed_image)
      return prediction
  





# Streamlit App
  def main():
      st.title("Diabetes Retinopathy Prediction System")

      uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

      if uploaded_image is not None:
          image = Image.open(uploaded_image)
          st.image(image, caption="Uploaded Image", use_column_width=True)

          if st.button("Predict"):
              prediction = predict_diabetes_retinopathy(image)
              st.subheader("Prediction Result:")
              st.write(f"Probability of Diabetic Retinopathy: {prediction[0][0]:.2%}")

  if __name__ == "__main__":
      main()



if selected == "About":
  st.write("Diabetic retinopathy is a complication of diabetes that affects the eyes, specifically the retina. The retina is the layer of tissue at the back of the eye that detects light and sends visual information to the brain. Diabetes can cause damage to the blood vessels in the retina, leading to various vision problems and potentially even blindness if left untreated. Here's a detailed explanation of diabetic retinopathy, including its types, symptoms, preventive measures, and treatment options:")

  st.header("Types of Diabetic Retinopathy:")

  st.write("1. Non-proliferative diabetic retinopathy (NPDR): In this early stage, small blood vessels in the retina leak blood or fluid, causing swelling or the formation of deposits called exudates. This stage may not cause noticeable symptoms initially.")

  st.write("2. Proliferative diabetic retinopathy (PDR): In advanced stages, new blood vessels grow on the surface of the retina or into the vitreous gel, leading to bleeding, scar tissue formation, and retinal detachment. This stage is more severe and can result in vision loss or blindness.")

  st.header("Symptoms of Diabetic Retinopathy:")

  st.write("Blurred or distorted vision")
  st.write("Floaters (spots or dark strings floating in your vision)")
  st.write("Difficulty seeing at night")
  st.write("Sudden loss of vision")
  st.write("Impaired color vision")
  st.write("Eye pain or pressure")
  
  st.header("Treatment Options:")

  st.write("1. Laser Photocoagulation (Laser Therapy):")

  st.write("Laser photocoagulation is a common treatment for diabetic retinopathy, particularly for managing proliferative diabetic retinopathy (PDR) and macular edema.")
  st.write("During this procedure, a laser is used to create small burns or seal leaking blood vessels in the retina. This helps to reduce swelling and prevent further damage to the retina.")
  st.write("Laser treatment can also be used to reduce abnormal blood vessel growth in the retina, which is characteristic of proliferative diabetic retinopathy.")

  st.write("2. Intravitreal Injections:")

  st.write("Intravitreal injections involve injecting medications directly into the vitreous gel of the eye.")
  st.write("Anti-VEGF (vascular endothelial growth factor) medications, such as ranibizumab (Lucentis), aflibercept (Eylea), and bevacizumab (Avastin), are commonly used to treat diabetic macular edema and proliferative diabetic retinopathy.")
  st.write("These medications help reduce swelling, leakage, and abnormal blood vessel growth in the retina.")
  
  
  st.write("3. Vitrectomy:")

  st.write("Vitrectomy is a surgical procedure used to treat severe cases of diabetic retinopathy, particularly when there is significant bleeding into the vitreous gel or tractional retinal detachment.")
  st.write("During vitrectomy, the vitreous gel is removed from the eye and replaced with a clear fluid to maintain the shape of the eye.")
  st.write("This procedure allows the surgeon to remove blood, scar tissue, and other debris from the retina and repair any retinal detachments.")


  st.write("4. Anti-VEGF Therapy:")
  st.write("Anti-VEGF therapy involves the use of medications that block the action of vascular endothelial growth factor (VEGF), a protein that promotes abnormal blood vessel growth in the retina.")
  st.write("These medications are injected directly into the eye and help reduce swelling, leakage, and abnormal blood vessel growth, thereby improving vision and slowing the progression of diabetic retinopathy.")
  



