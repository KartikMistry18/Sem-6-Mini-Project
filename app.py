
# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from streamlit_option_menu import option_menu

import tensorflow as tf



df = pd.read_csv("diabetesnew.csv")

selected = option_menu(
    menu_title=None,
    options=["Diabetes Prediction","Diabetes Retinopathy", "About"],
    icons=["search","search","book"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)
if selected == "Diabetes Prediction":
# HEADINGS
  st.title('Diabetes Checkup')
  st.sidebar.header('Patient Data')
  st.subheader('Training Data Stats')
  st.write(df.describe())


  # X AND Y DATA
  x = df.drop(['Outcome'], axis = 1)
  y = df.iloc[:, -1]
  x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


  # FUNCTION
  def user_report():
      pregnancies = st.sidebar.number_input('Pregnancies', 0,20)
      glucose = st.sidebar.slider('Glucose', 0,250, 120 )
      bp = st.sidebar.slider('Blood Pressure', 0,140, 70 )
      skinthickness = st.sidebar.slider('Skin Thickness', 0,100, 20 )
      insulin = st.sidebar.slider('Insulin', 0,850, 79 )
      bmi = st.sidebar.slider('BMI', 0,70, 20 )
      dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0,3.0, 0.47 )
      age = st.sidebar.slider('Age', 18 ,100, 33 )

      user_report_data = {
          'Pregnancies': pregnancies,
          'Glucose': glucose,
          'BloodPressure': bp,
          'SkinThickness': skinthickness,
          'Insulin': insulin,
          'BMI': bmi,
          'DiabetesPedigreeFunction': dpf,
          'Age': age
      }
      report_data = pd.DataFrame(user_report_data, index=[0])
      return report_data




  # PATIENT DATA
  user_data = user_report()
  st.subheader('Patient Data')
  st.write(user_data)




  # MODEL
  rf  = RandomForestClassifier()
  rf.fit(x_train, y_train)
  user_result = rf.predict(user_data)
  #user_result = loaded_model.predict(user_data)

  

  # VISUALISATIONS
  st.title('Visualised Patient Report')



  # COLOR FUNCTION
  if user_result[0]==0:
    color = 'blue'
  else:
    color = 'red'


  # Age vs Pregnancies
  st.header('Pregnancy count Graph (Others vs Yours)')
  fig_preg = plt.figure()
  ax1 = sns.scatterplot(x = 'Age', y = 'Pregnancies', data = df, hue = 'Outcome', palette = 'Greens')
  ax2 = sns.scatterplot(x = user_data['Age'], y = user_data['Pregnancies'], s = 150, color = color)
  plt.xticks(np.arange(10,100,5))
  plt.yticks(np.arange(0,20,2))
  plt.title('0 - Healthy & 1 - Unhealthy')
  st.pyplot(fig_preg)



  # Age vs Glucose
  st.header('Glucose Value Graph (Others vs Yours)')
  fig_glucose = plt.figure()
  ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = df, hue = 'Outcome' , palette='magma')
  ax4 = sns.scatterplot(x = user_data['Age'], y = user_data['Glucose'], s = 150, color = color)
  plt.xticks(np.arange(10,100,5))
  plt.yticks(np.arange(0,220,10))
  plt.title('0 - Healthy & 1 - Unhealthy')
  st.pyplot(fig_glucose)



  # Age vs Bp
  st.header('Blood Pressure Value Graph (Others vs Yours)')
  fig_bp = plt.figure()
  ax5 = sns.scatterplot(x = 'Age', y = 'BloodPressure', data = df, hue = 'Outcome', palette='Reds')
  ax6 = sns.scatterplot(x = user_data['Age'], y = user_data['BloodPressure'], s = 150, color = color)
  plt.xticks(np.arange(10,100,5))
  plt.yticks(np.arange(0,130,10))
  plt.title('0 - Healthy & 1 - Unhealthy')
  st.pyplot(fig_bp)


  # Age vs St
  st.header('Skin Thickness Value Graph (Others vs Yours)')
  fig_st = plt.figure()
  ax7 = sns.scatterplot(x = 'Age', y = 'SkinThickness', data = df, hue = 'Outcome', palette='Blues')
  ax8 = sns.scatterplot(x = user_data['Age'], y = user_data['SkinThickness'], s = 150, color = color)
  plt.xticks(np.arange(10,100,5))
  plt.yticks(np.arange(0,110,10))
  plt.title('0 - Healthy & 1 - Unhealthy')
  st.pyplot(fig_st)


  # Age vs Insulin
  st.header('Insulin Value Graph (Others vs Yours)')
  fig_i = plt.figure()
  ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = df, hue = 'Outcome', palette='rocket')
  ax10 = sns.scatterplot(x = user_data['Age'], y = user_data['Insulin'], s = 150, color = color)
  plt.xticks(np.arange(10,100,5))
  plt.yticks(np.arange(0,900,50))
  plt.title('0 - Healthy & 1 - Unhealthy')
  st.pyplot(fig_i)


  # Age vs BMI
  st.header('BMI Value Graph (Others vs Yours)')
  fig_bmi = plt.figure()
  ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = df, hue = 'Outcome', palette='rainbow')
  ax12 = sns.scatterplot(x = user_data['Age'], y = user_data['BMI'], s = 150, color = color)
  plt.xticks(np.arange(10,100,5))
  plt.yticks(np.arange(0,70,5))
  plt.title('0 - Healthy & 1 - Unhealthy')
  st.pyplot(fig_bmi)


  # Age vs Dpf
  st.header('DPF Value Graph (Others vs Yours)')
  fig_dpf = plt.figure()
  ax13 = sns.scatterplot(x = 'Age', y = 'DiabetesPedigreeFunction', data = df, hue = 'Outcome', palette='YlOrBr')
  ax14 = sns.scatterplot(x = user_data['Age'], y = user_data['DiabetesPedigreeFunction'], s = 150, color = color)
  plt.xticks(np.arange(10,100,5))
  plt.yticks(np.arange(0,3,0.2))
  plt.title('0 - Healthy & 1 - Unhealthy')
  st.pyplot(fig_dpf)



  # OUTPUT
  st.subheader('Your Report: ')
  output=''
  if user_result[0]==0:
    output = 'You are not Diabetic'
  else:
    output = 'You are Diabetic'
  st.title(output)
  st.subheader('Accuracy: ')
  st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')


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
  st.title("What is Diabetes?")
  st.write("Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into energy.")
  st.write("Your body breaks down most of the food you eat into sugar (glucose) and releases it into your bloodstream. When your blood sugar goes up, it signals your pancreas to release insulin. Insulin acts like a key to let the blood sugar into your body’s cells for use as energy.")

  st.write("With diabetes, your body doesn't make enough insulin or can’t use it as well as it should. When there isn’t enough insulin or cells stop responding to insulin, too much blood sugar stays in your bloodstream. Over time, that can cause serious health problems, such as heart disease, vision loss, and kidney disease.")

  st.write("There isn’t a cure yet for diabetes, but losing weight, eating healthy food, and being active can really help. Other things you can do to help:")

  st.write("1. Take medicine as prescribed.")
  st.write("2. Get diabetes self-management education and support.")
  st.write("3. Make and keep health care appointments.")
  st.image("D1.jpg")

  st.header("Prevention of Diabetes:")
  st.write("Preventing diabetes involves adopting a healthy lifestyle and making choices that promote overall well-being. Here are some key strategies for diabetes prevention:")

  st.write("1. Maintain a Healthy Weight: Being overweight or obese is a significant risk factor for type 2 diabetes. Aim for a healthy weight by adopting a balanced diet and engaging in regular physical activity.")

  st.write("2. Follow a Balanced Diet: Focus on consuming a variety of nutrient-dense foods, including fruits, vegetables, lean proteins, whole grains, and healthy fats. Limit intake of processed foods, sugary snacks, and beverages high in added sugars.")

  st.write("3. Stay Active: Regular physical activity can help lower blood sugar levels and improve insulin sensitivity. Aim for at least 150 minutes of moderate-intensity aerobic activity or 75 minutes of vigorous activity each week, along with muscle-strengthening exercises on two or more days per week.")

  st.write("4. Monitor Blood Sugar Levels: If you have prediabetes or are at risk of developing diabetes, monitoring your blood sugar levels regularly can help you track changes and take preventive measures.")

  st.write("5. Limit Alcohol Consumption: Excessive alcohol consumption can contribute to weight gain and increase the risk of developing type 2 diabetes. Limit alcohol intake to moderate levels, which is generally defined as up to one drink per day for women and up to two drinks per day for men.")

  st.write("6. Quit Smoking: Smoking increases the risk of various health complications, including type 2 diabetes. Quitting smoking can improve overall health and reduce the risk of developing diabetes.")

  st.write("7. Manage Stress: Chronic stress can affect blood sugar levels and increase the risk of developing type 2 diabetes. Practice stress-reducing techniques such as meditation, deep breathing exercises, or engaging in hobbies and activities you enjoy.")

  st.write("8. Get Adequate Sleep: Poor sleep habits and sleep deprivation can disrupt hormone levels and increase the risk of insulin resistance. Aim for 7-9 hours of quality sleep per night.")

  st.write("9. Regular Check-ups: Schedule regular check-ups with your healthcare provider to monitor your health, discuss any concerns, and receive guidance on diabetes prevention strategies tailored to your individual needs.")

  st.write("10. Know Your Risk Factors: Understanding your personal risk factors for diabetes, such as family history, age, and ethnicity, can help you take proactive steps to prevent or manage the condition effectively.")


  st.title("What is Diabetic retinopathy?")
  st.write("Diabetic retinopathy is an eye condition that can cause vision loss and blindness in people who have diabetes. It affects blood vessels in the retina (the light-sensitive layer of tissue in the back of your eye")

  st.write("If you have diabetes, it’s important to get a comprehensive dilated eye exam at least once a year. Diabetic retinopathy may not have any symptoms at first — but finding it early can help you take steps to protect your vision.") 

  st.write("Managing your diabetes — by staying physically active, eating healthy, and taking your medicine — can also help you prevent or delay vision loss.")

  st.header("What are the symptoms of diabetic retinopathy?") 
  st.write("The early stages of diabetic retinopathy usually don’t have any symptoms. Some people notice changes in their vision, like trouble reading or seeing faraway objects. These changes may come and go.")

  st.write("In later stages of the disease, blood vessels in the retina start to bleed into the vitreous (gel-like fluid that fills your eye). If this happens, you may see dark, floating spots or streaks that look like cobwebs. Sometimes, the spots clear up on their own — but it’s important to get treatment right away. Without treatment, scars can form in the back of the eye. Blood vessels may also start to bleed again, or the bleeding may get worse.")

  st.header("What other problems can diabetic retinopathy cause?")
  st.write("Diabetic retinopathy can lead to other serious eye conditions:") 

  st.write("1. Diabetic macular edema (DME). Over time, about 1 in 15 people with diabetes will develop DME. DME happens when blood vessels in the retina leak fluid into the macula (a part of the retina needed for sharp, central vision). This causes blurry vision.")
  st.write("2. Neovascular glaucoma. Diabetic retinopathy can cause abnormal blood vessels to grow out of the retina and block fluid from draining out of the eye. This causes a type of glaucoma (a group of eye diseases that can cause vision loss and blindness).")

  st.header("What causes diabetic retinopathy?")
  st.write("Diabetic retinopathy is caused by high blood sugar due to diabetes. Over time, having too much sugar in your blood can damage your retina — the part of your eye that detects light and sends signals to your brain through a nerve in the back of your eye (optic nerve).") 

  st.write("Diabetes damages blood vessels all over the body. The damage to your eyes starts when sugar blocks the tiny blood vessels that go to your retina, causing them to leak fluid or bleed. To make up for these blocked blood vessels, your eyes then grow new blood vessels that don’t work well. These new blood vessels can leak or bleed easily.")

  st.image("ret1.jpg")

  st.header("What can I do to prevent diabetic retinopathy?")
  st.write("Managing your diabetes is the best way to lower your risk of diabetic retinopathy. That means keeping your blood sugar levels in a healthy range. You can do this by getting regular physical activity, eating healthy, and carefully following your doctor’s instructions for your insulin or other diabetes medicines.")  

  st.write("To make sure your diabetes treatment plan is working, you’ll need a special lab test called an A1C test. This test shows your average blood sugar level over the past 3 months. You can work with your doctor to set a personal A1C goal. Meeting your A1C goal can help prevent or manage diabetic retinopathy.")

  st.header("What’s the treatment for diabetic retinopathy and DME?")
  
  st.write("In the early stages of diabetic retinopathy, your eye doctor will probably just keep track of how your eyes are doing. Some people with diabetic retinopathy may need a comprehensive dilated eye exam as often as every 2 to 4 months.") 

  st.write("In later stages, it’s important to start treatment right away — especially if you have changes in your vision. While it won’t undo any damage to your vision, treatment can stop your vision from getting worse. It’s also important to take steps to control your diabetes, blood pressure, and cholesterol.")

  st.image("ret2.jpg")
