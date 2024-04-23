
import streamlit as st
import numpy as np
import pickle
import pandas as pd


pipe = pickle.load(open('pipe.pkl', 'rb'))
model = pickle.load(open('Model.pkl', 'rb'))

car = pd.read_csv("Cleaned_Car_data.csv")
st.title("Vehicle Price Predictor")
from PIL import Image

img = Image.open("car.jpeg")



# display image using streamlit
# width is used to set the width of an image
st.image(img, width=200)
brand = st.selectbox('Brand', sorted(car['brand'].unique()))
model2 = st.selectbox('Model', sorted(car['model'].unique()))
year = st.selectbox('Year', sorted(car['year'].unique()))
mileage = st.text_input('Mileage')
color = st.selectbox('Color', sorted(car['color'].unique()))
state = st.selectbox('State', sorted(car['state'].unique()))


if st.button('Predict Price'):

    if mileage.isnumeric():
        query = np.array([brand, model2, year, mileage, color, state])
        query = query.reshape(1, 6)
        solution =(pipe.predict(pd.DataFrame(columns=['brand', 'model', 'year', 'mileage', 'color', 'state'], data=query)))
        st.subheader(f" You can expect ${solution[0]:.0f} for the vehicle")
    else:

        st.error("Please enter a valid mileage!")

