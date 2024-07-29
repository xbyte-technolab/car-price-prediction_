'''
import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import time

model = pk.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Car Price Prediction", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center;'>Car Price Prediction App</h1>
    """, unsafe_allow_html=True
)

image = Image.open("Designer (22).png")

width, height = image.size
new_width = 1000  
new_height = int((new_width / width) * height)
resized_image = image.resize((new_width, new_height))
st.image(resized_image)
st.markdown("<hr>", unsafe_allow_html=True)  

car_df = pd.read_csv("Cardetails.csv")

def brand_name(car_name):
    return car_name.split(" ")[0].strip()

def model_name(car_name):
    return " ".join(car_name.split(" ")[1:]).strip()

car_df["brand"] = car_df["name"].apply(brand_name)
car_df["model_name"] = car_df["name"].apply(model_name)

col1, col2 = st.columns([2, 1])
with col1:
    brand = st.selectbox("Select Car Brand", car_df["brand"].unique())
    st.write('') 
    model_options = car_df[car_df["brand"] == brand]["model_name"].unique()
    model_name = st.selectbox("Car Model Name", model_options)
    st.write('') 
    fuel = st.selectbox("Fuel Type", car_df["fuel"].unique())
    st.write('') 
    seller_type = st.selectbox("Seller Type", car_df["seller_type"].unique())
    st.write('') 
    transmission = st.selectbox("Transmission Type", car_df["transmission"].unique())
    st.write('')
    owner = st.selectbox("Owner Type", car_df["owner"].unique())

with col2:
    slider_col = st.container()

with slider_col:
    year = st.slider("Car Manufactured Year", 1994, 2024, key="year_slider")
    km_driven = st.slider("No of KM Driven", 1, 200000, key="km_slider")
    mileage = st.slider("Car Mileage", 10, 40, key="mileage_slider")
    engine = st.slider("Engine CC", 700, 5000, key="engine_slider")
    max_power = st.slider("Max Power", 0, 200, key="power_slider")
    seats = st.slider("No of Seats", 2, 10, key="seats_slider")

if st.button("Predict",use_container_width=True):
    data_model = pd.DataFrame(
        [[brand, year, km_driven, fuel, seller_type,
          transmission, owner, mileage, engine, max_power, seats, model_name]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats', 'model_name'] 
    )
    
    lab = LabelEncoder()
    data_model["name"] = lab.fit_transform(data_model["name"])
    data_model["transmission"] = lab.fit_transform(data_model["transmission"])
    data_model["seller_type"] = lab.fit_transform(data_model["seller_type"])
    data_model["fuel"] = lab.fit_transform(data_model["fuel"])
    data_model["owner"] = lab.fit_transform(data_model["owner"])
    data_model["model_name"] = lab.fit_transform(data_model["model_name"])

    #st.write(data_model)
    predicted_price = model.predict(data_model)

    message = f"Car Price Will be {predicted_price[0]} 🚗"
    words = message.split()
    placeholder = st.empty()
    display_message = ""
    for word in words:
        display_message += word + " "
        placeholder.markdown(f"<h1 style='font-size:35px;'>{display_message.strip()}</h1>", unsafe_allow_html=True)
        time.sleep(0.2)
        st.balloons()

'''
import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import time
import compress_pickle
import base64

# Load the model
model = compress_pickle.load(open("best_model.gz", "rb"))

# Set page configuration
st.set_page_config(page_title="Car Price Prediction", layout="centered")

# Function to load and encode the image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# background image
background_image_path = 'Designer (37).png'
bg_img_base64 = get_base64_of_bin_file(background_image_path)

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{bg_img_base64}");
    background-size: cover;
    background-position: center;
}}
[data-testid="stHeader"] {{
    background: rgba(0, 0, 0, 0);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown(
    """
    <h1 style='text-align: center;'>Car Price Prediction App</h1>
    """, unsafe_allow_html=True
)

# Load and resize the header image
#image = Image.open("Designer (21).png")
#width, height = image.size
#new_width = 1000  
#new_height = int((new_width / width) * height)
#resized_image = image.resize((new_width, new_height))
#st.image(resized_image)
#st.markdown("<hr>", unsafe_allow_html=True)

# Load car data
car_df = pd.read_csv("Cardetails.csv")

# Functions to extract brand and model names
def brand_name(car_name):
    return car_name.split(" ")[0].strip()

def model_name(car_name):
    return " ".join(car_name.split(" ")[1:]).strip()

car_df["brand"] = car_df["name"].apply(brand_name)
car_df["model_name"] = car_df["name"].apply(model_name)

col1, col2 = st.columns([2, 1])
with col1:
    brand = st.selectbox("Select Car Brand", car_df["brand"].unique())
    st.write('') 
    model_options = car_df[car_df["brand"] == brand]["model_name"].unique()
    model_name = st.selectbox("Car Model Name", model_options)
    st.write('') 
    fuel = st.selectbox("Fuel Type", car_df["fuel"].unique())
    st.write('') 
    seller_type = st.selectbox("Seller Type", car_df["seller_type"].unique())
    st.write('') 
    transmission = st.selectbox("Transmission Type", car_df["transmission"].unique())
    st.write('')
    owner = st.selectbox("Owner Type", car_df["owner"].unique())

with col2:
    slider_col = st.container()

with slider_col:
    year = st.slider("Car Manufactured Year", 1994, 2021, key="year_slider")
    km_driven = st.slider("No of KM Driven", 1, 200000, key="km_slider")
    mileage = st.slider("Car Mileage", 10, 40, key="mileage_slider")
    engine = st.slider("Engine CC", 700, 5000, key="engine_slider")
    max_power = st.slider("Max Power", 0, 200, key="power_slider")
    seats = st.slider("No of Seats", 2, 10, key="seats_slider")

# Prediction button
if st.button("Predict", use_container_width=True):
    data_model = pd.DataFrame(
        [[brand, year, km_driven, fuel, seller_type,
          transmission, owner, mileage, engine, max_power, seats, model_name]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats', 'model_name']
    )

    lab = LabelEncoder()
    data_model["name"] = lab.fit_transform(data_model["name"])
    data_model["transmission"] = lab.fit_transform(data_model["transmission"])
    data_model["seller_type"] = lab.fit_transform(data_model["seller_type"])
    data_model["fuel"] = lab.fit_transform(data_model["fuel"])
    data_model["owner"] = lab.fit_transform(data_model["owner"])
    data_model["model_name"] = lab.fit_transform(data_model["model_name"])

    # Predict the price
    predicted_price = model.predict(data_model)

    # Display the predicted price
    with st.spinner('Processing...'):
        time.sleep(2)  # Simulate a long computation
    st.success('Done!')
    message = f"Car Price Will be {round(predicted_price[0],2)} 🚗"
    st.markdown(f"<h1 style='font-size:35px;'>{message.strip()}</h1>", unsafe_allow_html=True)
    st.balloons()