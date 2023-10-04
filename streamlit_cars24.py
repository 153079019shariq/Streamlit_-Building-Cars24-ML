import streamlit as st
import datetime
import pickle
import pandas as pd

cars_df = pd.read_csv("./cars24-car-price.csv")

st.write(
    """
     # Cars24 Used Car Price Prediction
    """
)
st.dataframe(cars_df.head())

## Encoding Categorical features
encode_dict = {
    "fuel_type": {'Diesel': 1, 'Petrol': 2, 'CNG': 3, 'LPG': 4, 'Electric': 5},
    "seller_type": {'Dealer': 1, 'Individual': 2, 'Trustmark Dealer': 3},
    "transmission_type": {'Manual': 1, 'Automatic': 2}
}

def model_pred(fuel_type, transmission_type, engine, seats):

    ## loading the model
    with open("model_files/car_pred", 'rb') as file:
        reg_model = pickle.load(file)

        input_features = [[2018.0, 1, 4000, fuel_type, transmission_type, 19.70, engine, 86.30, seats]]

        return reg_model.predict(input_features)

col1, col2 = st.columns(2)

fuel_type = col1.selectbox("Select the fuel type",
                           ["Diesel", "Petrol", "CNG", "LPG", "Electric"])

engine = col1.slider("Set the Engine Power",
                     500, 5000, step=100)

transmission_type = col2.selectbox("Select the transmission type",
                                   ["Manual", "Automatic"])

seats = col2.selectbox("Enter the number of seats",
                       [4,5,7,9,11])

if (st.button("Predict Price")):

  fuel_type = encode_dict['fuel_type'][fuel_type]
  transmission_type = encode_dict['transmission_type'][transmission_type]

  price = model_pred(fuel_type, transmission_type, engine, seats)
  st.text("Predicted Price of the car is: "+str(price))



st.title("Understanding the data")
st.text("The value of a car drops right from the moment it is bought and the depreciation continues with each passing year. \nIn fact, in the first year itself, the value of a car decreases by 20 percent of its initial value.\nThe make and model of a car, total kilometers driven, overall condition of the vehicle and various other factors further affect the car's resale value.")
st.subheader("\n Here we will try to understand how the different features affect the selling price.\n ")


#######################################__EDA_Analysis__########################################################
import seaborn as sns
import matplotlib.pyplot as plt
df_orig = pd.read_csv("./cars24-car-price.csv")
st.header("Understanding the data")
st.dataframe(df_orig.head(25))

########  section 3   ########

col1, col2 = st.columns(2)

########  column 1   ########

col1.subheader("Pairplot against selling price")
var1 = col1.selectbox(" Select Column for pairplot: ",
                     ['year','mileage','seller_type','km_driven','fuel_type','transmission_type','engine','max_power','seats'])
 
col1.text("selling price vs "+var1)

plot=sns.pairplot(y_vars=['selling_price'],x_vars=[var1], data=df_orig, height=8)
col1.pyplot(plot)

########  column 2   ########

col2.subheader("Histogram")
var2 = col2.selectbox(" Select Column for histogram: ",
                     ['mileage','year','seller_type','km_driven','fuel_type','transmission_type','engine','max_power','seats'])

bins=col2.slider('bins', 10, 100)
col2.text("Histogram :"+var2)

fig = plt.figure()
sns.histplot(x = var2, data = df_orig, bins=bins)
col2.pyplot(fig)