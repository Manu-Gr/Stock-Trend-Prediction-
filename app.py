import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
from keras.models import load_model
import streamlit as st
import base64



start = '2000-01-01'
end   =  date.today()

st.title('Stock Trend Prediction')

st.markdown('Project by - **Manu G**')
st.markdown("***")
user_input = st.text_input("Enter the Stock Ticker",'TCS.NS')
st.markdown("***")
st.write("Please Find The Name Of the Stock Ticker From This [Link](https://finance.yahoo.com/quote/INFY?p=INFY&.tsrc=fin-srch)")

df= data.DataReader("user_input",'yahoo',start,end)

df1 = df.copy()

# Describing Data
st.markdown("***")
st.subheader('Stock Data from Year 2000 to Present Day')
st.write(df.describe())


# Visualizations
st.markdown("***")
st.subheader("Closing Price Vs Time")
close_plot = px.line(df,x=None,y='Close',)
st.plotly_chart(close_plot,)



st.markdown("***")
st.subheader("Closing Price Vs 100 Moving Avg")
ma100 = df1.Close.rolling(100).mean()

fig= plt.figure(figsize=(14,7))
plt.plot(df.Close)
plt.plot(ma100,)
st.pyplot(fig)

st.markdown("***")
st.subheader("Closing Price Vs 200 Moving Avg")
ma200 = df1.Close.rolling(200).mean()
fig= plt.figure(figsize=(14,7))
plt.plot(df.Close)
plt.plot(ma200,)
st.pyplot(fig)

st.markdown("***")
st.subheader("Closing Price Vs 100 and 200 Moving Avg")
ma200 = df1.Close.rolling(200).mean()
ma100 = df1.Close.rolling(100).mean()
fig= plt.figure(figsize=(14,7))
plt.plot(df.Close)
plt.plot(ma100,)
plt.plot(ma200,)
st.pyplot(fig)

# Spliting data into Train and Test

data_training = pd.DataFrame(df1['Close'][0:int(len(df1)*0.70)])
data_testing  = pd.DataFrame(df1['Close'][int(len(df1)*0.70):int(len(df1))])

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)

# Spliting data into x and y



# Load Model


model= load_model('keras_model100.h5')

# Testing Part

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing,ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])


x_test,y_test = np.array(x_test),np.array(y_test)

# Predictions 

y_predicted = model.predict(x_test)

scaler= scaler.scale_

scale_factor = 1/scaler[0]

y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final Visuals

st.markdown("***")
st.subheader('Predicted Trend VS Original Trend')
fig2= plt.figure(figsize=(12,6))
plt.plot(y_test,'g',label = 'Original Price')
plt.plot(y_predicted,'r',label= 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
