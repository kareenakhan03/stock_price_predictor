# import streamlit as st
# import pandas as pd
# import numpy as np
# from keras.models import load_model
# import matplotlib.pyplot as plt
# import yfinance as yf

# st.title('Stock Price Predictor App')   
# stock = st.text_input("Enter the stock ID", "GOOG")


# from datetime import datetime

# end = datetime.now()
# start = datetime(end.year - 20, end.month, end.day)

# google_data = yf.download(stock, start, end)

# model = load_model("Lastest_stock_price_model.keras")
# st.subheader("Stock Data")
# st.write(google_data)

# splitting_length = int(len(google_data) * 0.7)
# x_test= pd.DataFrame(google_data.Close[splitting_length:])

# def plot_graph(figsize, value, full_data):
#     fig = plt.figure(figsize=figsize)
#     plt.plot(value, 'Orange')
#     plt.plot(full_data.close , 'b')
#     return fig
    

import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

st.title('Stock Price Predictor App')

# store input to variable
stock = st.text_input("Enter the stock ID", "GOOG")

end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

google_data = yf.download(stock, start, end)
st.subheader("Stock Data")
st.write(google_data)

model = load_model("Lastest_stock_price_model.keras")
from tensorflow.keras.models import load_model

# model = load_model(r"C:\Users\91708\OneDrive\Desktop\login_page\web_stock_price_predictor\lastest_stock_price_model.keras")


splitting_length = int(len(google_data) * 0.7)
x_test = pd.DataFrame(google_data['Close'][splitting_length:])
x_test.columns = ['Close']  # Explicitly set column name

# def plot_graph(figsize, value, full_data, extra_data=0, extra_dataset=None):
#     fig = plt.figure(figsize=figsize)
#     plt.plot(full_data['Close'], label="Original", color='blue')
#     plt.plot(value, label="Predicted", color='orange')
#     if extra_data:
#         plt.plot(extra_dataset)
#     plt.legend()
#     plt.show()
#     st.pyplot(fig)


def plot_graph(figsize, value, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)

    plt.plot(full_data['Close'], label="Original", color='blue')
    plt.plot(value, label="Predicted", color='orange')

    if extra_data and extra_dataset is not None:
        # # Handle extra_dataset as Series or tuple
        # if isinstance(extra_dataset, tuple) and len(extra_dataset) == 2:
        #     plt.plot(extra_dataset[0], extra_dataset[1], label="Extra Data", color='green')
        # else:
            plt.plot(extra_dataset, label="Extra Data", color='green')

    plt.legend()
    plt.grid(True)
    plt.title("Stock Price Plot")
    plt.xlabel("Date")
    plt.ylabel("Price")
    
    st.pyplot(fig)


# later you can call plot_graph
st.subheader('Original close price and MA for 250 days')
google_data['MA_for_250_days'] = google_data.Close.rolling(window=250).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'], google_data,0))

st.subheader('Original close price and MA for 200 days')
google_data['MA_for_200_days'] = google_data.Close.rolling(window=200).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'], google_data,0))

st.subheader('Original close price and MA for 100 days')
google_data['MA_for_100_days'] = google_data.Close.rolling(window=100).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'], google_data,0))

st.subheader('Original close price and MA for 100 days and MA for 250 days')
# google_data['MA_for_250_days'] = google_data.Close.rolling(window=250).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'], google_data,1))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']])

# x_test = []
# y_test = []

# for i in range(100, len(scaled_data)):
#     x_test.append(scaled_data[i-100:i ])
#     y_test.append(scaled_data[i])

# x_data, y_data = np.array(x_data), np.array(y_data)
x_seq = []
y_seq = []

for i in range(100, len(scaled_data)):
    x_seq.append(scaled_data[i-100:i])
    y_seq.append(scaled_data[i])

x_data, y_data = np.array(x_seq), np.array(y_seq)


predicted_price = model.predict(x_data)

inv_pre = scaler.inverse_transform(predicted_price)

inv_y_test = scaler.inverse_transform(y_data)





# print(plotting_data.head())  


# plotting_data = pd.DataFrame(
#     {
#         'Original_test_data': inv_y_test.reshape(-1),
#         'Predictions': inv_pre.reshape(-1)
#     },
#     index=google_data.index[-len(inv_pre):]  # Make sure index length matches
# )

plotting_data = pd.DataFrame({
    'Original_test_data': inv_y_test.reshape(-1),
    'Predictions': inv_pre.reshape(-1)
}, index=google_data.index[-len(inv_pre):])

print(plotting_data.head())

st.subheader('Original value vs Predicted value')
st.write(plotting_data.head())

st.subheader('Original close price vs predicted close price')
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([google_data.Close[:splitting_length+100], plotting_data], axis=0))
plt.legend(["Data - not used", "Original test data", "Predicted test data"])
st.pyplot(fig)



