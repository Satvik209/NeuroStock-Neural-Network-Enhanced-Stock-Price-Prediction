import math
import numpy as np 
import pandas as pd
from pandas_datareader import data as pdr
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.express as px
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import load_model
from tensorflow import keras
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go
import pandas_ta as ta
import yfinance as yf

st.set_page_config( page_title="NeuroStock", page_icon="chart_with_upwards_trend",layout="wide")
st.title('NeuroStock: Neural Network Enhanced Stock Price Prediction')

default_ticker = "AAPL"
ticker=st.sidebar.text_input(f"Ticker (default: {default_ticker}):") or default_ticker
start_date = st.sidebar.date_input("Start date", datetime.date(2012, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date(2023, 9, 30))


st.sidebar.write("""
# Popular Stocks
### Microsoft Corporation (MSFT)
### Apple Inc. (AAPL)
### Tesla, Inc. (TSLA)
### NVIDIA Corporation Common Stock (NVDA)
### JP Morgan Chase & Co. Common Stock (JPM)
### Coca-Cola Company (The) Common Stock (KO)
### Reliance Industries Limited (RELIANCE.NS)
### Netflix, Inc. (NFLX)
### Vanguard Intermediate-Term Bond ETF (BIV)
### BlackRock Global Dividend Ptf Investor A (BABDX)
### Emerson Electric Company Common Stock (EMR)
### Meta Platforms, Inc. Class A Common Stock (META)
### Walmart Inc. Common Stock (WMT)
### Tata Motors Limited (TATAMOTORS.NS)
### Tata Steel Limited (TATASTEEL.NS)
### Alphabet Inc. (GOOG)
### Amazon (AMZN)
### International Business Machines Corporation (IBM)
### Infosys Limited (INFY)
### Tata Consultancy Services Limited (TCS.NS)
""")

data1=yf.download(ticker, start=start_date, end=end_date)
data3=yf.Ticker(ticker)
df = data3.history(period='1d', start=start_date, end=end_date).reset_index()

try:
    data1=yf.download(ticker, start=start_date, end=end_date)
    data3=yf.Ticker(ticker)
    df = data3.history(period='1d', start=start_date, end=end_date).reset_index()

    string_name = data3.info['longName']
    st.header('**%s**' % string_name)

    string_summary = data3.info['longBusinessSummary']
    st.info(string_summary)

except Exception as e:
    st.error(f"An error occurred while fetching data for {ticker}: {str(e)}")

fig1=go.Figure(data=[go.Candlestick(x=df['Date'],
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Close'])])
fig1.update_layout(
        title=(f'CandleStick Chart of {ticker}'),
        yaxis_title='Price($)',
        xaxis_rangeslider_visible=False)
st.plotly_chart(fig1)


pricing_data, fundamental_data,news,models= st.tabs(["Pricing Data","Fundamental Data", "Top 10 News","Models"])

with pricing_data:
    st.header('Price Movements')
    data2=data1
    data2["% Change"] = data1["Close"].pct_change()*100
    data2.dropna(inplace=True)
    st.write(data2)
    annual_return = data2['% Change'].mean()*252
    st.write('Annual Return :', annual_return,'%')
    sd=np.std(data2['% Change'])*np.sqrt(252)
    st.write('Standard Deviation :',sd,'%')
    st.write('Risk Adj. Return :',annual_return/(sd*100))


with fundamental_data:
    st.subheader('Balance Sheet')
    st.write(data3.balance_sheet)
    st.subheader('Cash Flow Statement')
    st.write(data3.cashflow)
    

with models:
    col1,col2=st.columns((2))

    with col1:
        st.subheader("LSTM Model")

        series = df['Close']

        model1 = keras.models.load_model("my_checkpoint1.h5")

        def plot_series(time, series, format="-", start=0, end=None, label=None ,color=None):
            plt.plot(time[start:end], series[start:end], format, label=label,color=color)
            plt.xlabel("Time")
            plt.ylabel("Value")
            if label:
                plt.legend(fontsize=14)
            plt.grid(True)

        window_size = 30
        test_split_date = '2019-12-31'
        x_test = df.loc[df['Date'] >= test_split_date]['Close']

        x_test_values = x_test.values.reshape(-1, 1)
        x_train_scaler = MinMaxScaler(feature_range=(0, 1))
        x_train_scaler.fit(series.values.reshape(-1, 1))
        normalized_x_test = x_train_scaler.transform(x_test_values)

        rnn_forecast = model1.predict(normalized_x_test[np.newaxis,:])
        rnn_forecast = rnn_forecast.flatten()
        rnn_unscaled_forecast = x_train_scaler.inverse_transform(rnn_forecast.reshape(-1,1)).flatten()

        fig2=plt.figure(figsize=(7,4))

        plt.ylabel('Dollars $',color='white')
        plt.xlabel('Timestep in Days',color='white')
        plt.title(f'LSTM Forecast vs Actual',color='white')
        plot_series(x_test.index, x_test, label="Actual")
        plot_series(x_test.index, rnn_unscaled_forecast, label="Forecast")
        st.plotly_chart(fig2,use_container_width=True)

        lstm_mea=keras.metrics.mean_absolute_error(x_test, rnn_unscaled_forecast).numpy()
        st.write('LSTM Mean Absolute Error :',lstm_mea)
        lstm_rmse = np.sqrt(mean_squared_error(x_test,rnn_unscaled_forecast))
        st.write('LSTM Root Mean Square Error :',lstm_rmse)

    
    with col2:
        st.subheader("NAIVE BAYES Model")

        series = df['Close']

        test_split_date = '2021-12-01'
        test_split_index = np.where(df.Date == test_split_date)[0][0]
        x_test = df.loc[df['Date'] >= test_split_date]['Close']

        naive_forecast = series[test_split_index-1 :-1]

        fig3=plt.figure(figsize=(7,4))
        
        plot_series(x_test.index, x_test, label="Actual")
        plot_series(x_test.index, naive_forecast, label="Forecast")
        plt.ylabel('Dollars $',color='white')
        plt.xlabel('Timestep in Days',color='white')
        plt.title('Naive Forecast vs Actual',color='white')
        st.plotly_chart(fig3,use_container_width=True)

        naive_forecast_mae = keras.metrics.mean_absolute_error(x_test, naive_forecast).numpy()
        st.write('Naive Bayes Mean Absolute Error :',naive_forecast_mae)
        naive_forecast_rmse = np.sqrt(mean_squared_error(x_test,naive_forecast))
        st.write('Naive Bayes Root Mean Square Error :',naive_forecast_rmse)
        


    col1,col2=st.columns((2))

    with col1:
        st.subheader("CNN Model")

        window_size = 20
        model2 = keras.models.load_model("my_checkpoint.h5")

        def model_forecast(model, series, window_size):
            ds = tf.data.Dataset.from_tensor_slices(series)
            ds = ds.window(window_size, shift=1, drop_remainder=True)
            ds = ds.flat_map(lambda w: w.batch(window_size))
            ds = ds.batch(32).prefetch(1)
            forecast = model.predict(ds)
            return forecast

        x_train_scaler = MinMaxScaler(feature_range=(0, 1))
        x_train_scaler.fit(series.values.reshape(-1, 1)) 
        spy_normalized_to_traindata = x_train_scaler.transform(series.values.reshape(-1, 1))

        cnn_forecast = model_forecast(model2, spy_normalized_to_traindata[:,  np.newaxis], window_size)
        cnn_forecast = cnn_forecast[x_test.index.min() - window_size:-1,-1,0]
        cnn_unscaled_forecast = x_train_scaler.inverse_transform(cnn_forecast.reshape(-1,1)).flatten()

        fig4=plt.figure(figsize=(7,4))
        plt.ylabel('Dollars $',color='white')
        plt.xlabel('Timestep in Days',color='white')
        plt.title(f'Full CNN Forecast vs Actual',color='white')
        plot_series(x_test.index, x_test,label="Actual")
        plot_series(x_test.index, cnn_unscaled_forecast,label="Forecast")
        st.plotly_chart(fig4,use_container_width=True)

        cnn_mae=keras.metrics.mean_absolute_error(x_test, cnn_unscaled_forecast).numpy()
        st.write('CNN Mean Absolute Error :',cnn_mae)
        cnn_rmse = np.sqrt(mean_squared_error(x_test,cnn_unscaled_forecast))
        st.write('CNN Root Mean Square Error :',cnn_rmse)


    with col2:
        st.subheader("Linear Model")

        model4 = keras.models.load_model("my_checkpoint.h5")
        window_size = 30

        
        lin_forecast = model_forecast(model4, spy_normalized_to_traindata.flatten()[x_test.index.min() - window_size:-1], window_size)[:, 0]
        lin_forecast = x_train_scaler.inverse_transform(lin_forecast.reshape(-1,1)).flatten()

        fig5=plt.figure(figsize=(7,4))
        plt.title('Linear Forecast',color='white')
        plt.ylabel('Dollars $',color='white')
        plt.xlabel('Timestep in Days',color='white')
        plot_series(x_test.index, x_test,label="Actual")
        plot_series(x_test.index, lin_forecast,label="Forecast")
        st.plotly_chart(fig5,use_container_width=True)


        linear_mea = keras.metrics.mean_absolute_error(x_test, lin_forecast).numpy()
        st.write('LINEAR Mean Absolute Error :',linear_mea)
        linear_rmse = np.sqrt(mean_squared_error(x_test,lin_forecast))
        st.write('LINEAR Root Mean Square Error :',linear_rmse)



    st.subheader("LSTM-CNN Hybrid Model")

    series = df['Close']

    model1 = keras.models.load_model("my_checkpoint1.h5")

    def plot_series(time, series, format="-", start=0, end=None, label=None ,color=None):
            plt.plot(time[start:end], series[start:end], format, label=label,color=color)
            plt.xlabel("Time")
            plt.ylabel("Value")
            if label:
                plt.legend(fontsize=14)
            plt.grid(True)

    window_size = 30
    test_split_date = '2019-12-31'
    x_test = df.loc[df['Date'] >= test_split_date]['Close']

    x_test_values = x_test.values.reshape(-1, 1)
    x_train_scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_scaler.fit(series.values.reshape(-1, 1))
    normalized_x_test = x_train_scaler.transform(x_test_values)

    rnn_forecast = model1.predict(normalized_x_test[np.newaxis,:])
    rnn_forecast = rnn_forecast.flatten()
    rnn_unscaled_forecast = x_train_scaler.inverse_transform(rnn_forecast.reshape(-1,1)).flatten()

    X = df[['Close']].values
    y = df['Close'].shift(-1).values 
    X = X[:-1]
    y = y[:-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
    from sklearn.preprocessing import MinMaxScaler
    from xgboost import XGBRegressor
    from sklearn.linear_model import LinearRegression

    rf_regressor = ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    gb_regressor = ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
    xgb_regressor = ('xgb', XGBRegressor(n_estimators=100, random_state=42))

    stacking_regressor = StackingRegressor(
        estimators=[rf_regressor, gb_regressor, xgb_regressor],
        final_estimator=LinearRegression(),
    )

    fig2=plt.figure(figsize=(7,4))
    plt.ylabel('Dollars $',color='white')
    plt.xlabel('Timestep in Days',color='white')
    plt.title(f'Hybrid Forecast vs Actual',color='white')
    plot_series(x_test.index, x_test, label="Actual")
    plot_series(x_test.index, rnn_unscaled_forecast, label="Forecast")
    st.plotly_chart(fig2,use_container_width=True)

    X_train_rf = X_train.reshape(X_train.shape[0], -1)
    X_test_rf = X_test.reshape(X_test.shape[0], -1)
    stacking_regressor.fit(X_train_rf, y_train)
    y_pred_rf = stacking_regressor.predict(X_test_rf)
    hybrid_mae_rf = mean_absolute_error(y_test, y_pred_rf)
    hybrid_rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    st.write('LSTM-CNN Hybrid Model Mean Absolute Error :',hybrid_mae_rf)
    st.write('LSTM-CNN Hybrid Model Root Mean Square Error :',hybrid_rmse_rf)
    

from stocknews import StockNews
with news:
    st.header(f'News of {ticker}')
    sn=StockNews(ticker, save_news=False)
    df_news= sn.read_rss()
    for i in range(10):
        st.subheader(f'News {i+1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment= df_news['sentiment_title'][i]
        st.write(f'Title Sentiment {title_sentiment}')
        news_sentiment= df_news['sentiment_summary'][i]
        st.write(f'News Sentiment {news_sentiment}')



# from pyChatGPT import ChatGPT
# session_token ='eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..mCRxKfY-DNrcnrxJ.CBgw3NN1V-Ec_jjsX1pRZs0IRBlQAHSS7D1E2rxn5H8C8Q9ubXLVLO_xdMTC_wNMK2B2-GXX5e2ns5fo7RNt1EMAYo0HBj1MOw9bFxm4e45WakyCXEVzHTX-EbvGTPwkww33Hlgj1OrENeyLN5w8dN3MvDhyrftjLhSJipaK05lFRmyOJ-qk39paX3CIsNnmIaFXd_fEFEF3XsQvx7ao2AJyVpue6DBCM1C8BxQcxpdSmmQTOEcfbz7ZzN072ghKZoeCqfKmVZ6wllY6nUTj_rpgmExoSR3KBdV9GJ27PlvxNPDseNc63uhqLdb9ZfTbM6EDBrOXwgaIk58cjDHSDsI1IzPBVJbrDtY8SKCKjNSgFlD2pJ9uueWn8J1RyZ6I0jPgGPQVAQiK3WXSx9EcQQW1UocXR05QZhBf5t4sUt6_xPz22oSTPPTDVvI3TW-y0MNEh2XC4Q8j1J1BF5E3dLFdmlNlFAfPyqnn4qJPX7uedOurJmqmODL9-hKkEvLT0Ng1KB5PRBqlFCGuAuTzQx89kOMhxllmJg7iD3IfQQH-MHrPqFi2BtsfnGSGr179Fpen-SPQlMaFoeedLWFiTLWXn7UW7LsEkE7j3yt4kODT9LQjQVolL-w0sFiHyYJyQIU37tPq3YcuV26f9Jwtl-QNMrTJmCPMAEJrrhCMVjeodKiCy_3zIXfmALZLG0eWJNdJe6NLB_J5NgRzad4g9KinX0ZTQ3USDcGqP74Tys9VOY5Da2ThqGh7ku-BXQ3qIGwSCs1pEAm3cvjyBPKzmy3ptECF3QW-N8Xb_B0Ys1dpbz5yJUMGbNgi7dx4USHr77kcwEWfS3R2DcSbHwL4_-7Zm3r7Ycr6GkRfi5aJuvHhagoJOyDEABWU5q4yxZoXX9EYlbRVXNCjRtFWqIrfHvMppwRw0sC0Qf-iPScTc8WzEcxRQ3sITLRJT6_grtudsmhoxQ4_H45oHJaj-KxToDBzllr2CTCdbMU-XMSCIblD3HMP6KO52uLdtnFS7QyaMdTPDk9n0AppsfyO4Irz9fvXmmABLMCNRN8AL3W2iP3GPSF6f_MtZv5udhDOyfuV81BfqBXJBktqRZ8uurUAm7dwZ5Sjrzpbt2e_kStB7DQaaWXklQpelKS2cgNLM7riJ4k_fAsDyqdY00zHUnroy8FSXZSwrUM7uc8BDszqHESXHtHUykDXfkQ0ilOZ2ovph-CtEMsInLkCE-_xG7dRY0faVazEvlK23ypwsCaKhL5sn3FhXXOE_A6HWfKEnZKr_E-aONrIClS-qslHZwinvF6WO-rUpkPyK32kgUTHMTtJOcMTJ5vf63pKDupmhw_C0LJTDgDE4WeM17JNj_9-LkriFyYxxM7Ex4FJQOVrCkkCN_35nsfxRR152JNt6ncJpUejQWE3FMUYsLpiGs6G3aIu4Kzc8ay9D0r86EKk3kMzhCzN420DY78nwRMJgojMB4K_WgqmUz_augqhcd-AmbUbXXvfVyVNTvvOokBbO6uV8IbdQiVmgxutXjn1Y-MRt3euTfCY9rRgoQwC6G4azUuk5QkqW4LBp4FTq3fAfFFQAGOauMsjN81Atn3VnM1NjyvySFM2rE6TBCTwW3631lMt7b067IwmvPqQ5gtlBCfRRwT29WILZ81A2eExj0d9czdPd-TioJgKfoPwtRy3b8MHd3-jmr4s_yzn-NBHRP__-zYsqiFuIRB7Vjm_dK2kwHR68CMw4SS4mvNz9JQ1QrJvsvuvw4BzdDinfAvtG9_xACBHsWR4XXh0fRwFTnziZhOXGOpP-UeOQSgIJC2l0sU2G76RO43gGMT8F5AHQmhFmLW3bnnKsi1IMMncBhWwT9lUHDyAmHrgNjdwMYqTkqteZHYFFKcLrY3uFdLIGkiuYiySP6x5-OFhLR6DFwPHeJHwMXvI80ioIWwtYKgto1fHMsgA72VakXsDJpipyz4SRZylLqBEJlTKCPqyglrKtm0LtQtRDceCuJCA07A8RWw1Nv7X1jMuFv8wrTiOlUP5b07kw1Rsp70PkbXZQE9kA_MMqkzmTKr9GXrwobFTK-nxL7Wjo4pZuUJKdzLNHOoy1op_zScnuWFENAJ8KwCttcIAKMm14aYuehF-TofWB5vfc_j-Pp2dyhCeA2-_VGRgJc4l3K1avkjcVnGyzy4yyy2TobYNQ55-WWIceBm-CO1HitY1B6zW1qh8Vd_uvufUMF7u467bGtr74OMsAv_M1a9Y_qxdfLppslZEIGL6IfQFSOvo7kCERbqOdA2-xcNdPjjGli-3jcyVFqkzlfiMdaLr69C2qRHlQyQPSJTqIsIZxmCkQfmFb1kHOndQ6blwSYG7779HD72DBJrB8hJ54HCYB_GT2DRJjG2m6CeoI31kBEdxhHe8Eyl-5Z0WSD-gitQZyjX_bBXDfYjIVilJrSAedJIhmcTxtMpql1Yl43pglOn5fbnA4vwi1EMCYAlIBHQ2f0ejFYkXExfCEPmQ60H9zwaI94Rl9RDeXIiEV5rHt9FEr5e_oBLAsEe3rFBPg2jRxpsrakwFEb1G6plP-NHovMdpydrjP2X5HzztNCzkjrYF5oOYafBHx9sVy1MDjSIHDAhkuNbOmwTBjU7svOIbGXWVCWmFucU-3kHIoMnMYlc_zHbeJq6GVqIbnI8XOEWZCker04addI_uBaQOhTrxfcQ0cOUrDMSlKNwuOggOvDtez04J94x61q79uVg7L5FpCK9qEtSBpOLiuG46hlJp2TxCYMVyXV9b614qLnKkQzWOobaymZS9J3OX126jHQHm3DUHBV1wUsUzHCMZzTSLuGKZoMLN3eoFwUaIemdqy8zTn-u3cf0PvMfKjpgQ6X-wVcnNgP6w8OBAuEhpYkC0mxyah0L8.Ao0Z5lvjkjJjAWa3tRbihA'
# api2=ChatGPT(session_token)
# buy = api2.send_message(f"5 Reasons to buy {ticker} stock")
# sell = api2.send_message(f"5 Reasons to sell {ticker} stock")
# swot = api2.send_message(f"SWOT Analysis of {ticker} stock")

# with openai:
#     buy_reason,sell_reason,swot_analysis=st.tabs["5 Reasons to Buy","5 Reasons to Sell","SWOT Analysis "]

#     with buy_reason:
#         st.subheader(f"5 Reasons on why to BUY {ticker} stock")
#         st.write(buy['message'])

#     with sell_reason:
#         st.subheader(f"5 Reasons on why to SELL {ticker} stock")
#         st.write(sell['message'])

#     with swot_analysis:
#         st.subheader(f"SWOT Analysis of {ticker} stock")
#         st.write(swot['message'])
