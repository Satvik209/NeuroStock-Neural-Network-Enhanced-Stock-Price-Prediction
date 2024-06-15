# NeuroStock: Neural Network Enhanced Stock Price Prediction

NeuroStock is an advanced tool designed for comprehensive stock analysis and prediction. The application allows users to fetch and customize stock data from Yahoo Finance for any chosen period and ticker symbol. It provides an in-depth view of stock price movements through interactive candlestick charts and calculates key financial metrics such as annual return, standard deviation, and risk-adjusted return.

Users can access fundamental data including the Balance Sheet and Cash Flow Statement of the selected stock, offering insights into the company's financial health. NeuroStock integrates multiple predictive models, including LSTM, CNN, Naive Bayes, Linear, and a sophisticated LSTM-CNN hybrid algorithm, to forecast future stock prices. Each model's predictions are visually compared against actual prices, and their performance is evaluated using mean absolute error and root mean square error metrics.

Additionally, the platform keeps users informed with the latest top 10 news articles related to the selected stock, complete with sentiment analysis of the headlines and summaries. NeuroStock combines state-of-the-art neural network algorithms with essential financial data and news, providing a powerful tool for stock market enthusiasts and investors.


## Application Features

**Interactive Stock Data Visualization:** Users can visualize stock data with interactive candlestick charts.

**Fundamental Data Display:** Users can view the balance sheet and cash flow statements of the selected stock.

**Predictive Models:** Includes LSTM, CNN, Naive Bayes, Linear, and a hybrid LSTM-CNN model for stock price prediction.

**Performance Metrics:** Calculates and displays mean absolute error and root mean square error for each model's predictions.

**News Integration:** Displays the latest news articles and sentiment analysis for the selected stock.

**Sidebar Options:** Allows users to input the ticker symbol and select the date range for analysis.
## Tech Stack

NeuroStock utilizes a combination of cutting-edge technologies and libraries to provide a comprehensive stock analysis and prediction platform. Here are the key technologies used in the application:

***1. Data Fetching and Processing***
**Yahoo Finance API (yfinance):** Used to fetch historical stock data and fundamental financial data.

**Pandas (pandas):** Utilized for data manipulation, cleaning, and analysis.
Numpy (numpy): Employed for numerical computations.

 ***2.Data Visualization***

**Plotly (plotly.express and plotly.graph_objects):** Used to create interactive candlestick charts and other visualizations.

**Matplotlib (matplotlib.pyplot):** Utilized for plotting data series and model forecasts.

***3. Machine Learning and Deep Learning Models***

**TensorFlow (tensorflow) and Keras (keras):** Deep learning frameworks used to build and load various predictive models like LSTM, CNN, and a Linear model.

**Scikit-Learn (sklearn):** Machine learning library used for preprocessing data (e.g., MinMaxScaler), model evaluation (e.g., mean absolute error, mean squared error), and building traditional ML models like Naive Bayes and Linear Regression.



***4. Web Application Development***

**Streamlit (streamlit):** Framework for creating and hosting the interactive web application. Streamlit is used for building the user interface, allowing for user inputs, displaying charts, and organizing tabs for different functionalities.

***5. Stock News Integration***

**StockNews API (stocknews):** Used to fetch the latest news articles related to the selected stock, providing an additional layer of information for the user.

***6. Financial and Technical Analysis***

**Pandas TA (pandas_ta):** Technical analysis library for adding various technical indicators to the dataset.

**DateTime (datetime):** Used for handling date and time operations in the application.

***7. Other Tools and Libraries***

**Python:** The core programming language used for developing the entire application.

**Try-Except Blocks:** Used for error handling to ensure the application runs smoothly and handles exceptions gracefully.
