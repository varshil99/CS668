
# Twitter Sentiment Analysis for Stock Price Movement

**Varshil R. Mehta**








## Objective
To Study the impact of Twitter sentiment data and study its correlation with the prices of multiple stocks movement and enabling predictive insights

## Motivation

This project focuses to explore the influence of social media trends(particularly Twitter), on stock market . With the growing day-to-day public communication and awareness posts, social platforms play a crucial role in shaping investor sentiment and market dynamics. By making the use of sentiment analysis, this study will help investors in making their decisions by identifying trends and behavioural patterns by public sentiment.

## Research Questions

1. How does Twitter sentiment impact short-term and long-term stock price movements?
2. Can sentiment analysis effectively predict bullish or bearish market trends?
3. What is the major impact of fake negative news on the stocks?
## Literature Review

Research indicates that social media, particularly Twitter, has a significant effect on market sentiment and stock price movements. Current studies leverage natural language processing (NLP) techniques and machine learning models to extract sentiment from tweets. Techniques such as VADER, TextBlob, and transformers (e.g., BERT) have strong impact on stock market trends.
## Dataset Overview

The project utilizes multiple datasets:

1. Twitter Data: Collected from Twitter’s API, which contains tweet text, timestamps, hashtags, and user metadata.
2. Stock Market Data: Historical price data for multiple stocks, obtained from Yahoo Finance

Key Features Include: 
• Tweet Data: Text content, sentiment score, hashtags, and user engagement metrics (likes, retweets are the major measures for check)
• Stock Data: Open, high, low, close prices, volume, and daily price changes.
## EDA & Methodology

Exploratory Data Analysis (EDA):
 • Used VADER and BERT-based models to perform sentiment analysis on public tweets. 
 • Collected sentiment scores with stock price movement (e.g., percentage change). 
 • Visualized sentiment trends against stock price trends of real time. Modeling Approach:

1. Sentiment Analysis:
o Tools: VADER, BERT, FinBERT 
o Outputs: Sentiment polarity (positive, neutral, negative) and compound scores.
2. Stock Movement Prediction:
o Features: Sentiment scores, volume, historical price trends, and event-based metrics. 
o Models Used: RandomForest, XGBoost, and LSTM for time-series analysis.
3. Optimization & Experimentation: 
o Fine-tuned hyperparameters via Grid Search. 
o Incorporated feature importance analysis using SHAP to interpret model outputs.

#Model Results & Evaluation

Sentiment Analysis Results: 
• Accuracy of sentiment classification: 83% (BERT-based model). 
Stock Price Movement Prediction: 
• Models Used: scikit-learn, LSTM 
• Evaluation Metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score.
• Results: 
o XGBoost: MAE = 1.10, RMSE = 2.35, R² = 0.79 
o LSTM: MAE = 1.04, RMSE = 2.07, R² = 0.81
## Model Results & Evaluation

Sentiment Analysis Results: 
• Accuracy of sentiment classification: 83% (BERT-based model). 
Stock Price Movement Prediction: 
• Models Used: scikit-learn, LSTM
• Evaluation Metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score.
• Results: o XGBoost: MAE = 1.10, RMSE = 2.35, R² = 0.79 o LSTM: MAE = 1.04, RMSE = 2.07, R² = 0.81
## Next Steps

1.	Model Tuning:
o	Experiment with additional advanced NLP models (e.g., RoBERTa, GPT) for sentiment analysis.
o	Refine hyperparameter tuning using Bayesian Optimization.
2.	Integration of Real-time Data:
o	Incorporate real-time tweet streaming for sentiment analysis and predictive updates.
3.	Expansion of Dataset:
o	Analyze additional stocks and broader market indices for generalization.
o	Include macroeconomic indicators and news sentiment for deeper insights.
4.	Dashboard Development:
o	Develop a Streamlit or Flask-based dashboard for real-time visualization of sentiment trends and stock predictions.
