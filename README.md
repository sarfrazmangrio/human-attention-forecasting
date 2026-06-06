# Human Attention Forecasting

**Sarfraz Hussain — Sarfraz Hussain**  
Generative AI and Strategic Forecasting — Final Project

## What this is

A forecasting system that tracks where human attention is moving across major digital platforms and content categories, and predicts where it will be six months from now.

Data is collected from Google Trends, Wikipedia page views, and Reddit community engagement. Four forecasting models are built and compared on historical held-out data, then retrained on the full dataset to generate genuine forward-looking predictions.

## Models

- ARIMA — statistical baseline with AIC-selected order
- Prophet — trend and seasonality model with real-world event regressors
- Multivariate LSTM — neural network trained on all 15 signals simultaneously
- Temporal Fusion Transformer — transformer architecture for time series

## Data sources

- Google Trends via pytrends — primary attention signal
- Wikipedia Wikimedia API — exploratory validation only
- Reddit Kaggle dataset — community engagement signal

## Scope

- 5 platforms: YouTube, TikTok, Instagram, LinkedIn, Reddit
- 10 content categories: Artificial Intelligence and Technology, Personal Finance, Mental Health and Self Improvement, Sports, Gaming, News and Politics, Entertainment and Media, Science and Education, Lifestyle and Advice, Culture Food and Travel
- 3 years of weekly historical data
- 26-week backtest evaluation
- 26-week true future forecast

## Live app

https://attention-forecasting.streamlit.app

## Output

The app shows the historical attention signal, the true six-month forward forecast, a model comparison table, and an AI-generated Attention Intelligence Brief for any platform or content category.
