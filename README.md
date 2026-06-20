# Human Attention Forecasting in the Age of AI Content

[![Live app](https://img.shields.io/badge/Live%20app-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://human-attention-forecasting.streamlit.app)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sarfrazmangrio/human-attention-forecasting/blob/main/notebook/Human_Attention_Forecasting.ipynb)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)

> AI is generating content at unprecedented scale while human attention stays finite. Most platform and topic decisions are made on where attention was months ago, not where it is heading. This project forecasts where attention is going.

A forecasting system that tracks attention across major digital platforms and content categories and predicts where it will move over the next six months (26 weeks). It compares four forecasting models, selects the best one per signal, and turns the result into a short strategy brief.

**Live app:** https://human-attention-forecasting.streamlit.app

## Data sources

Three independent public signals, each normalized to a 0 to 100 attention score:

- **Google Trends** via pytrends: the primary weekly search-interest signal
- **Wikipedia page views** via the Wikimedia API: a supporting public-interest signal
- **Reddit engagement** from a Kaggle dataset: community engagement across 50 major subreddits, mapped to the 10 content categories

## Models

- **ARIMA**: statistical baseline with an AIC-selected order
- **Prophet**: trend and seasonality with event regressors for the AI content shift, platform policy changes, and the creator economy shift
- **Multivariate LSTM**: a single network over all signals, forecasting recursively
- **Temporal Fusion Transformer**: a transformer for multivariate time series

The best model for each signal is chosen by backtest error, so different platforms and categories can use different models.

## Method

- **Backtest**: train on earlier data, predict a held-out 26 weeks, and score with MAE, RMSE, and MAPE
- **True future forecast**: retrain on all available data and predict the next 26 weeks
- **Attention Intelligence Brief**: a rule-based strategy brief, with an optional Groq LLM pass that only improves the wording and never changes a number

## Scope

- 5 platforms: YouTube, TikTok, Instagram, LinkedIn, Reddit
- 10 content categories: Artificial Intelligence and Technology, Personal Finance, Mental Health and Self Improvement, Sports, Gaming, News and Politics, Entertainment and Media, Science and Education, Lifestyle and Advice, Culture Food and Travel
- Three years of weekly data

## Repository structure

```
human-attention-forecasting/
  app.py                              Streamlit dashboard, reads the precomputed outputs
  notebook/
    Human_Attention_Forecasting.ipynb full pipeline: data, models, backtest, forecast
  *.csv                               exported signals and model outputs the app reads
  requirements.txt                    app runtime dependencies
  requirements-notebook.txt           full modeling stack for the notebook
  README.md
  LICENSE
```

## Running it

The app uses the precomputed CSV outputs, so it starts instantly and does not retrain anything:

```bash
pip install -r requirements.txt
streamlit run app.py
```

To reproduce the full pipeline, open the notebook in Colab with the badge above, or run it locally:

```bash
pip install -r requirements-notebook.txt
```

The notebook fetches live Google Trends and Wikipedia data and expects the Reddit Kaggle dataset as a zip in the working directory. If a live source is unavailable, it falls back to a synthetic signal so the pipeline still runs end to end.

## Limitations

All three sources are proxy signals. They do not measure total human attention directly, so the forecasts are decision support rather than guarantees.

## License

Released under the MIT License. See [LICENSE](LICENSE).
