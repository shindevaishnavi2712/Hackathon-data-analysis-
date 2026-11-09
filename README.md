# Smart Agriculture – Hackathon Starter Project

## What is included
- `app.py`: Streamlit dashboard with Overview, Crop Recommendation, Market Analysis, and About pages.
- `data/crop_recommendation_demo.csv`: synthetic demo training data for a Random Forest crop classifier.
- `data/market_prices_demo.csv`: synthetic illustrative monthly market-price data for charts.
- `requirements.txt`: Python packages.

## Run on a computer
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Run: `pip install -r requirements.txt`
4. Run: `streamlit run app.py`
5. Open the local URL shown in the terminal.

## Run using a cloud notebook
Upload the ZIP to a notebook environment, unzip it, install the packages with:
`!pip install -r requirements.txt`
A Streamlit app needs a supported app host/tunnel to display as a web app; it does not render inside a normal notebook cell by itself.

## Important data note
Both included datasets are SYNTHETIC DEMO DATA. They are for building and demonstrating the prototype only. The displayed market prices are not live mandi prices, and crop predictions are not agricultural advice. Before a real hackathon field demo, replace them with verified, sourced data for the target district and season and validate predictions with an agricultural expert.

## Suggested hackathon pitch
"We built a farmer decision-support prototype that combines soil-based crop recommendation with market trend visualization. It demonstrates how machine learning and data analytics can help organize information for better farm planning. Our next step is connecting verified local soil and official mandi datasets."


## Bilingual interface
The Streamlit sidebar includes a language selector for English and Marathi (मराठी). Choose either language to switch the dashboard labels and explanations.
