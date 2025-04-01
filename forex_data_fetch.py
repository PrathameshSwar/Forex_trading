import requests
import pandas as pd
API_KEY = 'PASTE YOUR KEY'  # Grom https://www.alphavantage.co/support/#api-key
BASE_URL = 'https://www.alphavantage.co/query'

params = {
    'function': 'FX_DAILY',
    'from_symbol': 'EUR',
    'to_symbol': 'USD',
    'apikey': API_KEY,
    'outputsize': 'full'
}

response = requests.get(BASE_URL, params=params)
data = response.json()

# Convert to DataFrame
df = pd.DataFrame.from_dict(data['Time Series FX (Daily)'], orient='index')
df = df.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close'})
df.index = pd.to_datetime(df.index)
df = df.astype(float).sort_index()

# Save to CSV
df.to_csv('EURUSD_historical_data.csv')
print("Data saved to EURUSD_historical_data.csv")
