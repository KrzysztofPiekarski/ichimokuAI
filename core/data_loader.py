
import pandas as pd
import requests
import yfinance as yf
import time
from datetime import datetime, timedelta
from forex_python.converter import CurrencyRates
from dotenv import load_dotenv
load_dotenv()

# 🔑 Podaj swój klucz API do Twelve Data
TWELVE_DATA_API_KEY = "YOUR_API_KEY"  # <-- wstaw swój klucz

def download_from_yfinance(symbol: str, interval: str) -> pd.DataFrame:
    df = None
    interval_map = {
        "1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m", "1h": "60m", "1d": "1d"
    }
    yf_interval = interval_map.get(interval, "60m")

    for attempt in range(3):
        try:
            print(f"📥 Próbuję pobrać dane z Yahoo Finance... (próba {attempt + 1})")
            df = yf.download(symbol + "=X", interval=yf_interval, period="60d", progress=False)

            if df.empty:
                raise ValueError("⚠️ Brak danych — pusty DataFrame.")

            # Obsługuje MultiIndex w kolumnach
            df.columns = df.columns.droplevel()

            df = df[["Open", "High", "Low", "Close"]].dropna().copy()
            return df

        except Exception as e:
            print(f"⚠️ Próba {attempt + 1} - Błąd Yahoo Finance: {e}")
            time.sleep(1)

    raise ValueError("❌ Nie udało się pobrać danych z Yahoo Finance po 3 próbach.")

def download_from_twelve_data(symbol: str, interval: str) -> pd.DataFrame:
    print("📥 Próbuję pobrać dane z Twelve Data...")
    base_url = "https://api.twelvedata.com/time_series"
    from_symbol = symbol[:3]
    to_symbol = symbol[3:6]

    params = {
        "symbol": f"{from_symbol}/{to_symbol}",
        "interval": interval,
        "outputsize": 100,
        "apikey": TWELVE_DATA_API_KEY,
        "format": "JSON"
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if "values" not in data:
        raise ValueError(f"❌ Brak danych w Twelve Data. Odpowiedź: {data}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)

    df["Open"] = pd.to_numeric(df["open"], errors="coerce")
    df["High"] = pd.to_numeric(df["high"], errors="coerce")
    df["Low"] = pd.to_numeric(df["low"], errors="coerce")
    df["Close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna()

def download_from_forex_python(symbol: str) -> pd.DataFrame:
    print("📥 Próbuję pobrać dane z forex-python...")
    try:
        c = CurrencyRates()
        end = datetime.now()
        start = end - timedelta(days=30)

        dates = pd.date_range(start=start, end=end)
        rates = []
        from_currency = symbol[:3]
        to_currency = symbol[3:6]

        for date in dates:
            try:
                rate = c.get_rate(from_currency, to_currency, date)
                rates.append({"datetime": date, "Close": rate})
            except Exception as e:
                print(f"⚠️ Błąd forex-python dla {date.date()}: {e}")

        df = pd.DataFrame(rates).set_index("datetime")
        df["Open"] = df["High"] = df["Low"] = df["Close"]  # brak świec, powiel dane
        return df

    except Exception as e:
        print(f"❌ Błąd forex-python: {e}")
        raise ValueError("❗ Nie udało się pobrać danych z forex-python.")

def load_data(symbol: str, interval: str) -> pd.DataFrame:
    # Próbuj pobrać dane z Yahoo Finance
    try:
        return download_from_yfinance(symbol, interval)
    except Exception as e:
        print(f"⚠️ Nie udało się pobrać z Yahoo Finance: {e}")

    # Jeśli z Yahoo Finance się nie udało, próbuj z Twelve Data
    try:
        return download_from_twelve_data(symbol, interval)
    except Exception as e:
        print(f"❌ Błąd Twelve Data: {e}")

    # Jeśli z Twelve Data się nie udało, próbuj z forex-python
    try:
        return download_from_forex_python(symbol)
    except Exception as e:
        print(f"❌ Błąd forex-python: {e}")

    # Jeśli wszystkie próby zawiodą
    raise ValueError("❗ Nie udało się pobrać danych z żadnego źródła.")
