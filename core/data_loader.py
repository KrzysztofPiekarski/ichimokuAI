import pandas as pd
import requests
import yfinance as yf
import time
import os
from datetime import datetime, timedelta
from forex_python.converter import CurrencyRates
from dotenv import load_dotenv
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ładowanie zmiennych środowiskowych
load_dotenv()

# 🔑 Pobierz klucz API z zmiennej środowiskowej
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")

# Konfiguracja domyślna
DEFAULT_PERIOD_DAYS = {
    "1min": 7,    # 1 tydzień dla minutowych
    "5min": 14,   # 2 tygodnie dla 5-minutowych
    "15min": 30,  # 1 miesiąc dla 15-minutowych
    "30min": 60,  # 2 miesiące dla 30-minutowych
    "1h": 90,     # 3 miesiące dla godzinowych
    "1d": 365     # 1 rok dla dziennych
}

def validate_symbol(symbol: str) -> bool:
    """Walidacja symbolu forex (np. EURUSD)"""
    if len(symbol) != 6:
        return False
    return symbol[:3].isalpha() and symbol[3:6].isalpha()

def download_from_yfinance(symbol: str, interval: str, period_days: int = None) -> pd.DataFrame:
    """
    Pobiera dane z Yahoo Finance z ulepszoną obsługą błędów i konfiguracją
    """
    if not validate_symbol(symbol):
        raise ValueError(f"❌ Nieprawidłowy symbol: {symbol}. Oczekiwany format: EURUSD")
    
    interval_map = {
        "1min": "1m", "5min": "5m", "15min": "15m", 
        "30min": "30m", "1h": "1h", "1d": "1d"
    }
    
    yf_interval = interval_map.get(interval)
    if not yf_interval:
        raise ValueError(f"❌ Nieobsługiwany interwał: {interval}")
    
    # Określ okres pobierania danych
    if period_days is None:
        period_days = DEFAULT_PERIOD_DAYS.get(interval, 60)
    
    # Yahoo Finance używa różnych formatów okresów
    if period_days <= 7:
        period = "7d"
    elif period_days <= 30:
        period = "1mo"
    elif period_days <= 90:
        period = "3mo"
    elif period_days <= 365:
        period = "1y"
    else:
        period = "2y"
    
    for attempt in range(3):
        try:
            logger.info(f"📥 Pobieranie z Yahoo Finance - {symbol}, interwał: {interval}, okres: {period} (próba {attempt + 1})")
            
            ticker = yf.Ticker(symbol + "=X")
            df = ticker.history(interval=yf_interval, period=period)

            if df.empty:
                raise ValueError("⚠️ Brak danych — pusty DataFrame.")

            # Normalizacja kolumn
            df = df.rename(columns={
                'Open': 'Open', 'High': 'High', 
                'Low': 'Low', 'Close': 'Close'
            })
            
            # Wybierz tylko potrzebne kolumny i usuń NaN
            df = df[["Open", "High", "Low", "Close"]].dropna().copy()
            
            logger.info(f"✅ Pobrano {len(df)} wierszy z Yahoo Finance")
            return df

        except Exception as e:
            logger.warning(f"⚠️ Próba {attempt + 1} - Błąd Yahoo Finance: {e}")
            if attempt < 2:  # Czekaj tylko jeśli to nie ostatnia próba
                time.sleep(2 ** attempt)  # Exponential backoff

    raise ValueError("❌ Nie udało się pobrać danych z Yahoo Finance po 3 próbach.")

def download_from_twelve_data(symbol: str, interval: str, outputsize: int = 100) -> pd.DataFrame:
    """
    Pobiera dane z Twelve Data z ulepszoną obsługą błędów
    """
    if not TWELVE_DATA_API_KEY:
        raise ValueError("❌ Brak klucza API dla Twelve Data. Ustaw zmienną TWELVE_DATA_API_KEY.")
    
    if not validate_symbol(symbol):
        raise ValueError(f"❌ Nieprawidłowy symbol: {symbol}")
    
    logger.info(f"📥 Pobieranie z Twelve Data - {symbol}, interwał: {interval}")
    
    base_url = "https://api.twelvedata.com/time_series"
    from_symbol = symbol[:3]
    to_symbol = symbol[3:6]

    params = {
        "symbol": f"{from_symbol}/{to_symbol}",
        "interval": interval,
        "outputsize": min(outputsize, 5000),  # Limit API
        "apikey": TWELVE_DATA_API_KEY,
        "format": "JSON"
    }

    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Sprawdź czy odpowiedź zawiera błąd
        if "code" in data and data["code"] != 200:
            raise ValueError(f"❌ Błąd API Twelve Data: {data.get('message', 'Nieznany błąd')}")
        
        if "values" not in data:
            raise ValueError(f"❌ Brak danych w odpowiedzi Twelve Data: {data}")

        df = pd.DataFrame(data["values"])
        
        if df.empty:
            raise ValueError("❌ Otrzymano pusty DataFrame z Twelve Data")
        
        # Konwersja i normalizacja danych
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)  # Sortuj chronologicznie

        # Konwersja kolumn na numeryczne
        numeric_columns = ["open", "high", "low", "close"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Normalizacja nazw kolumn
        df = df.rename(columns={
            "open": "Open", "high": "High", 
            "low": "Low", "close": "Close"
        })
        
        df = df[["Open", "High", "Low", "Close"]].dropna()
        
        logger.info(f"✅ Pobrano {len(df)} wierszy z Twelve Data")
        return df
        
    except requests.RequestException as e:
        raise ValueError(f"❌ Błąd połączenia z Twelve Data: {e}")
    except Exception as e:
        raise ValueError(f"❌ Błąd przetwarzania danych Twelve Data: {e}")

def download_from_forex_python(symbol: str, days: int = 30) -> pd.DataFrame:
    """
    Pobiera dane z forex-python (tylko ceny dzienne)
    """
    if not validate_symbol(symbol):
        raise ValueError(f"❌ Nieprawidłowy symbol: {symbol}")
    
    logger.info(f"📥 Pobieranie z forex-python - {symbol}, {days} dni")
    
    try:
        c = CurrencyRates()
        end = datetime.now()
        start = end - timedelta(days=days)

        dates = pd.date_range(start=start, end=end)
        rates = []
        from_currency = symbol[:3]
        to_currency = symbol[3:6]

        successful_requests = 0
        for date in dates:
            try:
                rate = c.get_rate(from_currency, to_currency, date)
                if rate:
                    rates.append({"datetime": date, "Close": rate})
                    successful_requests += 1
                    
            except Exception as e:
                logger.debug(f"⚠️ Błąd forex-python dla {date.date()}: {e}")
                continue
                
            # Przerwa żeby nie przeciążyć API
            time.sleep(0.1)

        if successful_requests == 0:
            raise ValueError("❌ Nie udało się pobrać żadnych danych")
            
        df = pd.DataFrame(rates).set_index("datetime")
        df.sort_index(inplace=True)
        
        # Dla forex-python brak świec OHLC, więc duplikujemy cenę zamknięcia
        df["Open"] = df["High"] = df["Low"] = df["Close"]
        
        logger.info(f"✅ Pobrano {len(df)} wierszy z forex-python")
        return df

    except Exception as e:
        raise ValueError(f"❌ Błąd forex-python: {e}")

def load_data(symbol: str, interval: str, period_days: int = None, preferred_source: str = None) -> pd.DataFrame:
    """
    Główna funkcja do ładowania danych z automatycznym fallback
    
    Args:
        symbol: Para walutowa (np. 'EURUSD')
        interval: Interwał czasowy ('1min', '5min', '15min', '30min', '1h', '1d')
        period_days: Liczba dni wstecz (opcjonalne)
        preferred_source: Preferowane źródło ('yfinance', 'twelvedata', 'forex_python')
    """
    if not validate_symbol(symbol):
        raise ValueError(f"❌ Nieprawidłowy symbol: {symbol}. Oczekiwany format: EURUSD")
    
    symbol = symbol.upper()
    sources = []
    
    # Określ kolejność źródeł na podstawie preferencji
    if preferred_source == "twelvedata":
        sources = ["twelvedata", "yfinance", "forex_python"]
    elif preferred_source == "forex_python":
        sources = ["forex_python", "yfinance", "twelvedata"]
    else:  # domyślnie yfinance
        sources = ["yfinance", "twelvedata", "forex_python"]
    
    errors = []
    
    for source in sources:
        try:
            if source == "yfinance":
                return download_from_yfinance(symbol, interval, period_days)
            elif source == "twelvedata":
                return download_from_twelve_data(symbol, interval)
            elif source == "forex_python":
                if interval != "1d":
                    logger.warning("⚠️ forex-python obsługuje tylko dane dzienne, pomijam...")
                    continue
                return download_from_forex_python(symbol, period_days or 30)
                
        except Exception as e:
            error_msg = f"❌ Błąd {source}: {e}"
            logger.warning(error_msg)
            errors.append(error_msg)
            continue
    
    # Jeśli wszystkie źródła zawiodły
    error_summary = "\n".join(errors)
    raise ValueError(f"❗ Nie udało się pobrać danych z żadnego źródła:\n{error_summary}")

# Funkcja pomocnicza do testowania
def test_data_sources(symbol: str = "EURUSD", interval: str = "1h"):
    """Funkcja do testowania wszystkich źródeł danych"""
    print(f"🧪 Testowanie źródeł danych dla {symbol}, interwał: {interval}")
    
    sources = [
        ("Yahoo Finance", lambda: download_from_yfinance(symbol, interval)),
        ("Twelve Data", lambda: download_from_twelve_data(symbol, interval)),
        ("Forex Python", lambda: download_from_forex_python(symbol) if interval == "1d" else None)
    ]
    
    for source_name, source_func in sources:
        try:
            if source_func is None:
                print(f"⏭️ {source_name}: Pominięto (nie obsługuje interwału {interval})")
                continue
                
            df = source_func()
            print(f"✅ {source_name}: OK - {len(df)} wierszy")
            print(f"   Okres: {df.index.min()} do {df.index.max()}")
            
        except Exception as e:
            print(f"❌ {source_name}: Błąd - {e}")

if __name__ == "__main__":
    # Test podstawowy
    test_data_sources("EURUSD", "1h")