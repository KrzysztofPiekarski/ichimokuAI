import pandas as pd
import logging

logger = logging.getLogger(__name__)

class IchimokuFeatureExtractor:
    def __init__(self):
        pass

    def calculate_ichimoku(self, df):
        """
        Oblicza wskaźniki Ichimoku z kontrolą jakości danych i ostrzeżeniem o brakach.
        """
        df = df.copy()  # nie modyfikujemy oryginału

        # Ostrzeżenie jeśli danych jest za mało
        if len(df) < 100:
            logger.warning("⚠️ Dane mają mniej niż 100 wierszy – mogą nie wystarczyć do obliczenia wszystkich wskaźników Ichimoku.")

        # Wymagane kolumny
        required_cols = ['High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"❌ Brak wymaganych kolumn: {[col for col in required_cols if col not in df.columns]}")

        # Upewnij się, że dane są numeryczne
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Usuń wiersze z NaN w podstawowych kolumnach
        df = df.dropna(subset=required_cols)

        # Obliczenia Ichimoku
        df['tenkan_sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
        df['kijun_sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        df['senkou_span_b'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
        df['chikou_span'] = df['Close'].shift(-26)

        # Sprawdzenie i ostrzeżenie, jeśli kolumny mają tylko NaN
        ichimoku_cols = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']
        for col in ichimoku_cols:
            if df[col].isna().all():
                logger.warning(f"⚠️ Kolumna {col} zawiera tylko NaN – dane mogą być zbyt krótkie.")

        return df
