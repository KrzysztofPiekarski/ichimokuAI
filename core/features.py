
import talib
import pandas as pd

class FeatureAdder:
    def add_rsi(self, df, period=14):
        if 'Close' not in df.columns:
            raise ValueError("❗ Brak kolumny 'Close' w danych.")
        
        # Debugowanie: Sprawdzamy kształt i typ 'Close' przed obróbką
        print(f"Debug: Kształt df['Close']: {df['Close'].shape}, Typ: {type(df['Close'])}")
        
        # Upewniamy się, że 'Close' jest jedną kolumną i przekształcamy w wektor
        close_values = df['Close'].values.flatten()  # Konwertujemy do jednowymiarowej tablicy
        
        # Debugowanie: Sprawdzamy kształt po konwersji
        print(f"Debug: Kształt close_values po flatten: {close_values.shape}")
        
        if close_values.shape[0] < period:
            raise ValueError("❗ Za mało danych do obliczenia RSI.")
        
        # Sprawdzamy, czy dane są jednowymiarowe
        if close_values.ndim != 1:
            raise ValueError(f"❗ Dane wejściowe mają nieprawidłowy wymiar. Powinny być jednowymiarowe, ale mają {close_values.ndim} wymiary.")
        
        # Obliczamy RSI na jednowymiarowej tablicy
        df['rsi'] = talib.RSI(close_values, timeperiod=period)
        return df
    
    def add_macd(self, df, fastperiod=12, slowperiod=26, signalperiod=9):
        if 'Close' not in df.columns:
            raise ValueError("❗ Brak kolumny 'Close' w danych.")
        
        close_values = df['Close'].values.flatten()  # Upewniamy się, że dane są jednowymiarowe
        macd, macdsignal, macdhist = talib.MACD(close_values, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        
        df['macd'] = macd
        df['macd_signal'] = macdsignal
        df['macd_hist'] = macdhist
        return df
