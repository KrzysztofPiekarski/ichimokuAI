import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from core.ichimoku import IchimokuFeatureExtractor
from core.features import FeatureAdder
from core.signals import IchimokuSignalGenerator

class IchimokuModel:
    def __init__(self):
        self.feature_extractor = IchimokuFeatureExtractor()
        self.feature_adder = FeatureAdder()
        self.signal_generator = IchimokuSignalGenerator()
        self.model = LogisticRegression()  # Możesz zmienić model ML

    def generate_features(self, df):
        """
        Generuje cechy wejściowe na podstawie danych Ichimoku.
        """
        # Sprawdzamy, czy 'Close' znajduje się w danych
        if 'Close' not in df.columns:
            raise ValueError("❗ Kolumna 'Close' nie została znaleziona w danych.")

        # Przekształcamy dane Ichimoku
        df = self.feature_extractor.calculate_ichimoku(df)
        df = self.feature_adder.add_rsi(df)
        df = self.feature_adder.add_macd(df)

        # Wybieramy cechy, usuwając puste wiersze
        features = df[['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span', 'rsi', 'macd']].dropna()
        return features

    def generate_signals(self, df):
        """
        Generuje sygnały Ichimoku na podstawie przetworzonych danych.
        """
        required_columns = ['Close', 'senkou_span_a', 'senkou_span_b']
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"Brak wymaganej kolumny: {col}")

        # Generowanie sygnałów
        df['ichimoku_bull_signal'] = np.where((df['Close'] > df['senkou_span_a']) & (df['Close'] > df['senkou_span_b']), 1, 0)
        df['ichimoku_bear_signal'] = np.where((df['Close'] < df['senkou_span_a']) & (df['Close'] < df['senkou_span_b']), -1, 0)

        return df

    def train_model(self, df):
        """
        Trenuje model na danych Ichimoku.
        """
        # Przekształcanie kolumn na typ numeryczny
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['senkou_span_a'] = pd.to_numeric(df['senkou_span_a'], errors='coerce')
        df['senkou_span_b'] = pd.to_numeric(df['senkou_span_b'], errors='coerce')

        # Usuwanie wierszy z NaN
        df = df.dropna(subset=['Close', 'senkou_span_a', 'senkou_span_b'])

        # Generowanie cech
        features = self.generate_features(df)
        df = self.generate_signals(df)

        # Przygotowanie danych wejściowych
        X = features
        y = df['ichimoku_bull_signal']  # Możesz wybrać inny sygnał

        # Podział danych na zbiór treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Trenowanie modelu
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        # Obliczanie dokładności modelu
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")

    def predict_signal(self, df):
        """
        Przewiduje sygnał na podstawie cech Ichimoku.
        """
        features = self.generate_features(df)
        signal = self.model.predict(features)
        return signal

