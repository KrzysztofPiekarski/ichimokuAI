import pandas as pd
import numpy as np

class IchimokuFeatureExtractor:
    def __init__(self):
        pass

    def calculate_ichimoku(self, df):
        """
        Oblicza wskaźniki Ichimoku.
        """
        nine_period_high = df['High'].rolling(window=9).max()
        nine_period_low = df['Low'].rolling(window=9).min()

        df['tenkan_sen'] = (nine_period_high + nine_period_low) / 2

        period26_high = df['High'].rolling(window=26).max()
        period26_low = df['Low'].rolling(window=26).min()

        df['kijun_sen'] = (period26_high + period26_low) / 2

        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        df['senkou_span_b'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)

        df['chikou_span'] = df['Close'].shift(-26)

        return df
