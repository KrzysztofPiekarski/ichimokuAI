import numpy as np

class IchimokuSignalGenerator:
    def __init__(self):
        pass

    def generate_signals(self, df):
        """
        Generuje sygnały na podstawie Ichimoku.
        """
        df['ichimoku_bull_signal'] = np.where((df['Close'] > df['senkou_span_a']) & (df['Close'] > df['senkou_span_b']), 1, 0)
        df['ichimoku_bear_signal'] = np.where((df['Close'] < df['senkou_span_a']) & (df['Close'] < df['senkou_span_b']), 1, 0)

        return df
