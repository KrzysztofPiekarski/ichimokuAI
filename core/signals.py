import numpy as np

class IchimokuSignalGenerator:
    def __init__(self):
        pass

    def generate_signals(self, df):
        required_cols = ['Close', 'senkou_span_a', 'senkou_span_b']
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"❌ Brak wymaganej kolumny: {col}")

        df = df.copy()
        df['ichimoku_bull_signal'] = np.where(
            (df['Close'] > df['senkou_span_a']) & (df['Close'] > df['senkou_span_b']), 1, 0)
        df['ichimoku_bear_signal'] = np.where(
            (df['Close'] < df['senkou_span_a']) & (df['Close'] < df['senkou_span_b']), 1, 0)

        df['ichimoku_signal'] = 0
        df.loc[df['ichimoku_bull_signal'] == 1, 'ichimoku_signal'] = 1
        df.loc[df['ichimoku_bear_signal'] == 1, 'ichimoku_signal'] = -1

        return df
