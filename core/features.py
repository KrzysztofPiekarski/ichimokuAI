import talib
import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class FeatureAdder:
    """
    Klasa do dodawania wskaźników technicznych do DataFrame z danymi finansowymi.
    
    Wykorzystuje bibliotekę TA-Lib do obliczania wskaźników takich jak RSI, MACD,
    średnie kroczące i inne popularne wskaźniki analizy technicznej.
    """
    
    def __init__(self, debug_mode: bool = False):
        """
        Inicjalizuje FeatureAdder.
        
        Args:
            debug_mode: Czy włączyć tryb debugowania (domyślnie False)
        """
        self.debug_mode = debug_mode
        self.required_columns = {
            'price_based': ['Close'],
            'ohlc_based': ['Open', 'High', 'Low', 'Close'],
            'volume_based': ['Close', 'Volume']
        }

    def _validate_and_prepare_data(self, df: pd.DataFrame, 
                                 required_cols: list, 
                                 min_periods: int) -> pd.DataFrame:
        """
        Waliduje i przygotowuje dane do obliczeń wskaźników.
        
        Args:
            df: DataFrame wejściowy
            required_cols: Lista wymaganych kolumn
            min_periods: Minimalna liczba okresów potrzebnych
            
        Returns:
            Zwalidowany DataFrame
            
        Raises:
            ValueError: Gdy dane są nieprawidłowe
        """
        if df.empty:
            raise ValueError("❗ DataFrame jest pusty")
            
        # Sprawdź wymagane kolumny
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"❗ Brak wymaganych kolumn: {missing_cols}")
        
        # Sprawdź minimalną liczbę wierszy
        if len(df) < min_periods:
            raise ValueError(f"❗ Za mało danych. Wymagane: {min_periods}, dostępne: {len(df)}")
        
        return df.copy()

    def _prepare_price_array(self, series: pd.Series, name: str = "price") -> np.ndarray:
        """
        Przygotowuje tablicę cenową dla TA-Lib.
        
        Args:
            series: Seria pandas z danymi cenowymi
            name: Nazwa serii dla debugowania
            
        Returns:
            Jednowymiarowa tablica numpy
            
        Raises:
            ValueError: Gdy dane są nieprawidłowe
        """
        if self.debug_mode:
            logger.info(f"Debug: Przygotowywanie {name} - kształt: {series.shape}, typ: {type(series)}")
        
        # Konwertuj do numerycznych i usuń NaN
        numeric_series = pd.to_numeric(series, errors='coerce')
        
        if numeric_series.isna().all():
            raise ValueError(f"❗ Wszystkie wartości w {name} są NaN po konwersji")
        
        # Konwertuj do tablicy i spłaszcz
        price_array = numeric_series.values.flatten()
        
        if self.debug_mode:
            logger.info(f"Debug: {name} po przygotowaniu - kształt: {price_array.shape}, wymiary: {price_array.ndim}")
        
        # Sprawdź wymiary
        if price_array.ndim != 1:
            raise ValueError(f"❗ {name} ma nieprawidłowy wymiar: {price_array.ndim}, powinien być 1")
        
        return price_array

    def add_rsi(self, df: pd.DataFrame, period: int = 14, 
                column: str = 'Close', new_column: Optional[str] = None) -> pd.DataFrame:
        """
        Dodaje wskaźnik RSI (Relative Strength Index) do DataFrame.
        
        Args:
            df: DataFrame z danymi
            period: Okres do obliczenia RSI (domyślnie 14)
            column: Kolumna do obliczenia RSI (domyślnie 'Close')
            new_column: Nazwa nowej kolumny (domyślnie 'rsi')
            
        Returns:
            DataFrame z dodanym RSI
            
        Raises:
            ValueError: Gdy dane są nieprawidłowe
        """
        if new_column is None:
            new_column = f'rsi_{period}' if period != 14 else 'rsi'
            
        df_validated = self._validate_and_prepare_data(df, [column], period)
        close_values = self._prepare_price_array(df_validated[column], column)
        
        try:
            rsi_values = talib.RSI(close_values, timeperiod=period)
            df_validated[new_column] = rsi_values
            
            if self.debug_mode:
                valid_rsi = pd.Series(rsi_values).notna().sum()
                logger.info(f"Debug: RSI obliczone - prawidłowych wartości: {valid_rsi}/{len(rsi_values)}")
                
        except Exception as e:
            raise ValueError(f"❗ Błąd podczas obliczania RSI: {str(e)}")
        
        return df_validated

    def add_macd(self, df: pd.DataFrame, fastperiod: int = 12, slowperiod: int = 26, 
                 signalperiod: int = 9, column: str = 'Close') -> pd.DataFrame:
        """
        Dodaje wskaźnik MACD do DataFrame.
        
        Args:
            df: DataFrame z danymi
            fastperiod: Okres szybkiej średniej (domyślnie 12)
            slowperiod: Okres wolnej średniej (domyślnie 26)
            signalperiod: Okres linii sygnałowej (domyślnie 9)
            column: Kolumna do obliczenia MACD (domyślnie 'Close')
            
        Returns:
            DataFrame z dodanymi wskaźnikami MACD
            
        Raises:
            ValueError: Gdy dane są nieprawidłowe
        """
        min_periods = max(slowperiod, fastperiod) + signalperiod
        df_validated = self._validate_and_prepare_data(df, [column], min_periods)
        close_values = self._prepare_price_array(df_validated[column], column)
        
        try:
            macd, macd_signal, macd_hist = talib.MACD(
                close_values, 
                fastperiod=fastperiod,
                slowperiod=slowperiod, 
                signalperiod=signalperiod
            )
            
            df_validated['macd'] = macd
            df_validated['macd_signal'] = macd_signal
            df_validated['macd_hist'] = macd_hist
            
            if self.debug_mode:
                valid_macd = pd.Series(macd).notna().sum()
                logger.info(f"Debug: MACD obliczone - prawidłowych wartości: {valid_macd}/{len(macd)}")
                
        except Exception as e:
            raise ValueError(f"❗ Błąd podczas obliczania MACD: {str(e)}")
        
        return df_validated

    def add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, 
                           std_dev: float = 2.0, column: str = 'Close') -> pd.DataFrame:
        """
        Dodaje wstęgi Bollingera do DataFrame.
        
        Args:
            df: DataFrame z danymi
            period: Okres średniej ruchomej (domyślnie 20)
            std_dev: Liczba odchyleń standardowych (domyślnie 2.0)
            column: Kolumna do obliczenia (domyślnie 'Close')
            
        Returns:
            DataFrame z dodanymi wstęgami Bollingera
        """
        df_validated = self._validate_and_prepare_data(df, [column], period)
        close_values = self._prepare_price_array(df_validated[column], column)
        
        try:
            upper, middle, lower = talib.BBANDS(
                close_values, 
                timeperiod=period,
                nbdevup=std_dev, 
                nbdevdn=std_dev
            )
            
            df_validated['bb_upper'] = upper
            df_validated['bb_middle'] = middle
            df_validated['bb_lower'] = lower
            df_validated['bb_width'] = (upper - lower) / middle * 100  # Szerokość wstęg w %
            
        except Exception as e:
            raise ValueError(f"❗ Błąd podczas obliczania wstęg Bollingera: {str(e)}")
        
        return df_validated

    def add_moving_averages(self, df: pd.DataFrame, periods: list = [20, 50], 
                           column: str = 'Close', ma_type: str = 'SMA') -> pd.DataFrame:
        """
        Dodaje średnie kroczące do DataFrame.
        
        Args:
            df: DataFrame z danymi
            periods: Lista okresów dla średnich (domyślnie [20, 50])
            column: Kolumna do obliczenia (domyślnie 'Close')
            ma_type: Typ średniej ('SMA' lub 'EMA')
            
        Returns:
            DataFrame z dodanymi średnimi kroczącymi
        """
        max_period = max(periods)
        df_validated = self._validate_and_prepare_data(df, [column], max_period)
        close_values = self._prepare_price_array(df_validated[column], column)
        
        for period in periods:
            try:
                if ma_type.upper() == 'SMA':
                    ma_values = talib.SMA(close_values, timeperiod=period)
                    col_name = f'sma_{period}'
                elif ma_type.upper() == 'EMA':
                    ma_values = talib.EMA(close_values, timeperiod=period)
                    col_name = f'ema_{period}'
                else:
                    raise ValueError(f"❗ Nieznany typ średniej: {ma_type}")
                
                df_validated[col_name] = ma_values
                
            except Exception as e:
                logger.warning(f"⚠️ Błąd przy obliczaniu {ma_type}_{period}: {str(e)}")
                continue
        
        return df_validated

    def add_stochastic(self, df: pd.DataFrame, k_period: int = 14, 
                      d_period: int = 3) -> pd.DataFrame:
        """
        Dodaje oscylator stochastyczny do DataFrame.
        
        Args:
            df: DataFrame z danymi OHLC
            k_period: Okres dla %K (domyślnie 14)
            d_period: Okres dla %D (domyślnie 3)
            
        Returns:
            DataFrame z dodanym oscylatorem stochastycznym
        """
        required_cols = ['High', 'Low', 'Close']
        min_periods = k_period + d_period
        df_validated = self._validate_and_prepare_data(df, required_cols, min_periods)
        
        try:
            high_values = self._prepare_price_array(df_validated['High'], 'High')
            low_values = self._prepare_price_array(df_validated['Low'], 'Low')
            close_values = self._prepare_price_array(df_validated['Close'], 'Close')
            
            slowk, slowd = talib.STOCH(
                high_values, low_values, close_values,
                fastk_period=k_period, slowk_period=d_period, slowd_period=d_period
            )
            
            df_validated['stoch_k'] = slowk
            df_validated['stoch_d'] = slowd
            
        except Exception as e:
            raise ValueError(f"❗ Błąd podczas obliczania oscylatora stochastycznego: {str(e)}")
        
        return df_validated

    def get_available_indicators(self) -> Dict[str, Dict[str, Any]]:
        """
        Zwraca listę dostępnych wskaźników z ich opisami.
        
        Returns:
            Słownik z dostępnymi wskaźnikami
        """
        return {
            'rsi': {
                'name': 'Relative Strength Index',
                'required_columns': ['Close'],
                'parameters': {'period': 14}
            },
            'macd': {
                'name': 'Moving Average Convergence Divergence',
                'required_columns': ['Close'],
                'parameters': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9}
            },
            'bollinger_bands': {
                'name': 'Bollinger Bands',
                'required_columns': ['Close'],
                'parameters': {'period': 20, 'std_dev': 2.0}
            },
            'moving_averages': {
                'name': 'Moving Averages (SMA/EMA)',
                'required_columns': ['Close'],
                'parameters': {'periods': [20, 50], 'ma_type': 'SMA'}
            },
            'stochastic': {
                'name': 'Stochastic Oscillator',
                'required_columns': ['High', 'Low', 'Close'],
                'parameters': {'k_period': 14, 'd_period': 3}
            }
        }