import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class IchimokuFeatureExtractor:
    """
    Ekstraktor cech Ichimoku do analizy technicznej.
    
    Oblicza wszystkie komponenty systemu Ichimoku Kinko Hyo:
    - Tenkan-sen (linia konwersji)
    - Kijun-sen (linia bazowa)  
    - Senkou Span A i B (chmura)
    - Chikou Span (linia opóźniona)
    """
    
    # Standardowe okresy Ichimoku
    DEFAULT_PERIODS = {
        'tenkan': 9,
        'kijun': 26,
        'senkou_b': 52,
        'displacement': 26
    }
    
    def __init__(self, tenkan_period: int = 9, kijun_period: int = 26, 
                 senkou_b_period: int = 52, displacement: int = 26):
        """
        Inicjalizuje ekstraktor z konfigurowalnymi okresami.
        
        Args:
            tenkan_period: Okres dla Tenkan-sen (domyślnie 9)
            kijun_period: Okres dla Kijun-sen (domyślnie 26)
            senkou_b_period: Okres dla Senkou Span B (domyślnie 52)
            displacement: Przesunięcie dla chmury (domyślnie 26)
        """
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement
        
        # Minimalna ilość danych potrzebna dla wszystkich wskaźników
        self.min_data_length = max(senkou_b_period, displacement * 2)

    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Waliduje i przygotowuje DataFrame do obliczeń.
        
        Args:
            df: DataFrame wejściowy
            
        Returns:
            Zwalidowany i oczyszczony DataFrame
            
        Raises:
            ValueError: Gdy brak wymaganych kolumn lub dane są nieprawidłowe
        """
        if df.empty:
            raise ValueError("❌ DataFrame jest pusty")
            
        # Wymagane kolumny
        required_cols = ['High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"❌ Brak wymaganych kolumn: {missing_cols}")

        # Stwórz kopię i przekonwertuj na numeryczne
        df_clean = df.copy()
        for col in required_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        # Usuń wiersze z NaN w podstawowych kolumnach
        initial_len = len(df_clean)
        df_clean = df_clean.dropna(subset=required_cols)
        
        if len(df_clean) != initial_len:
            logger.warning(f"⚠️ Usunięto {initial_len - len(df_clean)} wierszy z powodu brakujących danych")
            
        if df_clean.empty:
            raise ValueError("❌ Brak prawidłowych danych po oczyszczeniu")
            
        # Reset indeksu po oczyszczeniu
        df_clean = df_clean.reset_index(drop=True)
            
        return df_clean

    def _check_data_sufficiency(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Sprawdza czy jest wystarczająco danych dla każdego wskaźnika.
        
        Args:
            df: DataFrame do sprawdzenia
            
        Returns:
            Słownik z informacją o wystarczalności danych dla każdego wskaźnika
        """
        data_len = len(df)
        sufficiency = {
            'tenkan_sen': data_len >= self.tenkan_period,
            'kijun_sen': data_len >= self.kijun_period,
            'senkou_span_a': data_len >= max(self.tenkan_period, self.kijun_period) + self.displacement,
            'senkou_span_b': data_len >= self.senkou_b_period + self.displacement,
            'chikou_span': data_len >= self.displacement,
            'full_ichimoku': data_len >= self.min_data_length
        }
        
        return sufficiency

    def _calculate_midpoint(self, high_series: pd.Series, low_series: pd.Series, 
                          window: int) -> pd.Series:
        """
        Oblicza punkt środkowy (średnią z najwyższej i najniższej wartości w oknie).
        
        Args:
            high_series: Seria z wartościami wysokimi
            low_series: Seria z wartościami niskimi
            window: Rozmiar okna
            
        Returns:
            Seria z punktami środkowymi
        """
        # Sprawdź czy serie mają wystarczająco danych
        if len(high_series) < window or len(low_series) < window:
            logger.warning(f"⚠️ Za mało danych dla okna {window} (dostępne: {len(high_series)})")
            return pd.Series([np.nan] * len(high_series), index=high_series.index)
        
        # Używamy min_periods=window aby uniknąć częściowych obliczeń
        high_max = high_series.rolling(window=window, min_periods=window).max()
        low_min = low_series.rolling(window=window, min_periods=window).min()
        
        return (high_max + low_min) / 2

    def calculate_ichimoku(self, df: pd.DataFrame, validate_data: bool = True) -> pd.DataFrame:
        """
        Oblicza wskaźniki Ichimoku z kontrolą jakości danych.
        
        Args:
            df: DataFrame z danymi OHLC
            validate_data: Czy przeprowadzić walidację danych (domyślnie True)
            
        Returns:
            DataFrame z dodanymi wskaźnikami Ichimoku
            
        Raises:
            ValueError: Gdy dane są nieprawidłowe
        """
        if validate_data:
            df_clean = self._validate_dataframe(df)
        else:
            df_clean = df.copy()
            df_clean = df_clean.reset_index(drop=True)

        # Sprawdź wystarczalność danych
        sufficiency = self._check_data_sufficiency(df_clean)
        
        if not sufficiency['full_ichimoku']:
            logger.warning(
                f"⚠️ Dane mają {len(df_clean)} wierszy, ale zalecane minimum to "
                f"{self.min_data_length} dla pełnego systemu Ichimoku"
            )

        try:
            # Obliczenia Ichimoku z dodatkową walidacją
            logger.info(f"📊 Obliczanie Ichimoku dla {len(df_clean)} wierszy danych")
            
            # Tenkan-sen (9-okresowa linia konwersji)
            if sufficiency['tenkan_sen']:
                df_clean['tenkan_sen'] = self._calculate_midpoint(
                    df_clean['High'], df_clean['Low'], self.tenkan_period
                )
                logger.debug(f"✅ Obliczono tenkan_sen: {df_clean['tenkan_sen'].notna().sum()} wartości")
            else:
                df_clean['tenkan_sen'] = np.nan
                logger.warning(f"⚠️ Za mało danych dla tenkan_sen (potrzeba {self.tenkan_period})")
            
            # Kijun-sen (26-okresowa linia bazowa)
            if sufficiency['kijun_sen']:
                df_clean['kijun_sen'] = self._calculate_midpoint(
                    df_clean['High'], df_clean['Low'], self.kijun_period
                )
                logger.debug(f"✅ Obliczono kijun_sen: {df_clean['kijun_sen'].notna().sum()} wartości")
            else:
                df_clean['kijun_sen'] = np.nan
                logger.warning(f"⚠️ Za mało danych dla kijun_sen (potrzeba {self.kijun_period})")
            
            # Senkou Span A (średnia z Tenkan i Kijun przesunięta do przodu)
            if sufficiency['senkou_span_a']:
                senkou_a_base = (df_clean['tenkan_sen'] + df_clean['kijun_sen']) / 2
                # POPRAWKA: Senkou Span A powinno być przesunięte do PRZODU (dodatnie przesunięcie w przyszłość)
                df_clean['senkou_span_a'] = senkou_a_base.shift(self.displacement)
                logger.debug(f"✅ Obliczono senkou_span_a: {df_clean['senkou_span_a'].notna().sum()} wartości")
            else:
                df_clean['senkou_span_a'] = np.nan
                logger.warning(f"⚠️ Za mało danych dla senkou_span_a")
            
            # Senkou Span B (52-okresowy midpoint przesunięty do przodu)
            if sufficiency['senkou_span_b']:
                senkou_b_base = self._calculate_midpoint(
                    df_clean['High'], df_clean['Low'], self.senkou_b_period
                )
                # POPRAWKA: Senkou Span B również przesunięte do przodu
                df_clean['senkou_span_b'] = senkou_b_base.shift(self.displacement)
                logger.debug(f"✅ Obliczono senkou_span_b: {df_clean['senkou_span_b'].notna().sum()} wartości")
            else:
                df_clean['senkou_span_b'] = np.nan
                logger.warning(f"⚠️ Za mało danych dla senkou_span_b")
            
            # Chikou Span (cena zamknięcia przesunięta do tyłu)
            if sufficiency['chikou_span']:
                # POPRAWKA: Chikou span to przesunięcie do TYŁU (ujemne przesunięcie)
                df_clean['chikou_span'] = df_clean['Close'].shift(-self.displacement)
                logger.debug(f"✅ Obliczono chikou_span: {df_clean['chikou_span'].notna().sum()} wartości")
            else:
                df_clean['chikou_span'] = np.nan
                logger.warning(f"⚠️ Za mało danych dla chikou_span")

            # Dodatkowe wskaźniki pomocnicze
            df_clean['cloud_color'] = np.where(
                df_clean['senkou_span_a'] > df_clean['senkou_span_b'], 
                'green', 'red'
            )
            
            # Pozycja ceny względem chmury
            cloud_top = np.maximum(df_clean['senkou_span_a'], df_clean['senkou_span_b'])
            cloud_bottom = np.minimum(df_clean['senkou_span_a'], df_clean['senkou_span_b'])
            
            df_clean['price_vs_cloud'] = np.where(
                df_clean['Close'] > cloud_top, 'above',
                np.where(df_clean['Close'] < cloud_bottom, 'below', 'inside')
            )

        except Exception as e:
            logger.error(f"❌ Błąd podczas obliczania Ichimoku: {str(e)}")
            raise ValueError(f"Błąd obliczania Ichimoku: {str(e)}")

        # Końcowa walidacja
        ichimoku_cols = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']
        for col in ichimoku_cols:
            if col in df_clean.columns:
                valid_count = df_clean[col].notna().sum()
                total_count = len(df_clean)
                if valid_count == 0:
                    logger.warning(f"⚠️ Kolumna {col} zawiera tylko NaN – sprawdź długość danych")
                elif valid_count < total_count * 0.3:
                    logger.warning(f"⚠️ Kolumna {col} ma mniej niż 30% prawidłowych wartości ({valid_count}/{total_count})")
                else:
                    logger.info(f"✅ Kolumna {col}: {valid_count}/{total_count} prawidłowych wartości")

        logger.info(f"✅ Zakończono obliczanie Ichimoku")
        return df_clean

    def get_ichimoku_summary(self, df: pd.DataFrame) -> Dict:
        """
        Zwraca podsumowanie wskaźników Ichimoku.
        
        Args:
            df: DataFrame z obliczonymi wskaźnikami Ichimoku
            
        Returns:
            Słownik z podsumowaniem wskaźników
        """
        ichimoku_cols = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']
        
        summary = {
            'total_rows': len(df),
            'periods_used': {
                'tenkan': self.tenkan_period,
                'kijun': self.kijun_period,
                'senkou_b': self.senkou_b_period,
                'displacement': self.displacement
            },
            'data_completeness': {},
            'data_range': {}
        }
        
        # Dodaj informacje o zakresie danych
        if 'Close' in df.columns:
            close_series = df['Close'].dropna()
            if not close_series.empty:
                summary['data_range']['price'] = {
                    'min': float(close_series.min()),
                    'max': float(close_series.max()),
                    'last': float(close_series.iloc[-1]) if len(close_series) > 0 else None
                }
        
        for col in ichimoku_cols:
            if col in df.columns:
                valid_count = df[col].notna().sum()
                series_clean = df[col].dropna()
                
                summary['data_completeness'][col] = {
                    'valid_values': valid_count,
                    'completion_rate': valid_count / len(df) if len(df) > 0 else 0,
                    'first_valid_index': df[col].first_valid_index(),
                    'last_valid_index': df[col].last_valid_index()
                }
                
                # Dodaj statystyki dla niepustych serii
                if not series_clean.empty:
                    summary['data_range'][col] = {
                        'min': float(series_clean.min()),
                        'max': float(series_clean.max()),
                        'last': float(series_clean.iloc[-1]) if len(series_clean) > 0 else None
                    }
        
        return summary

    def validate_ichimoku_signals(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Waliduje czy DataFrame zawiera prawidłowe sygnały Ichimoku.
        
        Args:
            df: DataFrame do walidacji
            
        Returns:
            Tuple (czy_prawidłowe, lista_problemów)
        """
        problems = []
        ichimoku_cols = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']
        
        # Sprawdź czy kolumny istnieją
        missing_cols = [col for col in ichimoku_cols if col not in df.columns]
        if missing_cols:
            problems.append(f"Brak kolumn: {missing_cols}")
        
        # Sprawdź czy są jakiekolwiek dane
        for col in ichimoku_cols:
            if col in df.columns:
                valid_count = df[col].notna().sum()
                if valid_count == 0:
                    problems.append(f"Kolumna {col} nie ma prawidłowych wartości")
                elif valid_count < len(df) * 0.1:  # Mniej niż 10% danych
                    problems.append(f"Kolumna {col} ma zbyt mało prawidłowych wartości ({valid_count}/{len(df)})")
        
        # Sprawdź logiczność wartości
        if 'tenkan_sen' in df.columns and 'kijun_sen' in df.columns:
            # Sprawdź czy wartości są w rozsądnym zakresie względem cen
            if 'Close' in df.columns:
                close_range = df['Close'].max() - df['Close'].min()
                for col in ['tenkan_sen', 'kijun_sen']:
                    col_series = df[col].dropna()
                    if not col_series.empty:
                        col_range = col_series.max() - col_series.min()
                        if col_range > close_range * 3:  # Wskaźnik ma 3x większy zakres niż cena
                            problems.append(f"Podejrzanie duży zakres wartości w {col}")
        
        return len(problems) == 0, problems

    def get_latest_signals(self, df: pd.DataFrame) -> Dict:
        """
        Zwraca najnowsze sygnały Ichimoku.
        
        Args:
            df: DataFrame z obliczonymi wskaźnikami
            
        Returns:
            Słownik z aktualnymi sygnałami
        """
        if len(df) < 2:
            return {'error': 'Za mało danych dla analizy sygnałów'}
        
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        signals = {
            'timestamp': latest.name if hasattr(latest, 'name') else len(df) - 1,
            'price': latest.get('Close'),
            'signals': []
        }
        
        # Tenkan-Kijun cross
        if (pd.notna(latest.get('tenkan_sen')) and pd.notna(latest.get('kijun_sen')) and 
            pd.notna(previous.get('tenkan_sen')) and pd.notna(previous.get('kijun_sen'))):
            
            if (previous['tenkan_sen'] <= previous['kijun_sen'] and 
                latest['tenkan_sen'] > latest['kijun_sen']):
                signals['signals'].append({
                    'type': 'tenkan_kijun_cross',
                    'direction': 'bullish',
                    'description': 'Tenkan Sen przecięła Kijun Sen w górę'
                })
            elif (previous['tenkan_sen'] >= previous['kijun_sen'] and 
                  latest['tenkan_sen'] < latest['kijun_sen']):
                signals['signals'].append({
                    'type': 'tenkan_kijun_cross',
                    'direction': 'bearish',
                    'description': 'Tenkan Sen przecięła Kijun Sen w dół'
                })
        
        # Price vs Cloud
        if latest.get('price_vs_cloud'):
            signals['cloud_position'] = latest['price_vs_cloud']
            
        # Cloud color
        if latest.get('cloud_color'):
            signals['cloud_trend'] = latest['cloud_color']
        
        return signals