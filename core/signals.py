import numpy as np
import pandas as pd
from typing import Union


class IchimokuSignalGenerator:
    """
    Generator sygnałów handlowych na podstawie wskaźnika Ichimoku.
    
    Generuje sygnały kupna i sprzedaży na podstawie pozycji ceny względem
    chmury Ichimoku (senkou_span_a i senkou_span_b).
    """
    
    def __init__(self):
        """Inicjalizuje generator sygnałów Ichimoku."""
        self.required_columns = ['Close', 'senkou_span_a', 'senkou_span_b']

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Waliduje czy DataFrame zawiera wymagane kolumny.
        
        Args:
            df: DataFrame do walidacji
            
        Raises:
            ValueError: Gdy brak wymaganych kolumn lub DataFrame jest pusty
        """
        if df.empty:
            raise ValueError("❌ DataFrame jest pusty")
            
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            # POPRAWKA: Bardziej informacyjny komunikat błędu
            raise ValueError(
                f"❌ Brak wymaganych kolumn: {missing_cols}\n"
                f"💡 Najpierw oblicz wskaźniki Ichimoku używając IchimokuFeatureExtractor.calculate_ichimoku()"
            )

    def can_generate_signals(self, df: pd.DataFrame) -> bool:
        """
        Sprawdza czy można generować sygnały dla tego DataFrame.
        
        Args:
            df: DataFrame do sprawdzenia
            
        Returns:
            True jeśli można generować sygnały, False w przeciwnym razie
        """
        if df.empty:
            return False
            
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        return len(missing_cols) == 0
    
    def generate_signals(self, df: pd.DataFrame, inplace: bool = False, 
                        safe_mode: bool = False) -> pd.DataFrame:
        """
        Generuje sygnały handlowe na podstawie wskaźnika Ichimoku.
        
        Sygnały:
        - 1: Sygnał kupna (cena powyżej obu linii chmury)
        - -1: Sygnał sprzedaży (cena poniżej obu linii chmury)
        - 0: Brak sygnału (cena wewnątrz chmury)
        
        Args:
            df: DataFrame z danymi cenowymi i wskaźnikami Ichimoku
            inplace: Czy modyfikować oryginalny DataFrame (domyślnie False)
            safe_mode: Jeśli True, zwraca DataFrame z pustymi sygnałami zamiast błędu
            
        Returns:
            DataFrame z dodanymi sygnałami
            
        Raises:
            ValueError: Gdy DataFrame nie zawiera wymaganych kolumn (tylko gdy safe_mode=False)
        """
        required_cols = ['senkou_span_a', 'senkou_span_b', 'tenkan_sen', 'kijun_sen', 'chikou_span', 'Close']
    
        # POPRAWKA: Dodano tryb bezpieczny
        if safe_mode and not self.can_generate_signals(df):
            result_df = df if inplace else df.copy()
            result_df['ichimoku_signal'] = 0
            result_df['ichimoku_bull_signal'] = 0
            result_df['ichimoku_bear_signal'] = 0
            return result_df
        
        self._validate_dataframe(df)
        
        # Używaj oryginalnego DataFrame lub stwórz kopię
        result_df = df if inplace else df.copy()
        
        # Sprawdź czy cena jest powyżej lub poniżej chmury
        above_cloud = (result_df['Close'] > result_df['senkou_span_a']) & \
                     (result_df['Close'] > result_df['senkou_span_b'])
        below_cloud = (result_df['Close'] < result_df['senkou_span_a']) & \
                     (result_df['Close'] < result_df['senkou_span_b'])
        
        # Generuj sygnały w jednym kroku
        result_df['ichimoku_signal'] = np.where(
            above_cloud, 1,
            np.where(below_cloud, -1, 0)
        )
        
        # Opcjonalnie: dodaj pomocnicze kolumny boolean
        result_df['ichimoku_bull_signal'] = above_cloud.astype(int)
        result_df['ichimoku_bear_signal'] = below_cloud.astype(int)
        
        return result_df
    
    def get_current_signal(self, df: pd.DataFrame) -> Union[int, None]:
        """
        Zwraca najnowszy sygnał z DataFrame.
        
        Args:
            df: DataFrame z sygnałami
            
        Returns:
            Najnowszy sygnał lub None jeśli brak danych
        """
        if df.empty or 'ichimoku_signal' not in df.columns:
            return None
        return df['ichimoku_signal'].iloc[-1]
    
    def get_signal_summary(self, df: pd.DataFrame) -> dict:
        """
        Zwraca podsumowanie sygnałów.
        
        Args:
            df: DataFrame z sygnałami
            
        Returns:
            Słownik z podsumowaniem sygnałów
        """
        if df.empty or 'ichimoku_signal' not in df.columns:
            return {'total': 0, 'buy_signals': 0, 'sell_signals': 0, 'neutral': 0}
            
        signals = df['ichimoku_signal']
        return {
            'total': len(signals),
            'buy_signals': (signals == 1).sum(),
            'sell_signals': (signals == -1).sum(),
            'neutral': (signals == 0).sum(),
            'current_signal': self.get_current_signal(df)
        }