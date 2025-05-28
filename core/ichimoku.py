import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

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