import pandas as pd
from services.model import IchimokuModel
import streamlit as st
from core.data_loader import load_data

def run_analysis():
    """Funkcja do analizy danych z modelem Ichimoku"""
    
    # Załaduj dane
    try:
        df = load_data("EURUSD", "1h")
    except ValueError as e:
        st.error(f"❌ Błąd podczas ładowania danych: {e}")
        return None
    
    # Sprawdź czy dane zostały załadowane
    if df is None or df.empty:
        st.error("❌ Nie udało się załadować danych.")
        return None
    
    # Wyświetl informacje o danych
    st.write(f"✅ Załadowano {len(df)} wierszy danych")
    st.write(f"Dostępne kolumny: {df.columns.tolist()}")
    
    # Sprawdź czy kolumna 'Close' istnieje
    if 'Close' not in df.columns:
        st.error("❌ Brak kolumny 'Close' w danych.")
        return None
    
    # Stwórz i wytrenuj model Ichimoku
    ichimoku_model = IchimokuModel()
    
    try:
        ichimoku_model.train_model(df)
        st.success("✅ Model został wytrenowany pomyślnie")
    except KeyError as e:
        st.error(f"❌ Błąd podczas trenowania modelu: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Nieoczekiwany błąd: {e}")
        return None
    
    # Przewiduj sygnał na ostatnich danych
    try:
        signal = ichimoku_model.predict_signal(df.tail(1))
        st.success(f"🎯 Przewidywany sygnał: {signal}")
        return signal
    except Exception as e:
        st.error(f"❌ Błąd podczas przewidywania: {e}")
        return None

def main():
    """Główna funkcja aplikacji Streamlit"""
    st.title("📈 Analiza Ichimoku - EURUSD")
    st.markdown("---")
    
    # Uruchom analizę
    if st.button("🚀 Uruchom analizę"):
        with st.spinner("Analizuję dane..."):
            signal = run_analysis()
            
        if signal:
            st.balloons()

if __name__ == "__main__":
    main()