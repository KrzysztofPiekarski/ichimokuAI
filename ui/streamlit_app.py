import streamlit as st
import pandas as pd
from services.model import IchimokuModel
import traceback

def load_data():
    """
    Funkcja wczytująca dane z pliku CSV.
    """
    uploaded_file = st.file_uploader("Wybierz plik CSV", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Sprawdzenie czy dane nie są puste
            if df.empty:
                st.error("❌ Plik CSV jest pusty!")
                return None
            
            # Wyświetl podstawowe informacje o danych
            st.success(f"✅ Załadowano {len(df)} wierszy i {len(df.columns)} kolumn")
            
            # Sprawdź czy istnieją wymagane kolumny
            required_columns = ['Close']  # Dodaj inne wymagane kolumny jeśli potrzebne
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Brakuje wymaganych kolumn: {missing_columns}")
                st.info("Dostępne kolumny: " + ", ".join(df.columns.tolist()))
                return None
            
            # Podgląd danych
            st.write("📊 Podgląd danych:")
            st.dataframe(df.head(10))
            
            # Podstawowe statystyki
            with st.expander("📈 Podstawowe statystyki"):
                st.write(df.describe())
            
            return df
            
        except Exception as e:
            st.error(f"❌ Błąd podczas wczytywania pliku: {str(e)}")
            return None
    
    return None

def train_and_predict_model(df):
    """
    Funkcja trenująca model i wykonująca prognozy.
    """
    try:
        # Sprawdź czy dane są odpowiednie do trenowania
        if len(df) < 52:  # Ichimoku potrzebuje co najmniej 52 okresów
            st.error("❌ Zbyt mało danych do analizy Ichimoku (potrzeba co najmniej 26 wierszy)")
            return None
        
        # Inicjalizacja modelu
        ichimoku_model = IchimokuModel()
        
        # Progress bar dla trenowania
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Trenowanie modelu...")
        progress_bar.progress(33)
        
        # Trenowanie modelu
        ichimoku_model.train_model(df)
        progress_bar.progress(66)
        
        status_text.text("🔄 Przewidywanie sygnału...")
        
        # Przewidywanie sygnału na podstawie ostatnich danych
        signal = ichimoku_model.predict_signal(df.tail(1))
        progress_bar.progress(100)
        
        status_text.text("✅ Analiza zakończona!")
        
        return signal
        
    except KeyError as e:
        st.error(f"❌ Błąd: Brakuje wymaganej kolumny - {str(e)}")
        return None
    except ValueError as e:
        st.error(f"❌ Błąd wartości: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Nieoczekiwany błąd podczas analizy: {str(e)}")
        st.error("Szczegóły błędu:")
        st.code(traceback.format_exc())
        return None

def display_signal_result(signal):
    """
    Funkcja wyświetlająca wynik przewidywania w czytelny sposób.
    """
    if signal is None:
        return
    
    # Obsługa różnych formatów sygnału
    if isinstance(signal, (list, tuple)) and len(signal) > 0:
        signal_value = signal[0]
    else:
        signal_value = signal
    
    # Kolorowe wyświetlanie sygnału
    st.markdown("### 🎯 Wynik Analizy")
    
    if signal_value == "BUY" or signal_value == 1:
        st.success(f"📈 **SYGNAŁ KUPNA**: {signal_value}")
    elif signal_value == "SELL" or signal_value == -1:
        st.error(f"📉 **SYGNAŁ SPRZEDAŻY**: {signal_value}")
    else:
        st.info(f"➡️ **BRAK SYGNAŁU / NEUTRALNY**: {signal_value}")

def main():
    # Konfiguracja strony
    st.set_page_config(
        page_title="Model Ichimoku AI",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Model Ichimoku z AI - Predykcje i Sygnały")
    st.markdown("---")
    
    # Instrukcje dla użytkownika
    with st.expander("ℹ️ Instrukcje użytkowania"):
        st.markdown("""
        1. **Wgraj plik CSV** z danymi historycznymi (musi zawierać kolumnę 'Close')
        2. **Sprawdź podgląd** załadowanych danych
        3. **Kliknij 'Trenuj Model'** aby przeprowadzić analizę Ichimoku
        4. **Zobacz wyniki** przewidywania sygnału
        
        **Wymagania dla pliku CSV:**
        - Co najmniej 26 wierszy danych (dla analizy Ichimoku)
        - Kolumna 'Close' z cenami zamknięcia
        """)
    
    # Wczytaj dane
    df = load_data()

    if df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # Możliwość trenowania modelu
            if st.button("🚀 Trenuj Model i Przewiduj Sygnał", type="primary"):
                with st.spinner("Przeprowadzam analizę..."):
                    signal = train_and_predict_model(df)
                    display_signal_result(signal)
        
        with col2:
            # Dodatkowe opcje wyświetlania
            if st.checkbox("📊 Pokaż wykres cen zamknięcia"):
                if 'Close' in df.columns:
                    st.line_chart(df['Close'])
                else:
                    st.error("Brak kolumny 'Close' do wyświetlenia wykresu")
        
        # Dodatkowe analizy
        if st.checkbox("🔍 Pokaż dodatkowe analizy"):
            if 'Close' in df.columns:
                st.subheader("📊 Analiza statystyczna cen")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Cena min", f"{df['Close'].min():.4f}")
                with col2:
                    st.metric("Cena max", f"{df['Close'].max():.4f}")
                with col3:
                    st.metric("Cena średnia", f"{df['Close'].mean():.4f}")

if __name__ == "__main__":
    main()