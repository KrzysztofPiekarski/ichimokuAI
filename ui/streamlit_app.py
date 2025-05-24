import streamlit as st
import pandas as pd
from services.model import IchimokuModel

def load_data():
    """
    Funkcja wczytująca dane z pliku CSV.
    """
    uploaded_file = st.file_uploader("Wybierz plik CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Podgląd danych:")
        st.write(df.head())
        return df
    return None

def train_and_predict_model(df):
    """
    Funkcja trenująca model i wykonująca prognozy.
    """
    ichimoku_model = IchimokuModel()
    ichimoku_model.train_model(df)

    # Przewidywanie sygnału na podstawie ostatnich danych
    signal = ichimoku_model.predict_signal(df.tail(1))
    return signal

def main():
    st.title("Model Ichimoku z AI - Predykcje i Sygnały")

    # Wczytaj dane
    df = load_data()

    if df is not None:
        # Możliwość trenowania modelu
        if st.button("Trenuj Model i Przewiduj Sygnał"):
            signal = train_and_predict_model(df)
            st.write(f"Przewidywany sygnał: {signal[0]}")
        
        # Dodatkowe opcje wyświetlania
        if st.checkbox("Pokaż wykres danych"):
            st.line_chart(df['Close'])

if __name__ == "__main__":
    main()
