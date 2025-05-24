import pandas as pd
from services.model import IchimokuModel
import streamlit as st
from ui.streamlit_app import main
from core.data_loader import load_data


def main():
    # Załaduj dane (dostosuj ścieżkę do pliku)
    df = load_data("EURUSD", "1h")  # lub inny symbol i interwał jakiego potrzebujesz

    # Sprawdzanie kolumn przed przekazaniem do modelu
    print(df.columns)  # Opcjonalnie, sprawdź, które kolumny zostały wczytane
    ichimoku_model = IchimokuModel()
    try:
        ichimoku_model.train_model(df)
    except KeyError as e:
        print(f"Błąd: {e}")
        return

    # Sprawdź dostępne kolumny w danych
    if df is None or df.empty:
        st.error("❌ Nie udało się załadować danych.")
        st.stop()

    st.write(f"Dostępne kolumny: {df.columns.tolist()}")

    if 'Close' not in df.columns:
        st.error("❌ Brak kolumny 'Close' w danych.")
        st.stop()

    # Stwórz model Ichimoku
    ichimoku_model = IchimokuModel()

    # Trenuj model
    ichimoku_model.train_model(df)

    # Przewiduj sygnał na nowych danych
    signal = ichimoku_model.predict_signal(df.tail(1))  # Ostatni wiersz danych
    print(f"Przewidywany sygnał: {signal}")

    try:
        df = load_data("EURUSD", "1h")
    except ValueError as e:
        st.error(f"Błąd podczas ładowania danych: {e}")
        st.stop()

# Jedno wywołanie main()
if __name__ == "__main__":
    main()
