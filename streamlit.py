import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="ML Laboratorium Pro", layout="wide")
st.title("🔬 ML Sandbox Pro: Eksperymentalne Pierwsze Kroki")
st.markdown("---")

# --- BOCZNY PANEL STEROWANIA ---
st.sidebar.header("1. Dane i Preprocessing")

dataset_name = st.sidebar.selectbox(
    "Wybierz zbiór danych",
    ["Iris", "Penguins", "Diamonds (próba)", "Titanic (wiek vs klasa)"]
)


# Ładowanie danych z obsługą błędów i czyszczeniem
@st.cache_data
def load_data(name):
    if name == "Iris":
        df = sns.load_dataset("iris")
    elif name == "Penguins":
        df = sns.load_dataset("penguins").dropna()
    elif name == "Diamonds (próba)":
        df = sns.load_dataset("diamonds").sample(500, random_state=42)
        # Zamiana ceny na kategorię (tania/droga) dla klasyfikacji
        df['price_cat'] = pd.qcut(df['price'], 2, labels=['Tani', 'Drogi'])
        df = df.drop(columns=['price'])
    else:  # Titanic
        df = sns.load_dataset("titanic")[['age', 'pclass', 'fare', 'survived']].dropna()
    return df


df = load_data(dataset_name)

# Sabotaż danych (Szum)
noise_level = st.sidebar.slider("Poziom szumu (mieszanie etykiet) [%]", 0, 50, 0)

# Wybór cech do modelu (tylko numeryczne)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
target_col = df.columns[-1]

x_axis = st.sidebar.selectbox("Oś X (Wizualizacja)", numeric_cols, index=0)
y_axis = st.sidebar.selectbox("Oś Y (Wizualizacja)", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

st.sidebar.header("2. Wybór Algorytmu")
algo = st.sidebar.radio("Model", ["Drzewo Decyzyjne", "KNN (Sąsiedzi)", "Regresja Logistyczna"])

# --- PRZYGOTOWANIE DANYCH ---
X = df[[x_axis, y_axis]].copy()
y = df[target_col].copy()

# Enkodowanie etykiet (tekst -> liczby)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Aplikowanie szumu
if noise_level > 0:
    n_samples = int(len(y_encoded) * (noise_level / 100))
    idx_to_flip = np.random.choice(len(y_encoded), n_samples, replace=False)
    y_encoded[idx_to_flip] = np.random.choice(y_encoded, n_samples)

# Skalowanie (opcjonalne, ale ważne dla KNN)
use_scaler = st.sidebar.checkbox("Użyj Standaryzacji (StandardScaler)")
if use_scaler:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
else:
    X_scaled = X.values

# --- PARAMETRY MODELI ---
if algo == "Drzewo Decyzyjne":
    max_d = st.sidebar.slider("Głębokość (Max Depth)", 1, 20, 3)
    model = DecisionTreeClassifier(max_depth=max_d)
elif algo == "KNN (Sąsiedzi)":
    k = st.sidebar.slider("Liczba sąsiadów (K)", 1, 50, 5)
    model = KNeighborsClassifier(n_neighbors=k)
else:
    c_val = st.sidebar.select_slider("Regularyzacja (C)", options=[0.001, 0.01, 0.1, 1, 10, 100])
    model = LogisticRegression(C=c_val)

# --- TRENOWANIE I WIZUALIZACJA ---
col1, col2 = st.columns([1, 2])

with col1:
    st.write("### Statystyki")
    model.fit(X_scaled, y_encoded)
    acc = model.score(X_scaled, y_encoded)

    st.metric("Dokładność (Accuracy)", f"{acc:.2%}")
    st.write("**Podgląd danych po zmianach:**")
    st.dataframe(df[[x_axis, y_axis, target_col]].head(10))

    st.info("""
    **Zadanie:** Spróbuj tak dobrać parametry, aby granice na wykresie obok były 'gładkie', 
    ale wciąż dobrze rozdzielały kolory. Unikaj 'wysp' wokół pojedynczych punktów!
    """)

with col2:
    st.write(f"### Granice decyzyjne: {algo}")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Rysowanie tła decyzyjnego
    disp = DecisionBoundaryDisplay.from_estimator(
        model,
        X_scaled,
        response_method="predict",
        cmap="RdYlBu",
        alpha=0.5,
        ax=ax
    )

    # Rysowanie punktów danych
    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_encoded, edgecolors='k', cmap="RdYlBu")
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    # Dodanie legendy
    handles, labels = scatter.legend_elements()
    ax.legend(handles, le.classes_, title="Klasy")

    st.pyplot(fig)

# --- STOPKA ---
st.markdown("---")
st.caption("Laboratoria: Uczenie Maszynowe w Pythonie. Prowadzący: Twój Tutor.")