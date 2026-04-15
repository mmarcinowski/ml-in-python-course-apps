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

st.set_page_config(page_title="ML w Pythonie CDV", layout="wide")
st.title("🔬 ML w Pythonie: Sandbox")

st.sidebar.header("Dane i preprocessing")

dataset_name = st.sidebar.selectbox(
    "Wybierz zbiór danych",
    ["Iris", "Penguins", "Diamonds (próba)", "Titanic (wiek vs klasa)"]
)


@st.cache_data
def load_data(name):
    if name == "Iris":
        df = sns.load_dataset("iris")
    elif name == "Penguins":
        df = sns.load_dataset("penguins").dropna()
    elif name == "Diamonds (próba)":
        df = sns.load_dataset("diamonds").sample(500, random_state=42)
        df['price_cat'] = pd.qcut(df['price'], 2, labels=['Tani', 'Drogi'])
        df = df.drop(columns=['price'])
    else:
        df = sns.load_dataset("titanic")[['age', 'pclass', 'fare', 'survived']].dropna()
    return df

df = load_data(dataset_name)

noise_level = st.sidebar.slider("Poziom szumu (mieszanie etykiet) [%]", 0, 50, 0)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
target_col = df.columns[-1]

x_axis = st.sidebar.selectbox("Oś X (wizualizacja)", numeric_cols, index=0)

remaining_cols = [col for col in numeric_cols if col != x_axis]
if remaining_cols:
    y_axis = st.sidebar.selectbox("Oś Y (Wizualizacja)", remaining_cols, index=0)
else:
    y_axis = remaining_cols[0]
    st.sidebar.warning("Zbiór ma tylko jedną cechę numeryczną!")

st.sidebar.header("Wybór algorytmu")
algo = st.sidebar.radio("Model", ["Drzewo decyzyjne", "Algorytm k najbliższych sąsiadów", "Regresja logistyczna"])
st.sidebar.markdown("---")
use_scaler = st.sidebar.checkbox("Włącz skalowanie danych (Standard Scaler)", value=False)

X_raw = df[[x_axis, y_axis]].copy()
y = df[target_col].copy()

if x_axis == y_axis:
    X_raw.columns = [f"{x_axis}_x", f"{y_axis}_y"]

from sklearn.preprocessing import OrdinalEncoder
for col in X_raw.columns:
    if X_raw[col].dtype == 'object' or X_raw[col].dtype.name == 'category':
        enc = OrdinalEncoder()
        X_raw[col] = enc.fit_transform(X_raw[[col]])

le = LabelEncoder()
y_encoded = le.fit_transform(y)

if noise_level > 0:
    n_samples = int(len(y_encoded) * (noise_level / 100))
    if n_samples > 0:
        idx_to_flip = np.random.choice(len(y_encoded), n_samples, replace=False)
        y_encoded[idx_to_flip] = np.random.choice(np.unique(y_encoded), n_samples)

if use_scaler:
    scaler = StandardScaler()
    X_final = scaler.fit_transform(X_raw.values)
else:
    X_final = X_raw.values

if algo == "Drzewo decyzyjne":
    max_d = st.sidebar.slider("Głębokość (Max Depth)", 1, 20, 3)
    model = DecisionTreeClassifier(max_depth=max_d)
elif algo == "Algorytm k najbliższych sąsiadów":
    k = st.sidebar.slider("Liczba sąsiadów (K)", 1, 50, 5)
    model = KNeighborsClassifier(n_neighbors=k)
else:
    c_val = st.sidebar.select_slider("Regularyzacja (C)", options=[0.001, 0.01, 0.1, 1, 10, 100])
    model = LogisticRegression(C=c_val)

col1, col2 = st.columns([1, 2])

with col1:
    st.write("### Statystyki")
    model.fit(X_final, y_encoded)
    acc = model.score(X_final, y_encoded)

    st.metric("Trafność (accuracy)", f"{acc:.2%}")
    st.write("**Podgląd danych po zmianach:**")
    st.dataframe(df[[x_axis, y_axis, target_col]].head(10))

with col2:
    st.write(f"### Granice decyzyjne: {algo}")
    fig, ax = plt.subplots(figsize=(10, 6))

    disp = DecisionBoundaryDisplay.from_estimator(
        model,
        X_final,
        response_method="predict",
        cmap="RdYlBu",
        alpha=0.5,
        ax=ax
    )

    scatter = ax.scatter(X_final[:, 0], X_final[:, 1], c=y_encoded, edgecolors='k', cmap="RdYlBu")
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    handles, labels = scatter.legend_elements()
    ax.legend(handles, le.classes_, title="Klasy")

    st.pyplot(fig)

st.markdown("---")
st.caption("Laboratoria: Uczenie maszynowe w Pythonie. Prowadzący: MMarcinowski.")