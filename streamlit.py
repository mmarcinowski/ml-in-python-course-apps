import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder

st.set_page_config(page_title="ML w Pythonie CDV", layout="wide")
st.title("🔬 Klasyfikacja w Pythonie: Sandbox")

@st.cache_data
def load_data(name):
    if name == "Iris":
        return sns.load_dataset("iris")
    elif name == "Penguins":
        return sns.load_dataset("penguins").dropna()
    elif name == "Diamonds":
        df = sns.load_dataset("diamonds").sample(500, random_state=42)
        df['price_cat'] = pd.qcut(df['price'], 2, labels=['tani', 'drogi'])
        return df.drop(columns=['price'])
    else: # Titanic
        return sns.load_dataset("titanic")[['age', 'pclass', 'fare', 'survived']].dropna()

st.sidebar.header("Dane i preprocessing")
dataset_name = st.sidebar.selectbox("Wybierz zbiór", ["Iris", "Penguins", "Diamonds", "Titanic"])
df = load_data(dataset_name)

all_cols = df.columns.tolist()
target_col = st.sidebar.selectbox("Etykieta (target)", all_cols, index=len(all_cols)-1)

available_features = [c for c in all_cols if c != target_col]
x_axis = st.sidebar.selectbox("Oś X", available_features, index=0)
remaining_features = [c for c in available_features if c != x_axis]
y_axis = st.sidebar.selectbox("Oś Y", remaining_features, index=0)

noise_level = st.sidebar.slider("Poziom szumu (mieszanie etykiet) [%]", 0, 50, 0)
use_scaler = st.sidebar.checkbox("Użyj standaryzacji StandardScaler", value=False)

st.sidebar.header("Model i parametry")
algo = st.sidebar.radio("Algorytm", ["Drzewo decyzyjne", "kNN", "Regresja logistyczna", "Naive Bayes"])

if algo == "Drzewo decyzyjne":
    max_d = st.sidebar.slider("Max depth", 1, 15, 3)
    model = DecisionTreeClassifier(max_depth=max_d)
elif algo == "kNN":
    k_val = st.sidebar.slider("Liczba sąsiadów (K)", 1, 50, 5)
    model = KNeighborsClassifier(n_neighbors=k_val)
elif algo == "Naive Bayes":
    model = GaussianNB()
else:
    c_val = st.sidebar.select_slider("C (regularyzacja)", options=[0.01, 0.1, 1, 10, 100], value=1.0)
    model = LogisticRegression(C=c_val)

X_raw = df[[x_axis, y_axis]].copy()
y = df[target_col].copy()

for col in X_raw.columns:
    if X_raw[col].dtype == 'object' or X_raw[col].dtype.name == 'category':
        X_raw[col] = OrdinalEncoder().fit_transform(X_raw[[col]])

le = LabelEncoder()
y_encoded = le.fit_transform(y)

if noise_level > 0:
    n_samples = int(len(y_encoded) * (noise_level / 100))
    if n_samples > 0:
        idx_to_flip = np.random.choice(len(y_encoded), n_samples, replace=False)
        y_encoded[idx_to_flip] = np.random.choice(np.unique(y_encoded), n_samples)

X_final = StandardScaler().fit_transform(X_raw.values) if use_scaler else X_raw.values

model.fit(X_final, y_encoded)
acc = model.score(X_final, y_encoded)

col1, col2 = st.columns([2, 1])

with col1:
    st.write(f"### Granice decyzyjne: {algo}")
    fig, ax = plt.subplots(figsize=(10, 6))

    DecisionBoundaryDisplay.from_estimator(
        model, X_final, response_method="predict",
        cmap="RdYlBu", alpha=0.5, ax=ax
    )

    scatter = ax.scatter(X_final[:, 0], X_final[:, 1], c=y_encoded, edgecolors='k', cmap="RdYlBu")
    ax.set_xlabel(x_axis + (" (skalowane)" if use_scaler else ""))
    ax.set_ylabel(y_axis + (" (skalowane)" if use_scaler else ""))

    handles, labels = scatter.legend_elements()
    ax.legend(handles, le.classes_, title="Klasy")
    st.pyplot(fig)

with col2:
    st.write("### Wyniki")
    st.metric("Trafność / dokładność (accuracy)", f"{acc:.2%}")

    st.write("**Podgląd danych wejściowych:**")
    st.dataframe(df[[x_axis, y_axis, target_col]].head(10))

    if algo == "Naive Bayes":
        st.write("### Gaussian Naive Bayes zakłada, że cechy mają rozkład normalny i są od siebie niezależne.")

if algo == "Drzewo decyzyjne":
    st.markdown("---")
    st.write("### 🌳 Wizualizacja struktury drzewa")
    fig_tree, ax_tree = plt.subplots(figsize=(15, 8))
    plot_tree(
        model, feature_names=[x_axis, y_axis],
        class_names=le.classes_.astype(str),
        filled=True, rounded=True, ax=ax_tree
    )
    st.pyplot(fig_tree)