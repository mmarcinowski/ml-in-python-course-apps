import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="ML w Pythonie CDV", layout="wide")
st.title("📈 Regresja w Pythonie: Sandbox")

@st.cache_data
def load_reg_data(name):
    if name == "Diamonds (ceny)":
        return sns.load_dataset("diamonds").sample(600, random_state=42)
    elif name == "Tips (napiwki)":
        return sns.load_dataset("tips")
    elif name == "MPG (spalanie aut)":
        return sns.load_dataset("mpg").dropna()
    else:
        return sns.load_dataset("geyser")

st.sidebar.header("Wybór danych")
dataset_name = st.sidebar.selectbox("Zbiór danych", ["Diamonds (ceny)", "Tips (napiwki)", "MPG (spalanie aut)"])

df = load_reg_data(dataset_name)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
target_col = st.sidebar.selectbox("Cel (target)", num_cols, index=len(num_cols) - 1)
feature_col = st.sidebar.selectbox("Cecha (feature)", [c for c in num_cols if c != target_col])

st.sidebar.header("Model i parametry")
algo = st.sidebar.radio("Algorytm", ["Regresja", "Ridge (L2)", "Lasso (L1)", "Drzewo decyzyjne", "kNN"])

if algo in ["Regresja", "Ridge (L2)", "Lasso (L1)"]:
    degree = st.sidebar.slider("Stopień wielomianu", 1, 10, 1)
    if algo != "Regresja":
        alpha = st.sidebar.slider("Siła regularyzacji (alpha)", 0.001, 20.0, 1.0, format="%.3f")

    if algo == "Regresja":
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    elif algo == "Ridge (L2)":
        model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha))
    else:
        model = make_pipeline(PolynomialFeatures(degree), Lasso(alpha=alpha, max_iter=10000))
elif algo == "Drzewo decyzyjne":
    max_d = st.sidebar.slider("Max depth", 1, 20, 3)
    model = DecisionTreeRegressor(max_depth=max_d)
else:
    k = st.sidebar.slider("Sąsiedzi (k)", 1, 50, 5)
    model = KNeighborsRegressor(n_neighbors=k)

use_scaler = st.sidebar.checkbox("Użyj standaryzacji StandardScaler")

X = df[[feature_col]].values
y = df[target_col].values

if use_scaler:
    scaler = StandardScaler()
    X_plot = scaler.fit_transform(X)
else:
    X_plot = X

model.fit(X_plot, y)
y_pred = model.predict(X_plot)

col1, col2 = st.columns([1, 1])

with col1:
    st.write("### 📊 Dopasowanie modelu")
    fig_fit, ax_fit = plt.subplots(figsize=(8, 6))

    sns.scatterplot(x=X_plot.flatten(), y=y, alpha=0.5, label="Dane rzeczywiste", ax=ax_fit)

    x_range = np.linspace(X_plot.min(), X_plot.max(), 500).reshape(-1, 1)
    y_range = model.predict(x_range)
    ax_fit.plot(x_range, y_range, color='red', lw=3, label="Przewidywanie")

    ax_fit.set_xlabel(feature_col)
    ax_fit.set_ylabel(target_col)
    ax_fit.legend()
    st.pyplot(fig_fit)



with col2:
    if algo in ["Regresja", "Ridge (L2)", "Lasso (L1)"]:
        st.write("### Wartości wag")
        coefs = model.named_steps[algo.split()[0].lower() if "Regresja" not in algo else "linearregression"].coef_
        fig_w, ax_w = plt.subplots()
        ax_w.bar(range(len(coefs)), coefs)
        ax_w.set_ylabel("Wartość wagi")
        ax_w.set_xlabel("Nr współczynnika")
        st.pyplot(fig_w)
        st.caption("Ridge zmniejsza wagi, Lasso sprowadza je do zera.")

st.write("### Wykres reszt (diagnostyka)")
fig_r, ax_r = plt.subplots(figsize=(12, 3))
sns.scatterplot(x=y_pred, y=y-y_pred, alpha=0.4, ax=ax_r)
ax_r.axhline(0, color='red', linestyle='--')
st.pyplot(fig_r)

st.sidebar.markdown("---")
#st.sidebar.write("### Metryki")
#st.sidebar.metric("R² Score", f"{r2_score(y, y_pred):.2%}")
#st.sidebar.metric("MAE", f"{mean_absolute_error(y, y_pred):.2f}")

st.sidebar.markdown(":violet-badge[:material/star: Metryki]")
results_table = pd.DataFrame(
    {
        "Metryka": ["R² Score", "MAE"],
        "Wartość": [f"{r2_score(y, y_pred):.2%}",f"{mean_absolute_error(y, y_pred):.2f}"]

    })
    #index=[str(i + 1) for i in range(len(probable))])
st.sidebar.table(results_table, border="horizontal")
