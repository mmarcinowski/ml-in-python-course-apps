import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, OrdinalEncoder, PolynomialFeatures,
    KBinsDiscretizer
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

st.set_page_config(page_title="ML w Pythonie CDV", layout="wide")
st.title("🧪 Data preprocessing w Pythonie: Sandbox")

@st.cache_data
def get_data():
    df = sns.load_dataset("penguins")
    for col in df.columns:
        if col != 'species':
            df.loc[df.sample(frac=0.1, random_state=42).index, col] = np.nan
    return df

df_raw = get_data()

st.sidebar.header("1. Czyszczenie i imputacja")
impute_num = st.sidebar.selectbox("Braki w liczbach", ["mean", "median", "most_frequent"])
clip_outliers = st.sidebar.checkbox("Przycinanie outlierów (1% - 99%)")

st.sidebar.header("2. Transformacje nieliniowe")
apply_log = st.sidebar.checkbox("Log transform (log1p)")
n_bins = st.sidebar.slider("Dyskretyzacja (liczba koszyków)", 0, 10, 0)

st.sidebar.header("3. Skalowanie i kodowanie")
scaler_type = st.sidebar.selectbox("Skalowanie liczb", ["brak", "standard", "min-max", "robust"])
enc_type = st.sidebar.selectbox("Kodowanie tekstu", ["ordinal", "one-hot"])

st.sidebar.header("4. Inżynieria i selekcja")
poly_deg = st.sidebar.slider("Stopień wielomianów", 1, 3, 1)
var_thresh = st.sidebar.slider("Próg wariancji (selekcja)", 0.0, 1.0, 0.0, 0.01)

df_proc = df_raw.copy()

num_cols = df_proc.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_proc.select_dtypes(include=['object']).columns.tolist()

imputer_n = SimpleImputer(strategy=impute_num)
df_proc[num_cols] = imputer_n.fit_transform(df_proc[num_cols])

imputer_c = SimpleImputer(strategy="most_frequent")
df_proc[cat_cols] = imputer_c.fit_transform(df_proc[cat_cols])

if clip_outliers:
    for col in num_cols:
        df_proc[col] = df_proc[col].clip(df_proc[col].quantile(0.01), df_proc[col].quantile(0.99))

if apply_log:
    df_proc[num_cols] = np.log1p(df_proc[num_cols].clip(lower=0))

if n_bins > 1:
    kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform', subsample=None)
    df_proc[num_cols] = kbd.fit_transform(df_proc[num_cols])

if scaler_type != "brak":
    scalers = {"standard": StandardScaler(), "min-max": MinMaxScaler(), "robust": RobustScaler()}
    df_proc[num_cols] = scalers[scaler_type].fit_transform(df_proc[num_cols])

if enc_type == "one-hot":
    ohe = OneHotEncoder(sparse_output=False)
    encoded = ohe.fit_transform(df_proc[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cat_cols))
    df_final = pd.concat([df_proc[num_cols].reset_index(drop=True), encoded_df], axis=1)
else:
    oe = OrdinalEncoder()
    df_proc[cat_cols] = oe.fit_transform(df_proc[cat_cols])
    df_final = df_proc

if poly_deg > 1:
    poly = PolynomialFeatures(degree=poly_deg, include_bias=False)
    poly_data = poly.fit_transform(df_final)
    df_final = pd.DataFrame(poly_data, columns=poly.get_feature_names_out())

if var_thresh > 0:
    selector = VarianceThreshold(threshold=var_thresh)
    try:
        df_final = pd.DataFrame(selector.fit_transform(df_final), columns=selector.get_feature_names_out())
    except ValueError:
        st.error("Próg wariancji zbyt wysoki - usunięto wszystkie cechy!")

tab1, tab2, tab3 = st.tabs(["📋 Porównanie danych", "📊 Rozkłady", "🔗 Korelacje"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.write("### Przed (surowe)")
        st.dataframe(df_raw.head(10))
        st.write(f"Braki: {df_raw.isna().sum().sum()}")
    with c2:
        st.write("### Po (przetworzone)")
        st.dataframe(df_final.head(10))
        st.write(f"Liczba cech: {df_final.shape[1]}")

with tab2:
    col_plot = st.selectbox("Wybierz kolumnę do analizy", num_cols)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df_raw[col_plot], kde=True, ax=ax[0], color="skyblue")
    ax[0].set_title("Oryginalny rozkład")
    sns.histplot(df_final[col_plot] if col_plot in df_final.columns else df_final.iloc[:,0], kde=True, ax=ax[1], color="salmon")
    ax[1].set_title("Rozkład po zmianach")
    st.pyplot(fig)

with tab3:
    if df_final.shape[1] < 60:
        fig_corr, ax_corr = plt.subplots(figsize=(10, 7))
        sns.heatmap(df_final.corr(), cmap="RdYlGn", center=0, ax=ax_corr)
        st.pyplot(fig_corr)
    else:
        st.warning("Zbyt wiele cech do wyświetlenia mapy korelacji")