import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

st.set_page_config(page_title="ML w Pythonie CDV", layout="wide")
st.title("🏗️ ML Pipeline: Sandbox")

st.header("1. Dane i podział zbioru")
dataset_name = st.selectbox("Wybierz dane", ["Iris", "Penguins", "Titanic"])


@st.cache_data
def get_data(name):
    if name == "Iris": return sns.load_dataset("iris")
    if name == "Penguins": return sns.load_dataset("penguins")
    return sns.load_dataset("titanic")[['survived', 'pclass', 'sex', 'age', 'fare']]


df_raw = get_data(dataset_name)
target_col = df_raw.columns[0] if dataset_name == "Titanic" else df_raw.columns[-1]
X = df_raw.drop(columns=[target_col])
y = df_raw[target_col]

test_size = st.slider("Wielkość zbioru testowego", 0.1, 0.5, 0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

st.markdown("---")

st.header("2. Budowa pipeline'u")

c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("Ułóż kolejność operacji:")

    steps_order = st.multiselect(
        "Wybierz i ułóż kroki (kolejność ma znaczenie):",
        ["Ewaluacja", "Imputacja", "Kodowanie", "Skalowanie", "Uczenie"]
    )

with c2:
    st.subheader("Konfiguracja klocków:")
    scale_method = st.selectbox("Skalowanie", ["StandardScaler", "MinMaxScaler", "RobustScaler", "Brak"])
    cat_method = st.selectbox("Kodowanie", ["OneHotEncoder", "OrdinalEncoder"])
    num_impute = st.selectbox("Imputacja liczb", ["mean", "median"])


def validate_pipeline(steps):
    if not steps:
        return "❌ Wybierz przynajmniej jeden krok", False
    if "Skalowanie" in steps and "Imputacja" in steps:
        if steps.index("Skalowanie") < steps.index("Imputacja"):
            return "⚠️ BŁĄD: Skalowanie przed imputacją! Skalery nie potrafią liczyć średniej z brakujących danych (NaN).", False
    if "Kodowanie" in steps and "Imputacja" in steps:
        if steps.index("Kodowanie") < steps.index("Imputacja"):
            return "⚠️ BŁĄD: Kodowanie przed imputacją! Braki w tekście mogą zepsuć enkoder.", False
    if "Uczenie" not in steps:
        return "❌ Naucz model", False
    if "Ewaluacja" not in steps:
        return "❌ Zewaluuj model", False
    if "Ewaluacja" in steps and "Uczenie" in steps:
        if steps.index("Ewaluacja") < steps.index("Uczenie"):
            return "⚠️ BŁĄD: Ewaluacja przed uczeniem?", False
    if "Ewaluacja" in steps and "Imputacja" in steps:
        if steps.index("Ewaluacja") < steps.index("Imputacja"):
            return "⚠️ BŁĄD: Preprocessing po ewaluacji?", False
    if "Ewaluacja" in steps and "Skalowanie" in steps:
        if steps.index("Ewaluacja") < steps.index("Skalowanie"):
            return "⚠️ BŁĄD: Preprocessing po ewaluacji?", False
    if "Ewaluacja" in steps and "Kodowanie" in steps:
        if steps.index("Ewaluacja") < steps.index("Kodowanie"):
            return "⚠️ BŁĄD: Preprocessing po ewaluacji?", False
    if "Uczenie" in steps and "Imputacja" in steps:
        if steps.index("Uczenie") < steps.index("Imputacja"):
            return "⚠️ BŁĄD: Preprocessing po uczeniu?", False
    if "Uczenie" in steps and "Skalowanie" in steps:
        if steps.index("Uczenie") < steps.index("Skalowanie"):
            return "⚠️ BŁĄD: Preprocessing po uczeniu?", False
    if "Uczenie" in steps and "Kodowanie" in steps:
        if steps.index("Uczenie") < steps.index("Kodowanie"):
            return "⚠️ BŁĄD: Preprocessing po uczeniu?", False

    return "✅ Kolejność logicznie poprawna!", True


msg, is_valid = validate_pipeline(steps_order)
if is_valid:
    st.success(msg)
else:
    st.error(msg)

st.markdown("---")

st.header("3. Wybór algorytmu i grid search")
algo = st.radio("Wybierz model", ["Decision tree", "kNN"], horizontal=True)

if algo == "Decision tree":
    depth_range = st.slider("Zakres max depth", 1, 20, (1, 5))
    param_grid = {'classifier__max_depth': list(range(depth_range[0], depth_range[1] + 1))}
    model_base = DecisionTreeClassifier()
else:
    k_range = st.slider("Zakres sąsiadów (k)", 1, 50, (1, 15))
    param_grid = {'classifier__n_neighbors': list(range(k_range[0], k_range[1] + 1))}
    model_base = KNeighborsClassifier()

cv_folds = st.number_input("Cross-Validation", 2, 10, 5)

def build_full_pipeline():
    num_features = X.select_dtypes(include=['int64', 'float64']).columns
    cat_features = X.select_dtypes(include=['object', 'category']).columns

    num_sub_steps = []
    cat_sub_steps = []

    for step in steps_order:
        if step == "Imputacja":
            num_sub_steps.append(('imputer', SimpleImputer(strategy=num_impute)))
            cat_sub_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
        elif step == "Skalowanie" and scale_method != "Brak":
            scalers = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(),
                       "RobustScaler": RobustScaler()}
            num_sub_steps.append(('scaler', scalers[scale_method]))
        elif step == "Kodowanie":
            if cat_method == "OneHotEncoder":
                cat_sub_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore')))
            else:
                cat_sub_steps.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline(num_sub_steps) if num_sub_steps else 'passthrough', num_features),
        ('cat', Pipeline(cat_sub_steps) if cat_sub_steps else 'passthrough', cat_features)
    ])

    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model_base)])


st.markdown("---")

if st.button("🚀 Uruchom proces"):
    try:
        final_pipeline = build_full_pipeline()
        grid_search = GridSearchCV(final_pipeline, param_grid, cv=cv_folds, scoring='accuracy', return_train_score=True)

        with st.spinner("Pipeline pracuje..."):
            grid_search.fit(X_train, y_train)

        st.header("4. Analiza i diagnostyka")

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred)
        baseline_acc = y_test.value_counts(normalize=True).max()

        m1, m2, m3 = st.columns(3)
        m1.metric("Finalna trafność testowa", f"{acc_test:.2%}")
        m2.metric("Baseline (zgadywanie)", f"{baseline_acc:.2%}")
        m3.write(f"**Najlepsze parametry:** \n {grid_search.best_params_}")

        st.write("### Krzywa uczenia (tuning)")
        cv_res = pd.DataFrame(grid_search.cv_results_)
        param_col = [c for c in cv_res.columns if "param_classifier__" in c][0]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(cv_res[param_col], cv_res['mean_train_score'], 'o-', label="Błąd treningowy")
        ax.plot(cv_res[param_col], cv_res['mean_test_score'], 'o-', label="Błąd walidacji (CV)")
        ax.set_xlabel("Parametr")
        ax.set_ylabel("Dokładność")
        ax.legend()
        st.pyplot(fig)

        col_a, col_b = st.columns(2)
        with col_a:
            st.write("### Macierz pomyłek")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            st.pyplot(fig_cm)
        with col_b:
            st.write("### Raport klasyfikacji")
            st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

    except Exception as e:
        st.error(f"❌ Pipeline uległ awarii! Powód: {e}")
        st.info("Prawdopodobna przyczyna: Niewłaściwa kolejność kroków lub brak imputacji przy danych z brakami.")