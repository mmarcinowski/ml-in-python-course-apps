import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay

st.title("🚀 Moje pierwsze Laboratorium ML")

# 1. Wybór danych
dataset_name = st.sidebar.selectbox("Wybierz zbiór danych", ["Iris", "Penguins", "Diamonds"])
df = sns.load_dataset(dataset_name.lower()).dropna()

st.write(f"### Podgląd danych: {dataset_name}")
st.dataframe(df.head())

# 2. Wybór cech do wizualizacji
features = df.columns[:-1].tolist()
x_axis = st.sidebar.selectbox("Oś X", features)
y_axis = st.sidebar.selectbox("Oś Y", features, index=1)

# 3. Modelowanie
st.sidebar.markdown("---")
st.sidebar.write("### Parametry Modelu")
depth = st.sidebar.slider("Głębokość drzewa (Max Depth)", 1, 10, 3)

# Przygotowanie danych do prostego modelu 2D
X = df[[x_axis, y_axis]]
y = df.iloc[:, -1].astype('category').cat.codes

clf = DecisionTreeClassifier(max_depth=depth)
clf.fit(X, y)

# Wizualizacja wyników
fig, ax = plt.subplots()
DecisionBoundaryDisplay.from_estimator(
    clf, X, cmap=plt.cm.RdYlBu, response_method="predict", ax=ax, alpha=0.8
)
sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=df.columns[-1], palette="bright", ax=ax)
st.pyplot(fig)