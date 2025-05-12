import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris
import joblib
import numpy as np

model = joblib.load("iris_model.pkl")

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

st.sidebar.title("🔍 Filter Options")
species_filter = st.sidebar.multiselect(
    "Select species to display",
    options=df["species"].unique(),
    default=df["species"].unique()
)

x_axis = st.sidebar.selectbox("X-axis", df.columns[:-1])
y_axis = st.sidebar.selectbox("Y-axis", df.columns[:-1])

filtered_df = df[df["species"].isin(species_filter)]

st.title("🌸 Iris Dashboard with Prediction")

st.subheader("Filtered Dataset")
st.dataframe(filtered_df)

st.subheader("Interactive Plot")
fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color="species")
st.plotly_chart(fig, use_container_width=True)

st.subheader("🌟 Predict Iris Species")
with st.form("prediction_form"):
    sepal_length = st.number_input("Sepal length (cm)", value=5.1)
    sepal_width = st.number_input("Sepal width (cm)", value=3.5)
    petal_length = st.number_input("Petal length (cm)", value=1.4)
    petal_width = st.number_input("Petal width (cm)", value=0.2)

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)[0]
        pred_name = iris.target_names[prediction]
        st.success(f"Predicted Species: **{pred_name}**")
