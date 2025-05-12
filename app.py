import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
import plotly.express as px

# Load Iris data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Title
st.title("🌸 Iris Dataset Dashboard")

# Sidebar filters
st.sidebar.header("Filter options")

species_filter = st.sidebar.multiselect(
    "Select species to display",
    options=df["species"].unique(),
    default=df["species"].unique()
)

x_axis = st.sidebar.selectbox("X-axis", df.columns[:-1], index=0)
y_axis = st.sidebar.selectbox("Y-axis", df.columns[:-1], index=1)

filtered_df = df[df["species"].isin(species_filter)]

# Show filtered data
st.subheader("Filtered Dataset")
st.dataframe(filtered_df)

# Plot
st.subheader("Interactive Scatter Plot")
fig = px.scatter(
    filtered_df,
    x=x_axis,
    y=y_axis,
    color="species",
    title=f"{x_axis} vs {y_axis}",
    labels={x_axis: x_axis.title(), y_axis: y_axis.title()}
)
st.plotly_chart(fig, use_container_width=True)
