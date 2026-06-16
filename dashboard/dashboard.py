import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PAGE CONFIG

st.set_page_config(
    page_title="Air Quality Dashboard",
    page_icon="🌫️",
    layout="wide"
)

# LOAD DATA

@st.cache_data
def load_data():
    df = pd.read_csv("main_data.csv")
    return df

air_quality_df = load_data()

# SIDEBAR

st.sidebar.header("Filter Data")

selected_station = st.sidebar.multiselect(
    "Select Station",
    options=sorted(air_quality_df["station"].unique()),
    default=sorted(air_quality_df["station"].unique())
)

selected_year = st.sidebar.multiselect(
    "Select Year",
    options=sorted(air_quality_df["year"].unique()),
    default=sorted(air_quality_df["year"].unique())
)

filtered_df = air_quality_df[
    (air_quality_df["station"].isin(selected_station))
    &
    (air_quality_df["year"].isin(selected_year))
]

# TITLE

st.title("🌫️ Beijing Air Quality Dashboard")
st.markdown(
    """
    Dashboard untuk menganalisis kualitas udara Beijing
    berdasarkan konsentrasi PM2.5 dan faktor meteorologi.
    """
)

# KPI

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Average PM2.5",
        f"{filtered_df['PM2.5'].mean():.2f}"
    )

with col2:
    st.metric(
        "Maximum PM2.5",
        f"{filtered_df['PM2.5'].max():.2f}"
    )

with col3:
    st.metric(
        "Average Wind Speed",
        f"{filtered_df['WSPM'].mean():.2f}"
    )

highest_station = (
    filtered_df
    .groupby("station")["PM2.5"]
    .mean()
    .idxmax()
)

st.info(
    f"""
    Insight:
    Stasiun dengan rata-rata PM2.5 tertinggi pada data yang dipilih adalah **{highest_station}**.
    Kecepatan angin menunjukkan hubungan negatif terhadap PM2.5, yang mengindikasikan bahwa angin membantu mengurangi konsentrasi polutan.
    """
)

# CHART 1

st.subheader("Average PM2.5 by Station")

avg_pm25_station = (
    filtered_df
    .groupby("station")["PM2.5"]
    .mean()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(10,5))

sns.barplot(
    x=avg_pm25_station.index,
    y=avg_pm25_station.values,
    ax=ax
)

ax.set_xlabel("Station")
ax.set_ylabel("Average PM2.5")
ax.tick_params(axis="x", rotation=45)

st.pyplot(fig)

# CHART 2

st.subheader("Average Monthly PM2.5 Concentration")

monthly_pm25 = (
    filtered_df
    .groupby("month")["PM2.5"]
    .mean()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(10,5))

sns.lineplot(
    data=monthly_pm25,
    x="month",
    y="PM2.5",
    marker="o",
    ax=ax
)

ax.set_xlabel("Month")
ax.set_ylabel("Average PM2.5")

st.pyplot(fig)

# CHART 3

st.subheader("Correlation Heatmap")

corr = filtered_df[
    ["PM2.5", "TEMP", "PRES", "RAIN", "WSPM"]
].corr()

fig, ax = plt.subplots(figsize=(8,6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)

# CHART 4

st.subheader("PM2.5 vs Wind Speed")

sample_df = filtered_df.sample(
    min(5000, len(filtered_df)),
    random_state=42
)

fig, ax = plt.subplots(figsize=(8,5))

sns.scatterplot(
    data=sample_df,
    x="WSPM",
    y="PM2.5",
    alpha=0.5,
    ax=ax
)

ax.set_xlabel("Wind Speed (WSPM)")
ax.set_ylabel("PM2.5")

st.pyplot(fig)

# CLUSTERING

st.subheader("Station Pollution Level Clustering")

station_pm25 = (
    filtered_df
    .groupby("station")["PM2.5"]
    .mean()
    .reset_index()
)

def pollution_level(pm25):
    if pm25 >= 85:
        return "High"
    elif pm25 >= 75:
        return "Medium"
    else:
        return "Low"

station_pm25["pollution_level"] = (
    station_pm25["PM2.5"]
    .apply(pollution_level)
)

fig, ax = plt.subplots(figsize=(10,5))

sns.barplot(
    data=station_pm25.sort_values(
        "PM2.5",
        ascending=False
    ),
    x="station",
    y="PM2.5",
    hue="pollution_level",
    ax=ax
)

ax.tick_params(axis="x", rotation=45)

st.pyplot(fig)

st.dataframe(
    station_pm25.sort_values(
        "PM2.5",
        ascending=False
    ),
    use_container_width=True
)

# DATA TABLE

st.subheader("Dataset Preview")

st.dataframe(
    filtered_df.head(20),
    use_container_width=True
)