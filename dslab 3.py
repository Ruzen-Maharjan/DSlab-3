import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
import errors as err


# -----------------------------
# LOAD AND PREPARE DATA
# -----------------------------
# Read the World Bank GDP and CO2 datasets.
# skiprows=4 is used because World Bank CSV files contain extra header rows.
gdp = pd.read_csv("GDP.csv", skiprows=4)
co2 = pd.read_csv("Co2.csv", skiprows=4)

# Select one recent year for the clustering analysis.
year = "2020"

# Keep only country names and the selected year.
gdp = gdp[["Country Name", year]]
co2 = co2[["Country Name", year]]

# Rename columns to make the data easier to work with.
gdp.columns = ["Country", "GDP"]
co2.columns = ["Country", "CO2"]

# Merge the two datasets so each country has both GDP and CO2 values.
df = pd.merge(gdp, co2, on="Country")

# Remove rows with missing values.
df = df.dropna()

# Remove regional or income-group aggregates so only individual countries remain.
aggregates = [
    "World", "High income", "Low income", "Lower middle income",
    "Upper middle income", "European Union",
    "Africa Eastern and Southern", "Africa Western and Central",
    "Arab World", "East Asia & Pacific", "Europe & Central Asia",
    "Latin America & Caribbean", "Middle East & North Africa",
    "North America", "South Asia", "Sub-Saharan Africa"
]
df = df[~df["Country"].isin(aggregates)]


# -----------------------------
# CLUSTERING
# -----------------------------
# Use GDP per capita and CO2 emissions per capita for clustering.
cluster_data = df[["GDP", "CO2"]].copy()

# Standardise the variables so both contribute equally to the clustering.
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

# Apply KMeans clustering with 3 clusters.
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(scaled_data)

# Add the cluster labels back to the original dataframe.
df["Cluster"] = labels

# Convert the cluster centres back to the original scale for interpretation.
centres = scaler.inverse_transform(kmeans.cluster_centers_)

# Plot the countries on the original GDP and CO2 scale.
plt.figure(figsize=(8, 5))
plt.scatter(df["GDP"], df["CO2"], c=df["Cluster"])

# Mark the cluster centres clearly on the plot.
plt.scatter(centres[:, 0], centres[:, 1], c="red", marker="x", s=120)

plt.xlabel("GDP per capita")
plt.ylabel("CO2 emissions per capita")
plt.title("Clustering of Countries")
plt.savefig("cluster_plot.png", dpi=300)
plt.show()


# -----------------------------
# CURVE FITTING
# -----------------------------
# Read the full CO2 dataset again so that all years can be used for fitting.
df_full = pd.read_csv("Co2.csv", skiprows=4)

# Select one country for the time-series analysis.
country = "India"
country_data = df_full[df_full["Country Name"] == country]

# Extract all year columns from the dataset.
years = np.array([int(c) for c in df_full.columns if c.isdigit()])

# Get the CO2 values for the selected country across all years.
values = country_data[[str(y) for y in years]].values.flatten().astype(float)

# Remove missing values before fitting the model.
mask = ~np.isnan(values)
x = years[mask]
y = values[mask]

# Shift the years to improve the numerical stability of the quadratic fit.
start_year = x.min()

# Use a simple quadratic model, which is enough to show the overall trend.
def model(x, a, b, c):
    x_shift = x - start_year
    return a * x_shift**2 + b * x_shift + c

# Fit the model to the data.
params, covar = curve_fit(model, x, y)

# Calculate fitted values.
y_fit = model(x, *params)

# Estimate the confidence range using the provided error function.
sigma = err.error_prop(x, model, params, covar)
lower = y_fit - sigma
upper = y_fit + sigma

# Plot the original data, fitted curve, and confidence range.
plt.figure(figsize=(8, 5))
plt.plot(x, y, label="Data")
plt.plot(x, y_fit, label="Fit")
plt.fill_between(x, lower, upper, alpha=0.3, label="Confidence range")

plt.xlabel("Year")
plt.ylabel("CO2 emissions per capita")
plt.title("CO2 Emissions Trend (India)")
plt.legend()
plt.savefig("fit_plot.png", dpi=300)
plt.show()


# -----------------------------
# PREDICTION
# -----------------------------
# Use the fitted model to estimate CO2 emissions in 2030.
future = 2030
prediction = model(future, *params)

# Estimate the uncertainty in the prediction.
pred_sigma = err.error_prop(np.array([future]), model, params, covar)[0]

print("Prediction for 2030:", prediction)
print("Lower bound:", prediction - pred_sigma)
print("Upper bound:", prediction + pred_sigma)