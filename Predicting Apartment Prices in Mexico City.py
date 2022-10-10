# Import libraries here
import warnings
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import wqet_grader
from category_encoders import OneHotEncoder
from IPython.display import VimeoVideo
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge  # noqa F401
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils.validation import check_is_fitted

# Create a wrangle function that takes the name of a CSV file as input and returns a DataFrame.
# Build your `wrangle` function
def wrangle(filepath):
    df= pd.read_csv(filepath)
    
    #split place with parent names = "Distrito Federal"
    df["neighbourhood"] = df["place_with_parent_names"].str.split('|', expand=True)[2]
    df["borough"] = df["place_with_parent_names"].str.split('|', expand=True)[1]
    df.drop(columns="place_with_parent_names", inplace=True)

    #Mask data that needs to be filtered out
    mask_city = df["neighbourhood"].str.contains("Distrito Federal")
    mask_price = df["price_aprox_usd"] < 100000
    mask_property = df["property_type"].str.contains("apartment")

    df = df[mask_price & mask_city & mask_property]

    #Remove top/bottom 10% values from surface_covered_in_m2
    low, high = df["surface_covered_in_m2"].quantile([0.1,0.9])
    mask_area = df["surface_covered_in_m2"].between(low, high)
    df = df[mask_area]

    #SPlit lat-lon
    df[["lat","lon"]] = df["lat-lon"].str.split(',', expand=True).astype(float)
    df.drop(columns="lat-lon", inplace=True)

    #Remove colums with 50% or more null values
    df.drop(columns = ["surface_total_in_m2",
                        "price_usd_per_m2",
                        "floor",
                        "rooms",
                        "expenses"], inplace=True)

    #Drop values with high/low cardinality i.e. high/low occurences of distinct value
    df.drop(columns = ["operation","property_type","currency","properati_url","neighbourhood"], inplace = True)

    #Drop columns that would cause leakages to price_aprox_usd
    df.drop(columns=["price","price_aprox_local_currency","price_per_m2"], inplace=True)

    return df
# Use this cell to test your wrangle function and explore the data
df = wrangle("data/mexico-city-real-estate-1.csv")
df
#Use glob to create the list files containing the filenames of all the Mexico City real estate CSVs in the ./dataset directory, 
# except for mexico-city-test-features.csv
files = glob("data/mexico-city-real-estate-*.csv")
files

# Combine your wrangle function, a list comprehension
# and pd.concat to create a DataFrame df
frames = []
for file in files:
    df=wrangle(file)
    frames.append(df)
    
df = pd.concat(frames, ignore_index=True)
print(df.info())
df.head(10)

(df
 .borough
 .value_counts()
)

# EXPLORATORY DATA ANALYSIS 

# Build histogram showing the distribution of apartment prices
plt.hist(df["price_aprox_usd"])
plt.xlabel("Area [sq meters]")
plt.ylabel("Count")
plt.title("Distribution of Apartment Prices")

# Build scatter plot that shows apartment price ("price_aprox_usd") as a function of apartment size ("surface_covered_in_m2")
plt.scatter(x=df["surface_covered_in_m2"], y=df["price_aprox_usd"])
plt.xlabel("Price [USD]")
plt.ylabel("Area [sq meters]")
plt.title("Mexico City: Price vs. Area")

# Plot Mapbox location that shows the location of the apartments and represent their price using color.

fig = px.scatter_mapbox(
    df,  # Our DataFrame
    lat="lat",
    lon="lon",
    width=600,  # Width of map
    height=600,  # Height of map
    color="price_aprox_usd",
    hover_data=["price_aprox_usd"],  # Display price when hovering mouse over house
)

fig.update_layout(mapbox_style="open-street-map")

fig.show()

# Split data into feature matrix `X_train` and target vector `y_train` with target as "price_aprox_usd"
target = "price_aprox_usd"
features= ["surface_covered_in_m2", "lat", "lon", "borough"]
X_train = df[features]
y_train = df[target]

# BUILD MODEL
#Baseline
# Calculate the baseline mean absolute error for your model.
y_mean = y_train.mean()
y_pred_baseline = [y_mean]*len(y_train)
baseline_mae = mean_absolute_error(y_train, y_pred_baseline)
print("Mean apt price:", y_mean)
print("Baseline MAE:", baseline_mae)

#Iterate

#Instantiate
ohe = OneHotEncoder(use_cat_names =True)
#fit 
ohe.fit(X_train)
#Transform
XT_train = ohe.transform(X_train)
print(XT_train.shape)
XT_train.head()

#Instantiate a SimpleImputer to deal with issues like missing values
imputer = SimpleImputer()
#fit 
imputer.fit(X_train)
#Transform
XT_train = imputer.transform(X_train)
pd.DataFrame(XT_train, columns=X_train.columns).info()

# Build Model
model = make_pipeline(
    OneHotEncoder(use_cat_names =True),
    SimpleImputer(),
    Ridge()
)
# Fit model
model.fit(X_train, y_train)

#B. Evaluate model 
X_test = pd.read_csv("data/mexico-city-test-features.csv")[features]
print(X_test.info())
X_test.head()


#Use your model to generate a Series of predictions for X_test
y_test_pred = pd.Series(model.predict(X_test))
y_test_pred.head()

# Communicate Results
# Create a Series named feat_imp.
coefficients = model.named_steps["ridge"].coef_
intercept = model.named_steps["ridge"].intercept_
features = model.named_steps["ridge"].coef_
feature_names = model.named_steps["onehotencoder"].get_feature_names()
feat_imp = feat_imp = pd.Series(coefficients, index=feature_names)
feat_imp

print(f"price = {intercept.round(2)}")
for f, c in feat_imp.items():
    print(f"+ ({round(c, 2)} * {f})")

#Create a horizontal bar chart that shows the 10 most influential coefficients for your model.
# abel your x- and y-axis "Importance [USD]" and "Feature" with  "Feature Importances for Apartment Price" as title
# Build bar chart
feat_imp.sort_values(key=abs).tail(10).plot(kind="barh")
plt.xlabel("Importance [USD]")
plt.ylabel("Feature")
plt.title("Feature Importance for Apartment Price")