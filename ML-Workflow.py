#!/usr/bin/env python
# coding: utf-8

# # ML Workflow

# The objective of this exercise is to use the tools and methods that you learned during the previous weeks, in order to solve a **real challenge**.
# 
# The problem to solve is a **Kaggle Competition**: [New York City Taxi Fare Prediction](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction). The goal is to predict the fare amount (inclusive of tolls) for a taxi ride in New York City given the pickup and dropoff locations.

# Building a machine learning model requires a few different steps.

# ## Steps
# 1. Get the data
# 2. Explore the data
# 3. Data cleaning
# 4. Evaluation metric
# 5. Model baseline
# 6. Build your first model
# 7. Model evaluation
# 8. Kaggle submission
# 9. Model iteration

# ## 1. Get the data <a id='part1'></a>

# The dataset is available on [Kaggle](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data).
# 
# First of all:
# - Follow the instructions to download the training and test sets
# - Put the datasets in a separate folder on your local disk. You can name it "data" for example.

# Now we are going to use Pandas to read and explore the datasets.

# In[1]:


import pandas as pd


# The training dataset is relatively big (~5GB).
# So let's only open a portion of it.
# 👉 Go to [Pandas documentation](https://pandas.pydata.org/pandas-docs/stable/) to see how to open a portion of CSV file and store it into a DataFrame (ex: just read 1 million rows maximum)
# 
# 💡 NB: here we will read portion of the file
# 
# 

# In[2]:


df = pd.read_csv('data/train.csv', nrows=1000000)


# Now let's display the first rows to understand the different fields 

# In[3]:


df.head(10)


# ## 2. Explore the data <a id='part2'></a>

# Before trying to solve the prediction problem, we need to get a better understanding of the data.
# In order to do that, we are going to use libraries such as Pandas and Seaborn.
# First of all, make sure you have [Seaborn](https://seaborn.pydata.org/) installed and import it into your notebook.
# Note that it can also be useful to import `matplotlib.pyplot` in order to customize a few things.

# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 14
plt.figure(figsize=(12,5))
palette = sns.color_palette('Paired', 10)


# ### There are multiple things we want to do in terms of data exploration
# 
# - You first want to look at the distribution of the variable that you are going to predict: "fare_amount"
# - Then you want to visualize other variable distributions
# - And finally it is often very helpful to compute and visualize the correlation between the target variable and other variables
# - Also, lets look for any missing values, or other irregularities

# ### Explore the target variable
# - Compute simple statistics for the target variable (min, max, mean, std, etc)
# - Plot distributions

# In[5]:


df.describe()


# In[6]:


def plot_dist(series=df["fare_amount"], title="Fare Distribution"):
    
    sns.distplot(series)
plot_dist()


# In[7]:


# drop absurd values 
df = df[df.fare_amount.between(0, 60)]
plot_dist(df.fare_amount)


# In[8]:


import numpy as np 

# we can also visualize binned fare_amount variable
df['fare-bin'] = pd.cut(df['fare_amount'], bins = list(range(0, 50, 5)), include_lowest=True).astype('str')

# uppermost bin
df['fare-bin'] = df['fare-bin'].replace(np.nan, '[45+]')
# df.loc[df['fare-bin'] == 'nan', 'fare-bin'] = '[45+]'

# apply this to clean up the label of the first bin
df['fare-bin'] = df['fare-bin'].apply(lambda x: x.replace('-0.001', '0'))

# sort by fare the correct look in the chart
df = df.sort_values(by='fare_amount')


# In[9]:


sns.catplot(x="fare-bin", kind="count", palette=palette, data=df, height=5, aspect=3);
sns.despine()
plt.show()


# ### Explore other variables
# 
# - passenger_count (statistics + distribution)
# - pickup_datetime (you need to build time features out of the pickup datetime)
# - Geospatial features (pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude)
# - Find other variables that you can compute from existing data and that might explain the target

# #### Passenger Count

# In[10]:


df.passenger_count.describe()
df = df[df["passenger_count"].between(0, 12)]
df.passenger_count.describe()


# In[11]:


sns.distplot(df['passenger_count'], kde=False)

# Show the plot
plt.show()


# #### Pickup Datetime
# 
# - Extract time features from pickup_datetime (hour, day of week, month, year)
# - Create a method `def extract_time_features(_df)` that you will be able to re-use later
# - Be careful with the timezone
# - Explore the newly created features 

# In[12]:


def extract_time_features(df):
# Extract the year, month, day, and hour components and add them as separate columns
    df["pickup_datetime"] = pd.to_datetime(df['pickup_datetime'])
    # Extract the year, month, day, hour, minute, and second components and add them as separate columns
    df["pickup_datetime"] = pd.to_datetime(df['pickup_datetime'].dt.tz_convert("US/Eastern"))
    df['year'] = df['pickup_datetime'].dt.year
    df['month'] = df['pickup_datetime'].dt.month
    df['dow'] = df['pickup_datetime'].dt.dayofweek
    df['hour'] = df['pickup_datetime'].dt.hour
    return df


# In[13]:


df.head(100)


# In[14]:


get_ipython().run_cell_magic('time', '', 'df = extract_time_features(df)')


# In[15]:


# Plot hour of day
sns.catplot(x="hour", kind="count", palette=palette, data=df, height=5, aspect=3);
sns.despine()
plt.title('Hour of Day');
plt.show()


# In[16]:


# Plot  day of week
sns.catplot(x="dow", kind="count", palette=palette, data=df, height=5, aspect=3);
sns.despine()
plt.title('Day of Week');
plt.show()


# #### Add timezone features
# 
# - Extract time features from pickup_datetime (hour, day of week, month, year)
# - Create a method `def extract_time_features(_df)` that you will be able to re-use later
# - Be careful of timezone
# - Explore the newly created features

# In[17]:


df_test = pd.read_csv("./data/test.csv")


# In[18]:


# find the boudaries from the test set and remove the outliers from the training set
for col in ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]:
    MIN = df_test[col].min()
    MAX = df_test[col].max()
    print(col, MIN, MAX)


# In[19]:


df = df[df["pickup_latitude"].between(left = 40, right = 42 )]
df = df[df["pickup_longitude"].between(left = -74.3, right = -72.9 )]
df = df[df["dropoff_latitude"].between(left = 40, right = 42 )]
df = df[df["dropoff_longitude"].between(left = -74, right = -72.9 )]


# In[20]:


# make sure that you install folium first
import folium
mean_latitude = df["pickup_latitude"].mean()
mean_longitude = df["pickup_longitude"].mean()

import folium
from folium import plugins

# Create a map centered on New York City
nyc_map = folium.Map(location=[40.730610, -73.935242], zoom_start=10)

# Create a list of pickup and dropoff locations
locations = []
for i, row in df.iterrows():
    locations.append([row["pickup_latitude"], row["pickup_longitude"]])
    locations.append([row["dropoff_latitude"], row["dropoff_longitude"]])

# Add a heatmap layer to the map
heatmap = plugins.HeatMap(locations)
heatmap.add_to(nyc_map)
nyc_map


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# #### Distance
# 
# - Compute the distance between pickup and dropoff locations (tip: https://en.wikipedia.org/wiki/Haversine_formula)
# - Write a method `def haversine_distance(df, **kwargs)` that you will be able to reuse later
# - Compute a few statistics for distance and plot distance distribution

# In[133]:


import numpy as np
import haversine as hs
def haversine_distance(df,
                       start_lat="start_lat",
                       start_lon="start_lon",
                       end_lat="end_lat",
                       end_lon="end_lon"):

    def compute_distance(x):
        return hs.haversine((x[start_lat], x[start_lon]), (x[end_lat], x[end_lon]))

    df["distance"] = df.apply(compute_distance, axis=1)
 
    return df

# Use the haversine formula to calculate the distance between two points
haversine_distance(df,start_lat="pickup_latitude",
                       start_lon="pickup_longitude",
                       end_lat="dropoff_latitude",
                       end_lon="dropoff_longitude")


# In[117]:


df.distance.describe()


# In[23]:


get_ipython().run_line_magic('matplotlib', 'inline')
plot_dist(series=df[df.distance < 50].distance, title='Distance distribution')


# #### Explore how target variable correlate with other variables
# 
# - As a first step, you can visualize the target variable vs another variable. For categorical variables, it is often useful to compute the average target variable for each category (Seaborn has plots that do it for you!). For continuous variables (like distance, you can use scatter plots, or regression plots, or bucket the distance into different bins
# - But there many different ways to visualize correlation between features, so be creative

# In[24]:


sns.catplot(x="passenger_count", y="fare_amount", palette=palette, data=df, kind="bar", aspect=3)
sns.despine()
plt.show()


# In[25]:


sns.catplot(x="hour", y="fare_amount", palette=palette, data=df, kind="bar", aspect=3)
sns.despine()
plt.show()


# In[26]:


sns.catplot(x="dow", y="fare_amount", palette=palette, data=df, kind="bar", aspect=3)
sns.despine()
plt.show()


# In[27]:


sns.scatterplot(x="distance", y="fare_amount", data=df[df.distance < 80].sample(100000))
plt.show()


# In[28]:


sns.scatterplot(x="distance", y="fare_amount", hue="passenger_count", data=df[df.distance < 80].sample(100000))
plt.show()


# ## 3. Data cleaning <a id='part3'></a>

# As you probably saw in the previous section during your data exploration, there are some values that do not seem valid.
# In this section, you will take a few steps to clean the training data.

# Remove all the trips that look incorrect. We recommend that you write a method called `clean_data(df)` that you will be able to re-use in the next steps.

# In[29]:


print("trips with negative fares:", len(df[df.fare_amount <= 0]))
print("trips with too high distance:", len(df[df.distance >= 100]))
print("trips with too many passengers:", len(df[df.passenger_count > 8]))
print("trips with zero passenger:", len(df[df.passenger_count == 0]))


# In[31]:


def clean_data(df):
    # Remove trips with negative fares
    df = df[df.fare_amount > 0]
    # Remove trips with too high distance
    df = df[df.distance < 100]
    # Remove trips with too many passengers
    df = df[df.passenger_count <= 8]
    # Remove trips with zero passengers
    df = df[df.passenger_count > 0]
    
    return df

# Clean the data using the defined function
df_cleaned = clean_data(df)

"% data removed", (1 - len(df_cleaned) / len(df)) * 100


# In[32]:


df_cleaned.columns


# ## 4. Evaluation metric <a id='part4'></a>

# The evaluation metric for this competition is the root mean-squared error or RMSE. The RMSE measures the difference between the predictions of a model, and the corresponding ground truth. A large RMSE is equivalent to a large average error, so smaller values of RMSE are better.
# 
# More details here https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/overview/evaluation

# Write a method `def compute_rmse(y_pred, y_true)` that computes the RMSE given `y_pred` and `y_true` which are two numpy arrays corresponding to model predictions and ground truth values.
# 
# This method will be useful in order to evaluate the performance of your model.

# In[57]:


import numpy as np

def compute_rmse(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true)**2))


# ## 5. Model baseline <a id='part5'></a>

# Before building your model, it is often useful to get a performance benchmark. For this, you will use a baseline model that is a very dumb model and compute the evaluation metric on that model.
# Then, you will be able to see how much better your model is compared to the baseline. It is very common to see ML teams coming up with very sophisticated approaches without knowing by how much their model beats the very simple model.

# - Generate predictions based on a simple heuristic
# - Evaluate the RMSE for these predictions

# In[64]:


from sklearn.dummy import DummyRegressor
import pandas as pd
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(df.drop(columns = ["fare_amount"]), df["fare_amount"], test_size = 0.2)
model = DummyRegressor()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(compute_rmse(y_pred,y_test))


# In[ ]:





# ## 6. Build your first model <a id='part6'></a>

# Now it is time to build your model!
# 
# For starters we are going to use a linear model only. We will try more sophisticated models later.

# Here are the different steps that you have to follow:
# 
# 1. Split the data into two different sets (training and validation). You will be measuring the performance of your model on the validation set
# 2. Make sure that you apply the data cleaning on your training set
# 3. Think about the different features you want to add in your model
# 4. For each of these features, make sure you apply the correct transformation so that the model can correctly learn from them (this is true for categorical variables like `hour of day` or `day of week`)
# 5. Train your model

# ##### Training/Validation Split

# In[122]:


# training/validation
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.1)


# ##### Apply data cleaning on training set

# In[123]:


df_train = clean_data(df_train)


# ##### List features (continuous vs categorical)

# In[124]:


# features
target = "fare_amount"
features = ["distance", "hour", "dow", "passenger_count"]
categorical_features = ["hour", "dow", "passenger_count"]


# In[125]:


df_train.head()


# ##### Features transformation
# 
# - Write a method `def transform_features(df, **kwargs)` because you will have to make sure that you apply the same transformations on the validation (or test set) before making predictions
# - For categorical features transformation, you can use `pandas.get_dummies` method

# In[126]:


def transform_features(df, dummy_features=None):
    df = pd.get_dummies(df, columns = dummy_features)
    return df, dummy_features


# ##### Model training

# In[127]:


# model training
from sklearn.linear_model import LassoCV
model = LassoCV(cv=5, n_alphas=5)
X_train, dummy_features = transform_features(df_train, dummy_features = categorical_features)
X_train = df_train[features]
y_train = df_train.fare_amount
model.fit(X_train, y_train)


# ## 7. Model evaluation <a id='part7'></a>

# Now in order to evaluate your model, you need to use your previously trained model in order to make predictions on the validation set.
# 
# For this, follow these steps:
# 1. Apply the same transformations on the validation set
# 2. Make predictions
# 3. Evaluate predictions using `compute_rmse` method

# In[128]:


# X_val, _ = transform_features(df_val, dummy_features=dummy_features)
X_test = df_test[features]
df_test["y_pred"] = model.predict(X_test)
compute_rmse(df_test.y_pred, df_test.fare_amount)


# ## 8. Kaggle submission <a id='part8'></a>

# Now that you have a model, you can now make predictions on Kaggle test set and be evaluated by Kaggle directly.
# 
# - Download the test data from Kaggle
# - Follow the [instructions](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/overview/evaluation) to make sure that your predictions are in the right format
# - Re-train your model using all the data (do not split with train/validation)
# - Apply the feature engineering and transformation methods on the test set
# - Use the model to make predictions on the test set
# - Submit your predictions!

# In[129]:


# Re-train the model with all the data
df_cleaned = clean_data(df)
X = df_cleaned[features]
y = df_cleaned.fare_amount

from sklearn.linear_model import Lasso

lasso = Lasso(alpha=5)
lasso.fit(X,y)


# In[130]:


# load Kaggle's test set
df_test = pd.read_csv("./data/test.csv")
df_test.head(1)


# In[134]:


# feature engineering
df_test["distance"] = haversine_distance(df_test, 
                                         start_lat="pickup_latitude", start_lon="pickup_longitude",
                                         end_lat="dropoff_latitude", end_lon="pickup_longitude")

df_test = extract_time_features(df_test)
# X_test, _ = transform_features(df_test, dummy_features=dummy_features) 
X_test = df_test[features]

# prediction
df_test["y_pred"] = lasso.predict(X_test)


# In[107]:


df_test.head(1)


# In[108]:


df_test.reset_index(drop=True)[["key", "y_pred"]].rename(columns={"y_pred": "fare_amount"}).to_csv("lasso_v0_predictions.csv", index=False)


# ## 9. [OPTIONAL] Push further Feature Engineering <a id='part9'></a>

# You can improve your model by trying different things (But do not worry, some of these things will be covered during the next days).
# - Use more data to train
# - Build and add more features
# - Try different estimators
# - Adjust your data cleaning to remove more or less data
# - Tune the hyperparameters of your model

# In the following section we will focus on advanced feature engineering (keep in mind that relevant feateng is often key to significant increase in model performances):
# 
# 👉 **Manhattan distance** better suited to our problem  
# 👉 **Distance to NYC center** to highlight interesting pattern...
# 👉 **Direction**

# ###### Another Distance ?
# - Think about the distance you used, try and find a more adapted distance for our problem (ask TAs for insights)

# $$D(A,B) = \left( \sum_{i=1}^{n} \lvert x_{A_i} - x_{B_i} \rvert ^p \right)^\frac{1}{p}$$
# with $A=(x_{A_1}, x_{A_2}, ..., x_{A_n})$ and $B=(x_{B_1}, x_{B_2}, ..., x_{B_n})$

# In[ ]:


# the Minkowski Distance is actually the generic distance to compute different distances

# in a cartesion system of reference of 2 dimensions (x,y), the Minkowski distance can be implemented as follow:
def minkowski_distance(x1, x2, y1, y2, p):
   


# In[ ]:


# in a GPS coordinates system, the Minkowksi distance should be implented as follows:
# convert degrees to radians
def deg2rad(coordinate):
    return 

# convert radians into distance
def rad2dist(coordinate):
   
    return 

# correct the longitude distance regarding the latitude (https://jonisalonen.com/2014/computing-distance-between-coordinates-can-be-simple-and-fast/)
def lng_dist_corrected(lng_dist, lat):
    return 

def minkowski_distance_gps(lat1, lat2, lon1, lon2, p):
 
    return minkowski_distance(x1, x2, y1, y2, p)


# In[ ]:


# manhattan distance <=> minkowski_distance(x1, x2, y1, y2, 1)
df['manhattan_dist'] = minkowski_distance_gps(df['pickup_latitude'], df['dropoff_latitude'],
                                              df['pickup_longitude'], df['dropoff_longitude'], 1)


# In[ ]:


# euclidian distance <=> minkowski_distance(x1, x2, y1, y2, 2)
df['euclidian_dist'] = minkowski_distance_gps(df['pickup_latitude'], df['dropoff_latitude'],
                                              df['pickup_longitude'], df['dropoff_longitude'], 2)


# In[ ]:


df.head()


# ###### Distance from the center 
# 
# - Compute a new feature calculating the distance of pickup location from the center
# - Scatter Plot *distance_from_center* regarding *distance* 
# - What do you observe ? What new features could you add ? How are these new features correlated to the target ?

# In[ ]:


# let's compute the distance from the NYC center
# A COMPLETER

df['distance_to_center'] = haversine_distance(df, **args)


# In[ ]:


idx = (df.distance < 40) & (df.distance_to_center < 40)
sns.scatterplot(x="distance_to_center", y="distance", data=df[idx].sample(10000), hue="fare-bin")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()


# In[ ]:


df.distance_to_center.hist(bins=100)


# 👉 **Take some time to step back and try to observe an interesting pattern here. What are these clusters with a similar distance to the center?**

# In[ ]:


# seems to be fixed distance_to_center


# In[ ]:


df.pickup_distance_to_jfk.hist(bins=100)


# ###### Which direction  are you heading to ?
# 
# - Compute a new feature calculating the direction your are heading to
# - What do you observe ? What new features could you add ? How are these new features correlated to the target ?

# In[ ]:


def calculate_direction(d_lon, d_lat):

    return result


# In[ ]:


df['delta_lon'] = df.pickup_longitude - df.dropoff_longitude
df['delta_lat'] = df.pickup_latitude - df.dropoff_latitude
df['direction'] = calculate_direction(df.delta_lon, df.delta_lat)


# In[ ]:


plt.figure(figsize=(10,6))
df.direction.hist(bins=180)


# In[ ]:


# plot direction vs average fare amount for fares inside manhattan
def select_within_boundingbox(df, BB):
    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) &            (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) &            (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) &            (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])
BB_manhattan = (-74.025, -73.925, 40.7, 40.8)
idx_manhattan = select_within_boundingbox(df, BB_manhattan)

fig, ax = plt.subplots(1, 1, figsize=(14,6))
direc = pd.cut(df[idx_manhattan]['direction'], np.linspace(-180, 180, 37))
df[idx_manhattan].pivot_table('fare_amount', index=[direc], columns='year', aggfunc='mean').plot(ax=ax)
plt.xlabel('direction (degrees)')
plt.xticks(range(36), np.arange(-170, 190, 10))
plt.ylabel('average fare amount $USD');


# In[ ]:


corrs = df.corr()
l = list(corrs)
l.remove("fare_amount")
corrs['fare_amount'][l].plot.bar(color = 'b');
plt.title('Correlation with Fare Amount');


# In[ ]:




