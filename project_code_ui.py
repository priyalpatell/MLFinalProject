# %% [markdown]
# ## Command to Run UI:

# %% [markdown]
# $ streamlit run project_code_ui.py

# %% [markdown]
# ### **All imports needed to run the program:**

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gdp
import seaborn as sns
import pandas as pd
import scipy.stats as stats
import streamlit as st
import base64

from datetime import date
from dateutil.relativedelta import relativedelta
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, GridSearchCV, RandomizedSearchCV

import warnings
warnings.filterwarnings('ignore')

# %%
def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://images.unsplash.com/photo-1612512836264-5e58fab88bf0?q=80&w=1000&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8dWZvfGVufDB8fDB8fHww");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
set_bg_hack_url()
# image from: #https://unsplash.com/photos/green-and-black-trees-under-blue-sky-HWQXIYbs8PM
title = '<p style="color:White; font-size: 40px;">UFOs...</p>'
st.markdown(title, unsafe_allow_html=True)


# %% [markdown]
# ### **Question 1**

# %%
q1 = '<p style="color:White; font-size: 30px;">When will the next UFO sighting be in California?</p>'
st.markdown(q1, unsafe_allow_html=True)
qinput = '<p style="color:White; font-size: 20px;">Choose how many years ahead to predict a future UFO sighting</p>'
st.markdown(qinput, unsafe_allow_html=True)
year_input = st.number_input('years',min_value=1, max_value=None, value=1, step=1, label_visibility="hidden")

# %% [markdown]
# ### **Upload the dataset and extract the relevant column (datetime):**
# We are only predicting the next UFO sightings in California, so we only need the rows that have 'ca' in their `state` column. Also, we don't need all other columns except for `datetime` to make this prediction.

# %%
df = pd.read_csv("scrubbed.csv")

# List of columns to drop
columns_to_drop = ['comments', 'city', 'date posted', 'shape', 'duration (seconds)', 'duration (hours/min)', 'country', 'state', 'latitude', 'longitude ']

# Filter for rows where 'state' is 'ca' and 'country' is 'us', and drop specified columns and cities
df_filtered = df[(df['state'] == 'ca') & (df['country'] == 'us')].drop(columns=columns_to_drop, axis=1)

# %% [markdown]
# ### **Separate ``datetime`` column into two columns ``Date`` and ``Time``:**
# We are not predicting the exact time of occurence, and therefore we don't need the timing information. However, because the dataset has both date and time together in one column, we want to separate the two and keep the `Date` column only.
# 

# %%
df_filtered[['Date', 'Time']] = df_filtered['datetime'].str.split(' ', n=1, expand=True)
data = df_filtered.drop(columns=['datetime', 'Time'])

# Display the filtered DataFrame
data.head()

# %% [markdown]
# ### **Add extra information to set up for prediction:**
# Here we add an extra column named `Observed` which shows whether a sighting was observed on the corresponding date. Since we only have the dates of UFO sightings observed, we want to fill the dates in between occurences with 0s to indicate no observations.

# %%
# Convert the current date format MM/DD/YYY to YYYY-MM-DD
data['Date'] = pd.to_datetime(data['Date'])
# Add an additional column
data['Observed'] = 1
# Remove duplicate dates as we don't need them
data = data[~data['Date'].duplicated()]
# Sort the dates from earliest to latest and fill the rows in between observed dates with 0
r = pd.date_range(start=data['Date'].min(), end=data['Date'].max())
data = data.set_index('Date').reindex(r).fillna(0.0).rename_axis('Date').reset_index()

data.head()

# %% [markdown]
# ### **Feature engineering:**
# Prediction of next release dates heavily relies on feature engineering because we do not have any features besides the date itself. Therefore, we add extra columns to feed more information about the dates into our model.

# %%
# Extract/create more information about the dates
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
# The number of business day corresponding to the date (1st business day starts from 0)
# for example, 05/07/2014 was the 5th business day of that month, so it's shown as 4 in the chart
data['Workday_N'] = np.busday_count(
                    data['Date'].values.astype('datetime64[M]'),
                    data['Date'].values.astype('datetime64[D]'))
# Which day of the week that day was (0 to 6 for Monday to Sunday)
# for example, 08/15/1937 was Sunday, so it's shown as 6 in the chart
data['Week_day'] = data['Date'].dt.weekday
# Which week of the month that month was
# for example, the week of 08/16/1937 was the 4th week of that month
data['Week_of_month'] = (data['Date'].dt.day
                         - data['Date'].dt.weekday - 2) // 7 + 2
data['Weekday_order'] = (data['Date'].dt.day + 6) // 7
# Set the 'Date' itself as the index for better readability
data = data.set_index('Date')
data.head()

# %% [markdown]
# ### **Train and test split:**
# Split the preprocessed dataset with a ratio of 70:30 with the ``Observed`` column values as the target variables.

# %%
x = data.drop(['Observed'], axis=1)
y = data['Observed']  # Target variable

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# %% [markdown]
# ### **Train and test the data:**
# We used *RandomForestClassifier* because this model often results in higher accuracy compared to individual decision trees, as it reduces overfitting and variance. This is because it builds multiple trees and averages their predictions, which helps generalize well to unseen data.
# 
# But first, we do GridSearch for the best parameters to get high accuracy.

# %%
# GridSearch parameters:
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Model definition:
model = RandomForestClassifier(random_state=42)

# Instantiate GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
# Fit the grid search to the data
grid_search.fit(x_train, y_train)
# Print the best parameters found
print("Best Parameters:", grid_search.best_params_)

# Get the best model
best_model = grid_search.best_estimator_
# Predict using the best model
rf_pred = best_model.predict(x_test)
# Print classification report
print(classification_report(y_test, rf_pred))
report = classification_report(y_test, rf_pred, output_dict=True)

# %% [markdown]
# **Confusion matrix on the training result:**

# %%

rf_matrix = metrics.confusion_matrix(rf_pred, y_test)
tn, fp, fn, tp = rf_matrix.ravel()
print(f"\nConfusion Matrix:")
print(rf_matrix)
print(f"\nTP: {tp}")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")

# %% [markdown]
# ### **Predict future date:**
# We now create DataFrame with future dates for prediction and use our trained RandomForest model to predict future UFO sightings for one year ahead.

# %%
x_predict = pd.DataFrame(pd.date_range(date.today(), (date.today() +
            relativedelta(years=year_input)),freq='d'), columns=['Date'])

x_predict['Month'] = x_predict['Date'].dt.month
x_predict['Day'] = x_predict['Date'].dt.day
x_predict['Workday_N'] = np.busday_count(
                x_predict['Date'].values.astype('datetime64[M]'),
                x_predict['Date'].values.astype('datetime64[D]'))
x_predict['Week_day'] = x_predict['Date'].dt.weekday
x_predict['Week_of_month'] = (x_predict['Date'].dt.day -
                              x_predict['Date'].dt.weekday - 2)//7+2
x_predict['Weekday_order'] = (x_predict['Date'].dt.day + 6) // 7

x_predict = x_predict.set_index('Date')
prediction = best_model.predict(x_predict)
prediction = pd.DataFrame(prediction, columns=['Prediction'])
prediction['Date'] = x_predict.index

next_sightings = prediction[prediction['Prediction'] == 1]
next_sightings = next_sightings.reset_index()
next_sightings = next_sightings.drop(columns=['index'])

# %%
# display output in UI
st.write(next_sightings['Date'])
acc_message = f"""<style>p.a {{color:White; font-size: 15px;}}</style><p class="a">Note: Accuracy is {report['accuracy']} for predicting UFO sighting dates</p>"""
st.markdown(acc_message, unsafe_allow_html=True)

# %% [markdown]
# ### **Question 2**

# %%
q1 = '<p style="color:White; font-size: 30px;">Where in California will the next UFO sighting be?</p>'
st.markdown(q1, unsafe_allow_html=True)

# %% [markdown]
# ### **Load the Data & Filter for California Datapoints**

# %%
ufo = df
ufo.head()

# %%
# Filter for datapoints for California sightings
ufo = ufo[(ufo['state'] == 'ca') & (ufo['country'] == 'us')]
# Drop Nan
ufo.dropna(inplace=True)
# Rename columns for clarity
ufo = ufo.rename(columns = {'duration (seconds)': 'duration_second',
    'date posted': 'date_posted', 'longitude ': 'longitude'})

# %% [markdown]
# ### **Feature Engineering**

# %%
# Split Datetime into New Columns
ufo['datetime'] = pd.to_datetime(ufo['datetime'], errors='coerce')
ufo.replace([np.inf, -np.inf], np.nan, inplace=True)
ufo.dropna(inplace=True)

ufo['datetime_month'] = ufo['datetime'].dt.month.astype(int)
ufo['datetime_day'] = ufo['datetime'].dt.day.astype(int)
ufo['datetime_year'] = ufo['datetime'].dt.year.astype(int)
ufo['datetime_hour'] = ufo['datetime'].dt.hour.astype(int)
ufo['datetime_min'] = ufo['datetime'].dt.minute.astype(int)

# %% [markdown]
# ### **Remove Columns Not Used**

# %%
# Drop the original 'datetime', 'date_posted', 'comments', 'duration (hours/min)' columns
ufo = ufo.drop(columns=['datetime','date_posted','comments','duration (hours/min)', 'state', 'country'], axis=1)

# %% [markdown]
# ### **Reducing Shape Column Complexity**

# %%
#Visualizing the most frequently observed UFO shape
plt.figure(figsize=(11, 6))
sns.countplot(y='shape', data=ufo, palette='viridis')
plt.xlabel('UFO Shape')
plt.ylabel('Count')
plt.title('Types of UFO Seen')
plt.show()

# %%
# Reduce Shape Complexity By Grouping Values
print(ufo['shape'].unique())
shape = {
    'round': ['circle', 'disk', 'sphere', 'round'],
    'oval': ['egg', 'oval'],
    'triangular': ['triangle', 'cone'],
    'rectangular': ['rectangle', 'diamond'],
    'light': ['light', 'flash', 'flare'],
    'teardrop': ['teardrop', 'fireball'],
    'cylindrical': ['cylinder', 'cigar'],
    'cross': ['cross'],
    'chevron': ['chevron'],
    'other': ['other', 'unknown'],
    'other2':['formation', 'changing']     
}

for ufo_shape, ali in shape.items():
    ufo.loc[ufo['shape'].isin(ali), 'ufo_shape'] = ufo_shape

#Drop original column
ufo.drop(columns=['shape'], inplace=True)
ufo.head()

# %% [markdown]
# ### **Visualizing Features**

# %%
#Convert type columns from object to float
ufo['latitude'] = ufo['latitude'].astype('float64')
ufo['duration_second'] = ufo['duration_second'].astype('float64')

#Create list with columns dtype object
object_col = [x for x in ufo.columns if ufo[x].dtype == 'object']

# %%
# Correlation columns
plt.figure(figsize=(12, 5))
sns.heatmap(ufo.select_dtypes(include=['int64', 'float64']).corr(), annot=True)
plt.title('Correlation with Numerical Columns')
plt.show()

# %%
#Plot histograms of columns
ufo.hist(figsize=(15, 15))
plt.show()

# %%
hold = ufo.plot(
    kind='box', 
    subplots=True, 
    sharey=False, 
    figsize=(10, 6)
)
 
# increase spacing between subplots
plt.subplots_adjust(wspace=2) 
plt.show()

# %% [markdown]
# ### **Encoding Columns**

# %%
# Encoding Object Columns
ufo_temp = ufo.copy()
object_columns = list(ufo_temp.select_dtypes(include='object'))    
le = LabelEncoder()
for col in object_columns:
  ufo_temp[col] = le.fit_transform(ufo_temp[col])
ufo_temp.head()

# %% [markdown]
# ### **Split Data into Training & Testing sets**

# %%
X = ufo_temp.drop(columns=['latitude', 'longitude'], axis=1)
y_latitude = ufo_temp['latitude']
y_longitude = ufo_temp['longitude']

# %%
# Split the dataset into training and testing sets
X_train, X_test, y_train_latitude, y_test_latitude, y_train_longitude, y_test_longitude = train_test_split(X, y_latitude, y_longitude, test_size=0.2, random_state=42)
mms = MinMaxScaler()
X = mms.fit_transform(X)

# %% [markdown]
# ### **Model 1: Gradient Boosting Regressor**

# %%
# Train the models
gb_latitude = GradientBoostingRegressor()
gb_latitude.fit(X_train, y_train_latitude)

gb_longitude = GradientBoostingRegressor()
gb_longitude.fit(X_train, y_train_longitude)

# Make predictions
y_pred_latitude_hgb = gb_latitude.predict(X_test)
y_pred_longitude_hgb = gb_longitude.predict(X_test)

# %%
# Cross Validation
cv_results = cross_val_score(gb_latitude, X_train, y_train_latitude, cv=5, scoring='r2')
print("Latitude Cross Validation Scores: ", cv_results)
print("Average CV Score: ", cv_results.mean())
cv_results = cross_val_score(gb_longitude, X_train, y_train_longitude, cv=5, scoring='r2')
print("Longitude Cross Validation Scores: ", cv_results)
print("Average CV Score: ", cv_results.mean())

# %% [markdown]
# ### **Model 2: Histogram Gradient Boosting Regressor**

# %%
# Train the models
hgb_latitude = HistGradientBoostingRegressor()
hgb_latitude.fit(X_train, y_train_latitude)

hgb_longitude = HistGradientBoostingRegressor()
hgb_longitude.fit(X_train, y_train_longitude)

# Make predictions
y_pred_latitude_hgb = hgb_latitude.predict(X_test)
y_pred_longitude_hgb = hgb_longitude.predict(X_test)

# %%
# Cross Validation
cv_results = cross_val_score(hgb_latitude, X_train, y_train_latitude, cv=5, scoring='r2')
print("Latitude Cross Validation Scores: ", cv_results)
print("Average CV Score: ", cv_results.mean())
cv_results = cross_val_score(hgb_longitude, X_train, y_train_longitude, cv=5, scoring='r2')
print("Longitude Cross Validation Scores: ", cv_results)
print("Average CV Score: ", cv_results.mean())

# %% [markdown]
# ### **Model 3: Random Forest Regressor**

# %%
rf_latitude = RandomForestRegressor()
rf_latitude.fit(X_train, y_train_latitude)

rf_longitude = RandomForestRegressor()
rf_longitude.fit(X_train, y_train_longitude)

# Make predictions
y_pred_latitude_rf = rf_latitude.predict(X_test)
y_pred_longitude_rf = rf_longitude.predict(X_test)

# %%
# Cross Validation
cv_results = cross_val_score(rf_latitude, X_train, y_train_latitude, cv=5, scoring='r2')
print("Latitude Cross Validation Scores: ", cv_results)
print("Average CV Score: ", cv_results.mean())
cv_results = cross_val_score(rf_longitude, X_train, y_train_longitude, cv=5, scoring='r2')
print("Longitude Cross Validation Scores: ", cv_results)
print("Average CV Score: ", cv_results.mean())

# %% [markdown]
# After reviewing the cross validation scores from each model, we chose model 3, the random forest regressor, as our final model for question 2.

# %%
# Evaluate the performance
mae_latitude_rf = mean_absolute_error(y_test_latitude, y_pred_latitude_rf)
mse_latitude_rf = mean_squared_error(y_test_latitude, y_pred_latitude_rf)
r2_latitude_rf = r2_score(y_test_latitude, y_pred_latitude_rf)

mae_longitude_rf = mean_absolute_error(y_test_longitude, y_pred_longitude_rf)
mse_longitude_rf = mean_squared_error(y_test_longitude, y_pred_longitude_rf)
r2_longitude_rf = r2_score(y_test_longitude, y_pred_longitude_rf)

print(f"mean_absolute_error latitude: {mae_latitude_rf}")
print(f"mean_squared_error latitude: {mse_latitude_rf}")
print(f"r2_score latitude: {r2_latitude_rf}\n")
print(f"mean_absolute_error longitude: {mae_longitude_rf}")
print(f"mean_squared_error longitude: {mse_longitude_rf}")
print(f"r2_score longitude: {r2_longitude_rf}")

# %% [markdown]
# ### **Optimizing Parameters**

# %%
param_grid = { 
    'n_estimators': [100, 225, 350],
    'max_features': ['sqrt', 'log2', None], 
} 

# Create a random forest classifier
rf = RandomForestRegressor()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf, param_distributions=param_grid)

# Fit the random search object to the data
rand_search.fit(X_train, y_train_latitude)

# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

y_pred = best_rf.predict(X_test)
r2 = r2_score(y_test_latitude, y_pred)
print("r2 latitude:", r2)

# Fit the random search object to the data
rand_search.fit(X_train, y_train_longitude)

# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

y_pred = best_rf.predict(X_test)
r2 = r2_score(y_test_longitude, y_pred)
print("r2 longitude:", r2)

# %%
# Cross Validation
cv_results = cross_val_score(best_rf, X_train, y_train_latitude, cv=5, scoring='r2')
print("Cross Validation Scores: ", cv_results)
print("Average CV Score: ", cv_results.mean())

# %% [markdown]
# ### **Prediction Results**

# %%
df_pred_lat = pd.DataFrame({'predict_latitude_reg': y_pred_latitude_rf})
df_pred_lon = pd.DataFrame({'predict_longitude_reg': y_pred_longitude_rf})

ufo_temp['predict_latitude_ufo'] = df_pred_lat['predict_latitude_reg']
ufo_temp['predict_longitude_ufo'] = df_pred_lon['predict_longitude_reg']

# %%
plt.style.use('dark_background')
states = gdp.read_file('cb_2018_us_state_500k/cb_2018_us_state_500k.shp') 
states[states['NAME'] == 'California'].plot(figsize=(12, 12),color='#DFEEDA')
#plt.figure(figsize=(14, 8))
plt.scatter(ufo_temp['longitude'], ufo_temp['latitude'], color='pink', label='Real Data', s=3) ##9cc763
mask = ~ufo_temp['predict_latitude_ufo'].isnull() & ~ufo_temp['predict_longitude_ufo'].isnull()
plt.scatter(ufo_temp.loc[mask, 'predict_longitude_ufo'], ufo_temp.loc[mask, 'predict_latitude_ufo'], color='blue', label='Predictions', s=3)
plt.title('Real UFO Data vs Predictions', fontsize='18')
plt.xlabel('Longitude', fontsize='15')
plt.ylabel('Latitude', fontsize='15')
plt.xlim(-125, -113)
plt.ylim(32, 43)
plt.legend(fontsize="15",markerscale=3)
plt.show()

# %%
# Display Map in UI
st.pyplot(plt.gcf())
r2_message = f"""<style>p.a {{color:White; font-size: 15px;}}</style><p class="a">Note: R^2 is {r2_latitude_rf} for latitude and {r2_longitude_rf} for longitude</p>"""
st.markdown(r2_message, unsafe_allow_html=True)


