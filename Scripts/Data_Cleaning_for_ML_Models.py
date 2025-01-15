# %%
import pandas as pd
import numpy as np
import summarytools
import mysql.connector


# %% [markdown]
# # Inspect Data

# %%
# Connect with mysql database
connection = mysql.connector.connect(
    host='localhost',      # Replace with your database host
    user='root',  # Replace with your username
    password='1234',  # Replace with your password
    database='house'   # Replace with your database name
)

query = "SELECT * FROM house"
df = pd.read_sql(query, connection)
connection.close()


summarytools.dfSummary(df)

# %% [markdown]
# # Data Cleasing

# %%
df = df.drop(columns = ['address', 'agent_name', 'agent_url', 'house_name', 'description', 'house_url', 'maximum_months', 'minimum_months', 'status'])

# %%
df.info()

# %%
print(df['balcony'].unique()) # fillna with not present
print(df['construction_type'].unique()) # fillna with mode
print(df['dwelling_type'].unique())
print(df['interior'].unique()) # fillna using relationship
# print(df['property_type'].unique())
print(df['rental_agreement'].unique()) # fillna with not specified
print(df['smoking_allowed'].unique())
print(df['pets_allowed'].unique())

# %%
df['balcony'] = df['balcony'].fillna('Not present')
df['construction_type'] = df['construction_type'].fillna(df['construction_type'].mode()[0])
df['interior'] = df['interior'].fillna('Not specified')
df['rental_agreement'] = df['rental_agreement'].fillna('Not specified')
df['pets_allowed'] = df['pets_allowed'].fillna('No')
df['smoking_allowed'] = df['smoking_allowed'].fillna('No')

# %%
# Impute number_of_bedrooms and number_of_bathrooms
median_bathrooms = int(df['number_of_bathrooms'].median())

# Imputation logic
def ImputeBedrooms(row):
    if pd.isnull(row['number_of_bedrooms']):
        # Set to 1 if dwelling type is 'room'
        if row['dwelling_type'] == 'room':
            return 1  
        elif not pd.isnull(row['number_of_rooms']) and not pd.isnull(row['number_of_bathrooms']):
            # If both rooms and bathrooms are not null 
            if row['number_of_rooms'] - row['number_of_bathrooms'] <= 0:
                return row['number_of_rooms'] 
            else:
                return row['number_of_rooms'] - row['number_of_bathrooms']
        elif not pd.isnull(row['number_of_rooms']):
            return row['number_of_rooms']
    return row['number_of_bedrooms']

def ImputeBathrooms(row):
    if pd.isnull(row['number_of_bathrooms']):
        return median_bathrooms 
    return row['number_of_bathrooms']

def TransformRooms(row):
    if row['dwelling_type'] != 'room':
        rooms = row['number_of_bathrooms'] + row['number_of_bedrooms']
        return rooms
    else:
        return 1

# Apply the logic
df['number_of_bedrooms'] = df.apply(ImputeBedrooms, axis=1)
df['number_of_bathrooms'] = df.apply(ImputeBathrooms, axis=1)
df['number_of_rooms'] = df.apply(TransformRooms, axis = 1)

df['number_of_bathrooms'] = df['number_of_bathrooms'].astype('int32')
df['number_of_bedrooms'] = df['number_of_bedrooms'].astype('int32')
df['number_of_rooms'] = df['number_of_rooms'].astype('int32')

# %%
# service cost category column
def CategorizeServiceCost(value):
    if pd.isnull(value) or value.lower() == 'none':  # Handle NaN or 'None'
        return 'Not specified'
    value = value.lower()
    if 'electricity' in value and 'gas' in value and 'water' in value and 'internet' in value:
        return 'All included'
    elif 'electricity' in value or 'gas' in value or 'water' in value or 'internet' in value:
        return 'Partial included'
    elif 'excludes' in value:
        return 'Not included'
    else:
        return 'Not specified'

df['service_cost_category'] = df['service_cost'].astype(str).apply(CategorizeServiceCost)

# %% [markdown]
# ## Transform date columns

# %%
from datetime import datetime, timedelta
def StringToDate(value):
    if not pd.isnull(value):
        date = datetime.strptime(value,'%d-%m-%Y')
        return date

df['available'] = df['available'].apply(StringToDate)
df['offered_since'] = df['offered_since'].apply(StringToDate)

df['offer_to_available'] = (df['available'] - df['offered_since']).dt.days

median_duration = int(df['offer_to_available'].median())
df['offer_to_available'] = df['offer_to_available'].fillna(median_duration)

def AvailableFillNa(row):
    if pd.isnull(row['available']):
        date = row['offered_since'] + timedelta(days = median_duration)
        return date
    return row['available']

df['available'] = df.apply(AvailableFillNa, axis=1)

# extract month
month_dic = {1:'Jan.', 2: 'Feb.', 3: 'Mar.', 4: 'Apr.', 5: 'May.', 6: 'Jun.', 7: 'Jul', 8: 'Aug.', 9: 'Sep.', 10: 'Oct', 11: 'Nov.', 12: 'Dec'}
def GetMonth(value):
    return month_dic[value.month]

df['offered_month'] = df['offered_since'].apply(GetMonth)
df['available_month'] = df['available'].apply(GetMonth)



# house age
df['house_age'] = df['offered_since'].dt.year - df['year_of_construction']
df['house_age'] = df['house_age'].fillna(df['house_age'].median())
df = df.drop(columns = ['year_of_construction'])

df = df.drop(columns = ['available','offered_since'])


# %% [markdown]
# ## Impute energy_rating

# %%
def ToIntegerScale(value):
    if value is not None:
        if value == 'G':
            return 1
        elif value == 'F':
            return 2
        elif value == 'E':
            return 3
        elif value == 'D':
            return 4
        elif value == 'C':
            return 5
        elif value == 'B':
            return 6
        elif value == 'A':
            return 7
        elif '+' in value:
            lis = list(value)
            extra = lis.count('+')
            return 7+extra
    return value

df['energy_rating'] = df['energy_rating'].apply(ToIntegerScale).astype('Int32')


# %%
train_data = df[df['energy_rating'].notna()]
correlation = train_data[['energy_rating', 'house_age', 'living_area_m2']].corr()
print(f"Correlation for house age and living_area_m2: {correlation}")

from scipy.stats import f_oneway

# ANOVA for city
anova_city = f_oneway(*(train_data[train_data['city'] == city]['energy_rating'] for city in train_data['city'].unique()))
print("ANOVA for City:", anova_city)

# ANOVA for dwelling_type
anova_dwelling = f_oneway(*(train_data[train_data['dwelling_type'] == dwelling_type]['energy_rating'] for dwelling_type in train_data['dwelling_type'].unique()))
print("ANOVA for Dwelling Type:", anova_dwelling)

# %%
from sklearn.ensemble import RandomForestClassifier

# Train data (rows without missing energy_rating)
train_data = df[df['energy_rating'].notna()]
X_train = train_data[['dwelling_type', 'city']]
y_train = train_data['energy_rating']

# Test data (rows with missing energy_rating)
test_data = df[df['energy_rating'].isna()]
X_test = test_data[['dwelling_type', 'city']]

# One-hot encoding for categorical features
# One-hot encoding for X_train
X_train_encoded = pd.get_dummies(X_train, drop_first=True)

# One-hot encoding for X_test
X_test_encoded = pd.get_dummies(X_test, drop_first=True)

# Align the columns of X_test_encoded with X_train_encoded
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# Train model and predict
model = RandomForestClassifier()
model.fit(X_train_encoded, y_train)
df.loc[df['energy_rating'].isna(), 'energy_rating'] = model.predict(X_test_encoded)


# %% [markdown]
# ## Impute deposit

# %%
train_data = df[df['deposit'].notna()]
correlation = train_data[['deposit','price']].corr()
print(f"Correlation for deposit and price: {correlation}")

# ANOVA for city
anova_city = f_oneway(*(train_data[train_data['city'] == city]['deposit'] for city in train_data['city'].unique()))
print("ANOVA for City:", anova_city)

# ANOVA for dwelling type
anova_dwelling = f_oneway(*(train_data[train_data['dwelling_type'] == dwelling_type]['deposit'] for dwelling_type in train_data['dwelling_type'].unique()))
print("ANOVA for Dwelling_type:", anova_dwelling)


# %%
# Impute deposit by city and dwelling_type
df['deposit'] = df['deposit'].fillna(
    df.groupby(['city', 'dwelling_type'])['deposit'].transform('median')
)

df['deposit'] = df['deposit'].fillna(df['deposit'].median())


# %% [markdown]
# ## Inspect multicollinearity

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Filter numerical columns
numerical_features = df.select_dtypes(include=['float64', 'int32'])

# Calculate the correlation matrix
correlation_matrix = numerical_features.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("Correlation Heatmap")
plt.show()


# %%
from scipy.stats import chi2_contingency

def cramers_v(contingency_table):
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    return np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

# Function to calculate Cramér's V matrix
def cramers_v_matrix(df, categorical_features):
    matrix = pd.DataFrame(index=categorical_features, columns=categorical_features)
    for col1 in categorical_features:
        for col2 in categorical_features:
            contingency_table = pd.crosstab(df[col1], df[col2])
            matrix.loc[col1, col2] = cramers_v(contingency_table)
    return matrix.astype(float)

# List of categorical features
categorical_features = ['balcony', 'city', 'construction_type',
       'district', 'dwelling_type', 'interior',
       'pets_allowed', 'property_type', 'rental_agreement', 'service_cost', 'smoking_allowed',
       'service_cost_category', 'offered_month', 'available_month']

# Compute Cramér's V matrix
cramers_v_matrix_df = cramers_v_matrix(df, categorical_features)

# Ensure diagonal elements are 1
np.fill_diagonal(cramers_v_matrix_df.values, 1)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cramers_v_matrix_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Cramér's V Heatmap for Categorical Features")
plt.show()


# %%
df = df.drop(columns = ['property_type', 'district', 'service_cost', 'number_of_bedrooms'])

# %% [markdown]
# ## Reduce dimensionality of data

# %%
categorical_features = ['balcony', 'city', 'construction_type', 'dwelling_type', 
                        'interior', 'pets_allowed', 'rental_agreement', 
                        'smoking_allowed','service_cost_category', 'offered_month', 
                        'available_month']

threshold = 0.05 * len(df)  # Adjust threshold based on dataset size
for col in categorical_features:
    rare_categories = df[col].value_counts()[df[col].value_counts() < threshold].index
    df[col] = df[col].replace(rare_categories, 'Other')

# %% [markdown]
# ## One-Hot Encoding

# %%
df.columns

# %%
from sklearn.preprocessing import OneHotEncoder
df = pd.get_dummies(df, columns = ['balcony','city','construction_type', 'dwelling_type', 'interior', 'pets_allowed', 'rental_agreement',
                                   'smoking_allowed', 'service_cost_category', 'offered_month', 'available_month'])
df.head()

# %% [markdown]
# ## Inspect distribution

# %%
df.info()

# %%
# Select numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int32']).columns

import math
import matplotlib.pyplot as plt

# Calculate the number of rows needed (2 plots per row)
num_cols = 2
num_rows = math.ceil(len(numeric_columns) / num_cols)

# Create the subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 4 * num_rows))
# Flatten the axes array for easy iteration
axes = axes.flatten()  

# Loop through numeric columns and plot
for i, column in enumerate(numeric_columns):
    axes[i].hist(df[column].dropna(), bins=20, edgecolor='k', alpha=0.7)
    axes[i].set_xlabel(column, fontsize=12)
    axes[i].set_ylabel('Frequency', fontsize=12)
    axes[i].set_title(f'Distribution of {column}', fontsize=14)
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    print(f"{column} - min: {df[column].min()}, max: {df[column].max()}")

# Turn off any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()  # Adjust spacing
plt.show()



# %%
df.to_csv('Cleaned Data for ML.csv', index = False)


