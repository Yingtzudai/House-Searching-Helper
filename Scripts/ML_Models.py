# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('Cleaned Data for ML.csv')

# %%
df.info()

# %% [markdown]
# # Split and transform data

# %%
# Split the data for the common use
from sklearn.model_selection import train_test_split
X = df.drop(columns = ['price']).copy()
y = df['price'].copy()
# split data to train + test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

# %%
from sklearn.preprocessing import StandardScaler
import numpy as np

# Step 1: Log-transform specific numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).drop(
    columns=['energy_rating', 'price', 'number_of_bathrooms', 'number_of_rooms', 'house_age']
).columns

for c in numeric_columns:
    X_train[c] = np.log1p(X_train[c])  # Use log1p to handle zero values safely
    X_test[c] = np.log1p(X_test[c])

# Step 2: Shift and log-transform 'house_age'
dfs = {'X_train': X_train, 'X_test': X_test}  # Use dictionary for better management
shift_age_value = abs(X_train['house_age'].min()) + 1  # Calculate shift value from training data

for name, d in dfs.items():
    d['house_age'] = np.log1p(d['house_age'] + shift_age_value)  # Apply consistent transformation

# Step 3: Transform 'number_of_rooms' with square root and scale
scaler_rooms = StandardScaler()  # Initialize one scaler for consistent scaling
X_train['number_of_rooms'] = scaler_rooms.fit_transform(np.sqrt(X_train[['number_of_rooms']]))
X_test['number_of_rooms'] = scaler_rooms.transform(np.sqrt(X_test[['number_of_rooms']]))

# Step 4: Transform target variable (if needed for modeling)
y_train = np.log1p(y_train)  # Log-transform the target variable
y_test = np.log1p(y_test)



# %%
dfs = {'X_train': X_train, 'X_test': X_test}  # Use a dictionary to associate names with DataFrames
numeric_columns = X_train.select_dtypes(include=['float64', 'int64']).drop(columns=['energy_rating']).columns

for name, d in dfs.items():  # Iterate through the dictionary
    for column in numeric_columns:
        plt.figure(figsize=(4, 3))  # Set figure size
        plt.hist(d[column].dropna(), bins=20, edgecolor='k', alpha=0.7)  # Drop NaNs and plot histogram
        plt.xlabel(column, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Distribution of {name}_{column}', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)  # Optional: Add gridlines
        plt.show()


# %% [markdown]
# # ML models

# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import shap

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from sqlalchemy import create_engine
import pandas as pd

def StoreTableToMysql(dataframe, table_name, if_exists='replace'):
    try:
        # Create a connection to the MySQL database
        engine = create_engine('mysql+mysqlconnector://root:1234@localhost/house')

        # Ensure table name is valid and clean
        table_name = table_name.replace(' ', '_').lower()

        # Store the DataFrame into the MySQL table
        dataframe.to_sql(name=table_name, con=engine, if_exists=if_exists, index=False)

        print(f"Data successfully stored in the '{table_name}' table in the 'house' database.")
    except Exception as e:
        print(f"Error: {e}")




def result(alg_model, X_test, y_test):
    # Predict using the model
    y_pred = alg_model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    p = len(X_test.columns)

    # Create a DataFrame with results
    results = pd.DataFrame([{
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'R-Squared': r2
    }])
    
    display(results)

    model_name = alg_model.__class__.__name__  # Get the class name of the model
    StoreTableToMysql(results, f"{model_name}_Evaluation")
    # results.to_csv(f"{model_name}_Evaluation.csv", index=False)


  # print('Mean Absolute Error: {}\n\nMean Squared Error:\n {}\n\nR-Squared: {}\n\nAdjusted R-Squared:\n {}'
  #     .format(mean_absolute_error(y_test,y_pred), mean_squared_error(y_test,y_pred),r2_score(y_test, y_pred),adj_r2))
def gridCVModel(algorithm, X_train, y_train, grid):
  # https://www.projectpro.io/recipes/find-optimal-parameters-using-randomizedsearchcv-for-regression
  # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
  # https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py
  scoring = {'Negative Mean Absolute Error':'neg_mean_absolute_error','Negative Mean Squared Error':'neg_mean_squared_error', 'R-Squared':'r2'}
  randm_src = RandomizedSearchCV(estimator=algorithm, param_distributions = grid, cv = 5, n_iter = 10, n_jobs=-1,
                                 scoring = scoring,refit='Negative Mean Absolute Error', random_state = 42)
  randm_src.fit(X_train, y_train)

  print(" Results from Random Search " )
  print("\n The best estimator across ALL searched params:\n", randm_src.best_estimator_)
  print("\n The best score across ALL searched params:\n", randm_src.best_score_)
  print("\n The best parameters across ALL searched params:\n", randm_src.best_params_)
  # Calculate and print scores for the best estimator
  best_estimator = randm_src.best_estimator_
  for score_name, score_func in scoring.items():
      score = cross_val_score(best_estimator, X_train, y_train, cv=5, scoring=score_func)
      print(f"\n {score_name} score for the best estimator: {score.mean()}")


def model(algorithm, X_train, X_test, y_train, y_test):
  # Evaluate the model
  alg_model = algorithm.fit(X_train, y_train)
  result(alg_model, X_test, y_test)
  parameters = alg_model.get_params()
  print(f"Parameters of the model: {parameters}")

  # Store prediction to mysql
  predictions = alg_model.predict(X_test)
  predictions_df = pd.DataFrame({'Predicted': predictions, 'Actual':y_test.values})
  model_name = alg_model.__class__.__name__
  StoreTableToMysql(predictions_df, f"Predictions_{model_name}")

# %% [markdown]
# ## Linear Regression

# %%
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
print('Linear Regression Model \n')
model(reg, X_train, X_test, y_train, y_test)

# %% [markdown]
# ## Decision Tree

# %%
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(random_state = 42)

# The maximum depth of the tree.
max_depth = np.arange(3, 11, 1).tolist()
# The minimum number of samples required to split an internal node:
min_samples_split = [5, 10, 20]
# The minimum number of samples required to be at a leaf node
min_samples_leaf = [5, 10, 20]
# The number of features to consider when looking for the best split:
max_features = [None, 'sqrt', 'log2']
# A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
min_impurity_decrease = [0.01, 0.05]
# Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
max_leaf_nodes = [10, 20, 30]
# The function to measure the quality of a split.
criterion = ['squared_error', 'friedman_mse']


random_grid = {'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'max_features': max_features,
               'min_impurity_decrease': min_impurity_decrease,
               'max_leaf_nodes': max_leaf_nodes,
               'criterion': criterion}

gridCVModel(tree, X_train, y_train, random_grid)

# %%
print("Decision Tree Test Performance:\n")
tree_tuned = DecisionTreeRegressor(random_state = 42, min_samples_split= 5, min_samples_leaf = 5, min_impurity_decrease= 0.01, 
                                   max_leaf_nodes= 20, max_features= None, max_depth= 5, criterion = 'friedman_mse' )
model(tree_tuned, X_train, X_test, y_train, y_test)

# %% [markdown]
# ## Random Forest

# %%
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 42)

# Number of trees
n_estimators = [50, 100, 200, 500, 1000]
# tree depth
max_depth = [None, 5, 10, 20, 50]
# minimum samples to split
min_samples_split = [2, 5, 10, 20]
# minimum samples per leaf
min_samples_leaf = [1, 2, 5, 10]
# Features for splits
max_features = ['sqrt', 'log2', None, 0.5, 0.8]
# Bootstrapping
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split, 
               'min_samples_leaf': min_samples_leaf, 'max_features': max_features, 'bootstrap': bootstrap}

gridCVModel(rf, X_train, y_train, random_grid)

# %%
print('Random Forest Test Performance:\n')
rf_tuned = RandomForestRegressor(random_state= 42, n_estimators = 200, min_samples_split = 10, min_samples_leaf = 5,
                        max_features = 0.5, max_depth = 10, bootstrap = False)

model(rf_tuned, X_train, X_test, y_train, y_test)


# %% [markdown]
# ## Gradient Boosting Regressor

# %%
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(random_state = 42)

# The number of boosting stages (trees)
n_estimators = [50, 100, 200, 300, 500]
# Controls the contribution of each tree to the final prediction
learning_rate = [0.01, 0.05, 0.1, 0.2]
# The maximum depth of each decision tree
max_depth = [3, 5, 7, 10]
# Minimum number of samples required to be in a leaf node
min_samples_split = [2, 5, 10, 20]
# The number of features to consider when looking for the best split
max_features = [None, 'sqrt', 'log2', 0.5, 0.8]
# The fraction of samples used for training each tree
subsample = [0.6, 0.8, 1.0]

random_grid = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'max_depth': max_depth,
               'min_samples_split': min_samples_split, 'max_features': max_features, 'subsample': subsample}


gridCVModel(gbr, X_train, y_train, random_grid)


# %%
print('Gradient Boosting Regressor Test Performance:\n')
gbr_tuned = GradientBoostingRegressor(random_state= 42, subsample = 1, n_estimators = 100, min_samples_split = 2,
                                      max_features = 0.5, max_depth = 5, learning_rate = 0.05)

model(gbr_tuned, X_train, X_test, y_train, y_test)

# %% [markdown]
# ## Stacking Regressor

# %%
# Stacking with default models
from sklearn.ensemble import StackingRegressor

stacking_default = StackingRegressor(
    estimators=[
        ('reg', LinearRegression()),
        ('rf', RandomForestRegressor(random_state=42)), 
        ('tree', DecisionTreeRegressor(random_state=42)),
    ],
    final_estimator=GradientBoostingRegressor(random_state=42)
)

# Train and evaluate
print("Default Stacking Model Test Performance:\n")
model(stacking_default, X_train, X_test, y_train, y_test)

# %%
stacking_tuned = StackingRegressor(
    estimators=[
        ('reg', LinearRegression()),
        ('rf', rf_tuned), 
        ('tree', tree_tuned),
    ],
    final_estimator=gbr_tuned
)

# Train and evaluate
print("Tuned Stacking Model Test Performance:\n")
model(stacking_tuned, X_train, X_test, y_train, y_test)

# %% [markdown]
# # Interpret Models

# %% [markdown]
# ## Learning Curve for Each Model

# %%
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(model, X, y, model_name, cv=5, scoring='neg_mean_squared_error'):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, scoring=scoring, train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, label="Training Error")
    plt.plot(train_sizes, test_scores_mean, label="Validation Error")
    plt.title(f"Learning Curve of {model_name}", fontsize=16)
    plt.xlabel("Training Set Size")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

plot_learning_curve(tree_tuned, X_train, y_train, model_name="Decision Tree")
plot_learning_curve(rf_tuned, X_train, y_train, model_name="Random Forest")
plot_learning_curve(gbr_tuned, X_train, y_train, model_name="Gradient Boosting Regressor")
plot_learning_curve(stacking_tuned, X_train, y_train, model_name="Stacking Model")

# %% [markdown]
# ## SHAP for Gradient Boosting Regressor

# %%
import shap

# SHAP Analysis
explainer = shap.TreeExplainer(gbr_tuned)
shap_values = explainer.shap_values(X_test)

# Summary Plot
shap.summary_plot(shap_values, X_test)
# plt.savefig('Shap Plot of Gradient Boosting Regressor.png', dpi=300, bbox_inches='tight')
plt.close()



# %% [markdown]
# ## Permutation Feature Importance for Gradient Boosting Regressor

# %%
from sklearn.inspection import permutation_importance

models = [gbr_tuned, tree_tuned, rf_tuned, stacking_tuned]
for m in models:
    model_name = m.__class__.__name__
    perm_importance = permutation_importance(m, X_test, y_test, scoring='neg_mean_squared_error', random_state=42)
    # Create a DataFrame for visualization
    perm_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': perm_importance['importances_mean']
    }).sort_values(by='Importance', ascending=False)
    perm_df = perm_df[(perm_df['Importance'] > 0.001) | (perm_df['Importance'] < - 0.001)]
    StoreTableToMysql(perm_df, f"perm_importance_{model_name}")

    # Plot permutation importance with values
    plt.figure(figsize=(15, 9))
    bars = plt.barh(perm_df['Feature'], perm_df['Importance'])
    plt.xlabel('Permutation Importance')
    plt.title(f"Permutation Feature Importance of {model_name}")
    plt.gca().invert_yaxis()

    # Add importance values as text on the bars
    for bar, value in zip(bars, perm_df['Importance']):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{value:.4f}', va='center', fontsize=10)
    plt.show()





