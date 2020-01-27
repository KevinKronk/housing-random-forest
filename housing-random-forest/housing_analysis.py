import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


# Load the Housing File
filename = "housing.csv"


def load_housing_data(filename):
    housing = pd.read_csv(filename)
    return housing


housing = load_housing_data(filename)


# Gain Insight into the dataset
# print(housing.info())
# print("-----")
# print(housing['ocean_proximity'].value_counts())
# print("-----")
# print(housing.describe())
# print(f"-----\n")

# Create new category for stratified sampling
# Divide by 1.5 to get a distribution from 0 to 10. Ceiling so values fall in definitive categories
housing["income_cat"] = np.ceil(housing['median_income'] / 1.5)
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)

# Split data into train and test sets
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.iloc[train_index]
    strat_test_set = housing.iloc[test_index]

# Ratio of total values in each income category
print(housing['income_cat'].value_counts() / len(housing))
print(train_set['income_cat'].value_counts() / len(train_set))
print(strat_train_set['income_cat'].value_counts() / len(strat_train_set))
print(f"\n-----\n")

# remove 'income_cat' so data is back to its original state
strat_train_set = strat_train_set.drop(columns=['income_cat'])
strat_test_set = strat_test_set.drop(columns=['income_cat'])


# Plot the columns in the dataset
strat_train_set.hist(bins=50, figsize=(20, 15))
plt.show()


# Make copy to explore it without harming training set
housing = strat_train_set.copy()


housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
             cmap=plt.get_cmap('jet'), colorbar=True, c='median_house_value',
             s=housing['population']/100, label='population', figsize=(10, 7))
plt.legend()
plt.show()

corr_matrix = housing.corr()
print(f"Correlation with Median House Value\n")
print(corr_matrix['median_house_value'].sort_values(ascending=False))
print(f"\n-----\n")

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()

# Perhaps some more specific values would be important
housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
housing['population_per_household'] = housing['population'] / housing['households']

corr_matrix = housing.corr()
print(f"New Correlation with Median House Value\n")
print(corr_matrix['median_house_value'].sort_values(ascending=False))
print(f"\n-----\n")


# Separate the predictors and the target values (drop creates a copy)
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()


# Custom Transformers
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)

    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self

    def transform(self, x, y=0):
        return self.encoder.transform(x)


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


housing_num = housing.drop('ocean_proximity', axis=1)

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', MyLabelBinarizer()),
])

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
])

housing_prepared = full_pipeline.fit_transform(housing)


# Make Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# Choose a few data points to make predictions
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print('Predictions:', [int(x) for x in lin_reg.predict(some_data_prepared)])
print('Labels:', [int(x) for x in list(some_labels)])
print(f"\n-----\n")


# Compare the rmse of Linear Regression to Decision Tree
# Linear Regression
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(f"Linear Regression RMSE: {lin_rmse}")

# Decision Tree
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print(f"Decision Tree RMSE: {tree_rmse}")
print(f"\n-----\n")


# Show Decision Tree Cross Validation Scores
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-scores)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Std:", scores.std())
    print(f"\n-----\n")


# Decision Tree
print("Decision Tree Scores")
display_scores(tree_rmse_scores)

# Show Lin Regression Cross Validation Scores
scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                         scoring='neg_mean_squared_error', cv=10)
lin_rmse_scores = np.sqrt(-scores)
print("Linear Regression Scores")
display_scores(lin_rmse_scores)


# Random Forest with Cross Validation Scores
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                         scoring='neg_mean_squared_error', cv=10)
forest_rmse_scores = np.sqrt(-scores)
print("Random Forest Scores")
display_scores(forest_rmse_scores)


# Grid Search Random Forest
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)
print("Random Forest Grid Search")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Estimator: {grid_search.best_estimator_}")
print(f"\n-----\n")

# Scores for every set of params
# cvres = grid_search.cv_results_
# for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
#     print(np.sqrt(-mean_score), params)


# Analyze model and its error
feature_importances = grid_search.best_estimator_.feature_importances_
# Add in feature names
extra_attribs = ['rooms_per_hhold', 'pop_per_hhold', 'bedrooms_per_room']
attributes = num_attribs + extra_attribs
print("Feature Importances\n")
print(sorted(zip(feature_importances, attributes), reverse=True))
print(f"\n-----\n")


# Take the best model, run through pipeline, get final rmse
final_model = grid_search.best_estimator_

x_test = strat_test_set.drop('median_house_value', axis=1)
y_test = strat_test_set['median_house_value'].copy()

x_test_prepared = full_pipeline.transform(x_test)
final_predictions = final_model.predict(x_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print(f"Randon Forest Best Model RMSE: {final_rmse}")
