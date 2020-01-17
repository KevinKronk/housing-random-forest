import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

# from sklearn.preprocessing import

filename = "housing.csv"


def load_housing_data(filename):
    housing = pd.read_csv(filename)
    return housing


housing = load_housing_data(filename)

# print(housing.info())
# print(housing['ocean_proximity'].value_counts())
# print(housing.describe())


# create new category for stratified sampling
# divide by 1.5 to get a better distribution, ceiling for definitive categories
housing["income_cat"] = np.ceil(housing['median_income'] / 1.5)
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# print(len(train_set), len(test_set))


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.iloc[train_index]
    strat_test_set = housing.iloc[test_index]

# print(len(strat_train_set), len(strat_test_set))
# print(housing['income_cat'].value_counts() / len(housing))

# print(train_set['income_cat'].value_counts() / len(train_set))
# print(strat_train_set['income_cat'].value_counts() / len(strat_train_set))

# remove 'income_cat' so data is back to its original state

strat_train_set = strat_train_set.drop(columns=['income_cat'])
strat_test_set = strat_test_set.drop(columns=['income_cat'])


# strat_train_set.hist(bins=50, figsize=(20, 15))
# plt.show()

# Make copy to explore it without harming training set
housing = strat_train_set.copy()

# housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
#              cmap=plt.get_cmap('jet'), colorbar=True, c='median_house_value',
#              s=housing['population']/100, label='population', figsize=(10, 7))
# plt.legend()
# plt.show()

corr_matrix = housing.corr()

print(corr_matrix["median_house_value"].sort_values(ascending=False))

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]

scatter_matrix(housing[attributes], figsize=(12, 8))
housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)

# plt.show()

housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
housing['population_per_household'] = housing['population'] / housing['households']

corr_matrix = housing.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))

# Separate the predictors and the target values (drop creates a copy)
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()

housing.dropna(subset=['total_bedrooms'])  # gets rid of NA data
housing.drop('total_bedrooms', axis=1)  # gets rid of entire feature
median = housing['total_bedrooms'].median()  # fills NA data with value
housing['total_bedrooms'].fillna(median, inplace=True)

imputer = SimpleImputer(strategy='median')

# Imputer only works for numerical data
housing_num = housing.drop('ocean_proximity', axis=1)
imputer.fit(housing_num)

print(imputer.statistics_)

# transforms the training set by replacing NA values with the learned medians
X = imputer.transform(housing_num)
# place back into Pandas DataFrame
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

# Convert text labels to numbers
encoder = LabelEncoder()
housing_cat = housing['ocean_proximity']
housing_cat_encoded = encoder.fit_transform(housing_cat)
print(housing_cat_encoded)
print(encoder.classes_)

# OneHotEncoding
encoder = OneHotEncoder()
housing_cat_one = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
print(housing_cat_one)
print(housing_cat_one.toarray())

# Performs both the text to integer conversion and the one hot conversion at once
# encoder =
# encoder = CategoricalEncoder(encoding='onehot')
# housing_cat_one = encoder.fit_transform(housing_cat)
# print(housing_cat_one)

# Custom Transformers
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


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


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

housing_num_tr = num_pipeline.fit_transform(housing_num)


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


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
    ('label_binarizer', OnCategoricalEncoder(encoding='onehot')),
])

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
])

housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared, housing_prepared.shape)
