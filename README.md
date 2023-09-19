## Introduction  
In this lab, we would like to make an XGBoost model to study the e-commerce behavior from a multi-category store. First, we need to download the data to your local machine, then we will load the data into a Pandas DataFrame.

## Objectives
- Apply XGBoost to an example

## Instruction
* Accept the Kaggle policy and download the data from [Kaggle](https://www.kaggle.com/code/tshephisho/ecommerce-behaviour-using-xgboost/data)
* For the first model you will only use the 2019-Nov csv data (which is still around ~2gb zipped)


```python
# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from datetime import datetime
```


```python
# fill this in with your path (absolute path works as well)
path_to_file = " "
df = pd.read_csv(path_to_file)
```

Start with some exploratory analysis. First, take a look at the first five rows of the DataFrame. Then get the information about the DataFrame, what is the shape of the DataFrame, and what are the coumn names?


```python
# your code
```


```python
# your code
```


```python
# your code
```


```python
# your code
```

# Know your Customers
How many unique customers visit the site? Assign the number of visitors to the visitor variable and print it out


```python
# your code
```

# Visitors Daily Trend
Does traffic fluctuate by date? Try using the `event_time` and `user_id` to see traffic. First you need to select by `event_time` and `user_id`, then you will `drop_duplicates` and `groupby` `event_time` and `user_id`.


```python
d = df.loc[:, ["event_time", "user_id"]]
d["event_time"] = d["event_time"].apply(lambda s: str(s)[0:10])
visitor_by_date = (
    d.drop_duplicates()
    .groupby(["event_time"])["user_id"]
    .agg(["count"])
    .sort_values(by=["event_time"], ascending=True)
)
x = pd.Series(visitor_by_date.index.values).apply(
    lambda s: datetime.strptime(s, "%Y-%m-%d").date()
)
y = visitor_by_date["count"]
```


```python
plt.rcParams["figure.figsize"] = (17, 5)
plt.plot(x, y)
plt.show()
```

### By Category and Product
Which category do customers interact with the most? What brand do they view the most? You can choose just the categories with at least 30 records in order to construct the plots.


```python
max_category_num = 30
top_category = (
    df.loc[:, "category_code"]
    .value_counts()[:max_category_num]
    .sort_values(ascending=False)
)
```


```python
plt.bar(
    height=top_category,
    x=top_category.index.array,
    color=["red", "cyan", "green", "orange", "blue", "grey"],
    alpha=0.7,
)
plt.axis("off")
plt.show()
```

## Purchases

When the event_type is "purchase", what item do customers buy?

Try using `'event_type' == 'purchase'` and drop empty rows to assess which categories customers buy.


```python
# your code
```

## What brands do the customers buy?
Try grouping by brand and sorting the values by the brand name.


```python
# your code
```


```python
del d  # free memory
```

# Modeling: predict at the time of addition to a shopping cart if the user will purchase a given product or not

### Feature engineering

The goal of this modeling is to predict if the user will purchase a product or not when they add the product to the cart. This is called `cart abandonment` if the user does not purchase.

First, reconstruct and restructure the data to feed into the machine learning model. For this use case, target only the data for which customers have "put" the product into the cart. The relevant `event_type`s are thus "cart" and "purchase".

Create these new features in the training data set:
- `activity_count`: number of activity in that session
- `category_level1`: category
- `category_level2`: sub-category --> split on the "." in the category name
- `weekday`: weekday of the event --> convert `event_time` to a datetime object, then use `pandas.Timestamp.weekday`
- `is_purchased`: whether the is purchased after being put in the cart, this will be the categorical output.

Make sure to de-dup any record.

**Prepare a dataframe for counting activity in the session**


```python
# your code
# first just eliminate the records where event_type = "view" and drop NA values and duplicates
```


```python
# now you get the number of activities by user session

activity_in_session = (
    cart_purchase_users_all_activity.groupby(["user_session"])["event_type"]
    .count()
    .reset_index()
)
activity_in_session = activity_in_session.rename(
    columns={"event_type": "activity_count"}
)
df_targets = cart_purchase_users_all_activity.copy()
```


```python
# create the two new columns for the category levels 1 and 2
# your code here
```


```python
# Change the event_time to a timestamp
# your code
```


```python
# Use pandas.dt.dayofweek to get the day of the week
# your code
```


```python
# add the is_purchased feature
# your code
```


```python
df_targets = df_targets.merge(activity_in_session, on="user_session", how="left")
df_targets["activity_count"] = df_targets["activity_count"].fillna(0)
df_targets["brand"] = df_targets["brand"].astype("category")
df_targets.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event_time</th>
      <th>event_type</th>
      <th>product_id</th>
      <th>category_id</th>
      <th>category_code</th>
      <th>brand</th>
      <th>price</th>
      <th>user_id</th>
      <th>user_session</th>
      <th>category_level1</th>
      <th>category_level2</th>
      <th>timestamp</th>
      <th>weekday</th>
      <th>is_purchased</th>
      <th>activity_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-11-01 00:00:01 UTC</td>
      <td>view</td>
      <td>17302664</td>
      <td>2053013553853497655</td>
      <td>NaN</td>
      <td>creed</td>
      <td>28.31</td>
      <td>561587266</td>
      <td>755422e7-9040-477b-9bd2-6a6e8fd97387</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2019-11-01 00:00:01+00:00</td>
      <td>4</td>
      <td>0.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-11-01 00:00:01 UTC</td>
      <td>view</td>
      <td>3601530</td>
      <td>2053013563810775923</td>
      <td>appliances.kitchen.washer</td>
      <td>lg</td>
      <td>712.87</td>
      <td>518085591</td>
      <td>3bfb58cd-7892-48cc-8020-2f17e6de6e7f</td>
      <td>appliances</td>
      <td>kitchen</td>
      <td>2019-11-01 00:00:01+00:00</td>
      <td>4</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-11-01 00:00:02 UTC</td>
      <td>view</td>
      <td>1004258</td>
      <td>2053013555631882655</td>
      <td>electronics.smartphone</td>
      <td>apple</td>
      <td>732.07</td>
      <td>532647354</td>
      <td>d2d3d2c6-631d-489e-9fb5-06f340b85be0</td>
      <td>electronics</td>
      <td>smartphone</td>
      <td>2019-11-01 00:00:02+00:00</td>
      <td>4</td>
      <td>0.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-11-01 00:00:05 UTC</td>
      <td>view</td>
      <td>4600658</td>
      <td>2053013563944993659</td>
      <td>appliances.kitchen.dishwasher</td>
      <td>samsung</td>
      <td>411.83</td>
      <td>526595547</td>
      <td>aab33a9a-29c3-4d50-84c1-8a2bc9256104</td>
      <td>appliances</td>
      <td>kitchen</td>
      <td>2019-11-01 00:00:05+00:00</td>
      <td>4</td>
      <td>0.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-11-01 00:00:09 UTC</td>
      <td>view</td>
      <td>17501048</td>
      <td>2053013558752445019</td>
      <td>NaN</td>
      <td>eveline</td>
      <td>7.59</td>
      <td>515849878</td>
      <td>31e80b9c-e5b3-437b-9112-c2a110e5c38a</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2019-11-01 00:00:09+00:00</td>
      <td>4</td>
      <td>0.0</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>



### Save new data structure if desired


```python
# df_targets.to_csv('training_data.csv')
```


```python
df_targets.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 26749090 entries, 0 to 26749089
    Data columns (total 15 columns):
     #   Column           Dtype              
    ---  ------           -----              
     0   event_time       object             
     1   event_type       object             
     2   product_id       int64              
     3   category_id      int64              
     4   category_code    object             
     5   brand            category           
     6   price            float64            
     7   user_id          int64              
     8   user_session     object             
     9   category_level1  category           
     10  category_level2  category           
     11  timestamp        datetime64[ns, UTC]
     12  weekday          int32              
     13  is_purchased     float64            
     14  activity_count   float64            
    dtypes: category(3), datetime64[ns, UTC](1), float64(3), int32(1), int64(3), object(4)
    memory usage: 2.4+ GB



```python
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
from sklearn.utils import resample
from sklearn import metrics
```

## Resampling training set


```python
is_purcahase_set = df_targets[df_targets["is_purchased"] == 1]
is_purcahase_set.shape[0]
```




    2991287




```python
not_purcahase_set = df_targets[df_targets["is_purchased"] == 0]
not_purcahase_set.shape[0]
```




    23757802




```python
n_samples = 500000
is_purchase_downsampled = resample(
    is_purcahase_set, replace=False, n_samples=n_samples, random_state=27
)
not_purcahase_set_downsampled = resample(
    not_purcahase_set, replace=False, n_samples=n_samples, random_state=27
)
```


```python
downsampled = pd.concat([is_purchase_downsampled, not_purcahase_set_downsampled])
downsampled["is_purchased"].value_counts()
```




    is_purchased
    1.0    500000
    0.0    500000
    Name: count, dtype: int64




```python
# select the brand, price, weekday, category_level1, cateogry_level2, and activity_count features

features = None
```


```python
# __SOLUTION
# select the brand, price, weekday, category_level1, cateogry_level2, and activity_count features

features = downsampled.loc[
    :,
    [
        "brand",
        "price",
        "weekday",
        "category_level1",
        "category_level2",
        "activity_count",
    ],
]
```

## Encode categorical variables


```python
features.loc[:, "brand"] = LabelEncoder().fit_transform(
    downsampled.loc[:, "brand"].copy()
)
features.loc[:, "weekday"] = LabelEncoder().fit_transform(
    downsampled.loc[:, "weekday"].copy()
)
features.loc[:, "category_level1"] = LabelEncoder().fit_transform(
    downsampled.loc[:, "category_level1"].copy()
)
features.loc[:, "category_level2"] = LabelEncoder().fit_transform(
    downsampled.loc[:, "category_level2"].copy()
)

is_purchased = LabelEncoder().fit_transform(downsampled["is_purchased"])
features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>brand</th>
      <th>price</th>
      <th>weekday</th>
      <th>category_level1</th>
      <th>category_level2</th>
      <th>activity_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>24522991</th>
      <td>2858</td>
      <td>458.96</td>
      <td>2</td>
      <td>2</td>
      <td>18</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>20968274</th>
      <td>2180</td>
      <td>84.43</td>
      <td>2</td>
      <td>1</td>
      <td>35</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>21415500</th>
      <td>2283</td>
      <td>195.31</td>
      <td>3</td>
      <td>7</td>
      <td>40</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>22427083</th>
      <td>142</td>
      <td>1003.57</td>
      <td>5</td>
      <td>7</td>
      <td>40</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>24304376</th>
      <td>1226</td>
      <td>299.93</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(list(features.columns))
```

    ['brand', 'price', 'weekday', 'category_level1', 'category_level2', 'activity_count']


## Split the data
Use a test size of 0.3 and a random state of 86 to split the data into test and train subsets


```python
X_train, X_test, y_train, y_test = None, None, None, None
```

## Train the model
Choose learning rate of 0.1 on XGBClassifier, fit the model, and make predictions on the test set


```python
from xgboost import XGBClassifier

model = None
# fit the model on the train sets
y_pred = None
```


```python
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("fbeta:", metrics.fbeta_score(y_test, y_pred, average="weighted", beta=0.5))
```

    Accuracy: 0.7050833333333333
    Precision: 0.6922538949077398
    Recall: 0.7388172100338549
    fbeta: 0.7053105460567092


## Feature importance
Plot the feature importance using plot_importance


```python
plot_importance(model, max_num_features=10, importance_type="gain")
# plt.rcParams['figure.figsize'] = (40,10)
plt.show()
```
