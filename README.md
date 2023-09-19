## Introduction  
In this lab, we would like to make an XGBoost model to study the e-commerce behavior from a multi-category store. First, we need to download the data to your local machine, then we will load the data into a Pandas DataFrame.

## Objectives
- Apply XGBoost to an example

## Instruction
* Accept the Kaggle policy and download the data from [Kaggle](https://www.kaggle.com/code/tshephisho/ecommerce-behaviour-using-xgboost/data)
* For the first model you will only use the 2019-Nov csv data (which is still around ~2gb zipped)

Start with some exploratory analysis. First, take a look at the first five rows of the DataFrame. Then get the information about the DataFrame, what is the shape of the DataFrame, and what are the coumn names?


```python
# your code

df.head()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-11-01 00:00:00 UTC</td>
      <td>view</td>
      <td>1003461</td>
      <td>2053013555631882655</td>
      <td>electronics.smartphone</td>
      <td>xiaomi</td>
      <td>489.07</td>
      <td>520088904</td>
      <td>4d3b30da-a5e4-49df-b1a8-ba5943f1dd33</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-11-01 00:00:00 UTC</td>
      <td>view</td>
      <td>5000088</td>
      <td>2053013566100866035</td>
      <td>appliances.sewing_machine</td>
      <td>janome</td>
      <td>293.65</td>
      <td>530496790</td>
      <td>8e5f4f83-366c-4f70-860e-ca7417414283</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-11-01 00:00:01 UTC</td>
      <td>view</td>
      <td>17302664</td>
      <td>2053013553853497655</td>
      <td>NaN</td>
      <td>creed</td>
      <td>28.31</td>
      <td>561587266</td>
      <td>755422e7-9040-477b-9bd2-6a6e8fd97387</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-11-01 00:00:01 UTC</td>
      <td>view</td>
      <td>3601530</td>
      <td>2053013563810775923</td>
      <td>appliances.kitchen.washer</td>
      <td>lg</td>
      <td>712.87</td>
      <td>518085591</td>
      <td>3bfb58cd-7892-48cc-8020-2f17e6de6e7f</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-11-01 00:00:01 UTC</td>
      <td>view</td>
      <td>1004775</td>
      <td>2053013555631882655</td>
      <td>electronics.smartphone</td>
      <td>xiaomi</td>
      <td>183.27</td>
      <td>558856683</td>
      <td>313628f1-68b8-460d-84f6-cec7a8796ef2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# your code

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 67501979 entries, 0 to 67501978
    Data columns (total 9 columns):
     #   Column         Dtype  
    ---  ------         -----  
     0   event_time     object 
     1   event_type     object 
     2   product_id     int64  
     3   category_id    int64  
     4   category_code  object 
     5   brand          object 
     6   price          float64
     7   user_id        int64  
     8   user_session   object 
    dtypes: float64(1), int64(3), object(5)
    memory usage: 4.5+ GB



```python
# your code

df.shape
```




    (67501979, 9)




```python
# your code

df.columns
```




    Index(['event_time', 'event_type', 'product_id', 'category_id',
           'category_code', 'brand', 'price', 'user_id', 'user_session'],
          dtype='object')



# Know your Customers
How many unique customers visit the site? Assign the number of visitors to the visitor variable and print it out


```python
# your code

visitor = df["user_id"].nunique()
print("visitors: {}".format(visitor))
```

    visitors: 3696117


# Visitors Daily Trend
Does traffic fluctuate by date? Try using the `event_time` and `user_id` to see traffic. First you need to select by `event_time` and `user_id`, then you will `drop_duplicates` and `groupby` `event_time` and `user_id`.

### By Category and Product
Which category do customers interact with the most? What brand do they view the most? You can choose just the categories with at least 30 records in order to construct the plots.

## Purchases

When the event_type is "purchase", what item do customers buy?

Try using `'event_type' == 'purchase'` and drop empty rows to assess which categories customers buy.


```python
# your code

purchase = df.loc[df["event_type"] == "purchase"]
purchase = purchase.dropna(axis="rows")
purchase.head()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>168</th>
      <td>2019-11-01 00:01:04 UTC</td>
      <td>purchase</td>
      <td>1005161</td>
      <td>2053013555631882655</td>
      <td>electronics.smartphone</td>
      <td>xiaomi</td>
      <td>211.92</td>
      <td>513351129</td>
      <td>e6b7ce9b-1938-4e20-976c-8b4163aea11d</td>
    </tr>
    <tr>
      <th>707</th>
      <td>2019-11-01 00:04:51 UTC</td>
      <td>purchase</td>
      <td>1004856</td>
      <td>2053013555631882655</td>
      <td>electronics.smartphone</td>
      <td>samsung</td>
      <td>128.42</td>
      <td>562958505</td>
      <td>0f039697-fedc-40fa-8830-39c1a024351d</td>
    </tr>
    <tr>
      <th>939</th>
      <td>2019-11-01 00:06:33 UTC</td>
      <td>purchase</td>
      <td>1801881</td>
      <td>2053013554415534427</td>
      <td>electronics.video.tv</td>
      <td>samsung</td>
      <td>488.80</td>
      <td>557746614</td>
      <td>4d76d6d3-fff5-4880-8327-e9e57b618e0e</td>
    </tr>
    <tr>
      <th>942</th>
      <td>2019-11-01 00:06:34 UTC</td>
      <td>purchase</td>
      <td>5800823</td>
      <td>2053013553945772349</td>
      <td>electronics.audio.subwoofer</td>
      <td>nakamichi</td>
      <td>123.56</td>
      <td>514166940</td>
      <td>8ef5214a-86ad-4d0b-8df3-4280dd411b47</td>
    </tr>
    <tr>
      <th>1107</th>
      <td>2019-11-01 00:07:38 UTC</td>
      <td>purchase</td>
      <td>30000218</td>
      <td>2127425436764865054</td>
      <td>construction.tools.welding</td>
      <td>magnetta</td>
      <td>254.78</td>
      <td>515240495</td>
      <td>0253151d-5c84-4809-ba02-38ac405494e1</td>
    </tr>
  </tbody>
</table>
</div>



## What brands do the customers buy?
Try grouping by brand and sorting the values by the brand name.


```python
# your code

top_sellers = (
    purchase.groupby("brand")["brand"]
    .agg(["count"])
    .sort_values("count", ascending=False)
)
top_sellers.head(20)
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
      <th>count</th>
    </tr>
    <tr>
      <th>brand</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>samsung</th>
      <td>198670</td>
    </tr>
    <tr>
      <th>apple</th>
      <td>165681</td>
    </tr>
    <tr>
      <th>xiaomi</th>
      <td>57909</td>
    </tr>
    <tr>
      <th>huawei</th>
      <td>23466</td>
    </tr>
    <tr>
      <th>oppo</th>
      <td>15080</td>
    </tr>
    <tr>
      <th>lg</th>
      <td>11828</td>
    </tr>
    <tr>
      <th>artel</th>
      <td>7269</td>
    </tr>
    <tr>
      <th>lenovo</th>
      <td>6546</td>
    </tr>
    <tr>
      <th>acer</th>
      <td>6402</td>
    </tr>
    <tr>
      <th>bosch</th>
      <td>5718</td>
    </tr>
    <tr>
      <th>indesit</th>
      <td>5187</td>
    </tr>
    <tr>
      <th>respect</th>
      <td>4557</td>
    </tr>
    <tr>
      <th>hp</th>
      <td>4002</td>
    </tr>
    <tr>
      <th>midea</th>
      <td>3984</td>
    </tr>
    <tr>
      <th>elenberg</th>
      <td>3944</td>
    </tr>
    <tr>
      <th>haier</th>
      <td>3826</td>
    </tr>
    <tr>
      <th>beko</th>
      <td>3813</td>
    </tr>
    <tr>
      <th>casio</th>
      <td>3477</td>
    </tr>
    <tr>
      <th>tefal</th>
      <td>3343</td>
    </tr>
    <tr>
      <th>vitek</th>
      <td>3095</td>
    </tr>
  </tbody>
</table>
</div>



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

df_targets = df.loc[df["event_type"].isin(["cart", "purchase"])].drop_duplicates(
    subset=["event_type", "product_id", "price", "user_id", "user_session"]
)
cart_purchase_users = df.loc[
    df["event_type"].isin(["cart", "purchase"])
].drop_duplicates(subset=["user_id"])
cart_purchase_users.dropna(how="any", inplace=True)
cart_purchase_users_all_activity = df.loc[
    df["user_id"].isin(cart_purchase_users["user_id"])
]
```


```python
# create the two new columns for the category levels 1 and 2
# your code here
df_targets["category_level1"] = (
    df_targets["category_code"].str.split(".", expand=True)[0].astype("category")
)
df_targets["category_level2"] = (
    df_targets["category_code"].str.split(".", expand=True)[1].astype("category")
)
```


```python
# Change the event_time to a timestamp
# your code

df_targets["timestamp"] = pd.to_datetime(df_targets["event_time"])
```


```python
# Use pandas.dt.dayofweek to get the day of the week
# your code
df_targets["weekday"] = df_targets["timestamp"].dt.dayofweek
```


```python
# add the is_purchased feature
# your code

df_targets["is_purchased"] = np.where(df_targets["event_type"] == "purchase", 1, 0)
df_targets["is_purchased"] = df_targets.groupby(["user_session", "product_id"])[
    "is_purchased"
].transform("max")
```

### Save new data structure if desired

## Resampling training set


```python
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

## Split the data
Use a test size of 0.3 and a random state of 86 to split the data into test and train subsets


```python
X_train, X_test, y_train, y_test = train_test_split(
    features, is_purchased, test_size=0.3, random_state=86
)
```

## Train the model
Choose learning rate of 0.1 on XGBClassifier, fit the model, and make predictions on the test set


```python
from xgboost import XGBClassifier

model = XGBClassifier(learning_rate=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

    /Users/pisel/miniconda3/envs/learn-env-m1tf/lib/python3.9/site-packages/xgboost/data.py:298: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if is_sparse(dtype):
    /Users/pisel/miniconda3/envs/learn-env-m1tf/lib/python3.9/site-packages/xgboost/data.py:300: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      elif is_categorical_dtype(dtype) and enable_categorical:
    /Users/pisel/miniconda3/envs/learn-env-m1tf/lib/python3.9/site-packages/xgboost/data.py:298: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if is_sparse(dtype):
    /Users/pisel/miniconda3/envs/learn-env-m1tf/lib/python3.9/site-packages/xgboost/data.py:300: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      elif is_categorical_dtype(dtype) and enable_categorical:


## Feature importance
Plot the feature importance using plot_importance
