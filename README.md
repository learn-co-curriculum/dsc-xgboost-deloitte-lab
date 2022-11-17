## Introduction  
In this lab, we would like to make an XGBoost model to study over the ecommerce behavior from a multi-category store. First, we need to download the data to your local machine, then we will load the data from the local machine onto a Pandas Dataframe.

## Objectives
- Apply XGBoost to an example

## Instruction

- Accept the kaggle policy and download the data from here https://www.kaggle.com/code/tshephisho/ecommerce-behaviour-using-xgboost/data
- For the first model building, we'll only use the 2019-Nov csv data (which is still around ~2gb)


```python
# import necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
```


```python
import matplotlib.pyplot as plt
import squarify
import matplotlib.dates as dates
from datetime import datetime

%matplotlib inline
```


```python
# fill this in with your path (absolute path works as well)
path_to_file = ""
df = pd.read_csv(path_to_file)
```

Let's do some exploratory analysis on the data!


```python
# take a quick look of the data
df.head()
```


```python
df.info()
```


```python
df.shape
```


```python
df.columns
```

# Know your Customers
How many unique customers visit the site?


```python
visitor = df['user_id'].nunique()
print ("visitors: {}".format(visitor))
```

# Visitors Daily Trend
Does traffic flunctuate by date? Try using the `event_time` and `user_id` to see traffic, and draw out the plots for visualization. 


```python
    d = df.loc[:,['event_time','user_id']]
    d['event_time'] = d['event_time'].apply(lambda s: str(s)[0:10])
    visitor_by_date = d.drop_duplicates().groupby(['event_time'])['user_id'].agg(['count']).sort_values(by=['event_time'], ascending=True)
    x = pd.Series(visitor_by_date.index.values).apply(lambda s: datetime.strptime(s, '%Y-%m-%d').date())
    y = visitor_by_date['count']
    plt.rcParams['figure.figsize'] = (20,8)

    plt.plot(x,y)
    plt.show()
```

### By Category and Product
Which category customers interact the most? What brand the view to most? You can limit the number of category number to 30 to draw out the plots.


```python
max_category_num = 30
top_category = df.loc[:,'category_code'].value_counts()[:max_category_num].sort_values(ascending=False)
squarify.plot(sizes=top_category, label=top_category.index.array, color=["red","cyan","green","orange","blue","grey"], alpha=.7  )
plt.axis('off')
plt.show()
```

## event_type is "purchase", what item do customers buy?

Try using `'event_type' == 'purchase'` and drop empty rows to assess which categories customers buy.


```python
# your code
```

## What brands the customers buy?
Try grouping by brand and sort values by the brand name.


```python
# your code
```

# Modeling: predict at time of addition to shopping cart if user will purchase a given product or not
### Feature engineering

The goal of the modeling is to predict if the user will purchase a product or not when they add the product on the cart. This is called `cart abandonment` if the user does not purchase.

First, reconstruct and restructure the data to feed into the machine learning model. For this use case, target only the data which customers have "put" the product in the cart.

Create these new features into the training data set:
- `category_level1`: category
- `category_level2`: sub-category
- `weekday`: weekday of the event
- `activity_count`: number of activity in that session
- `is_purchased`: whether the put in cart item is purchased, this will be our categorical output.

Make sure to de-dup any record.

**Prepare a dataframe for counting activity in the session**


```python
activity_in_session = cart_purchase_users_all_activity.groupby(['user_session'])['event_type'].count().reset_index()
activity_in_session = activity_in_session.rename(columns={"event_type": "activity_count"})
```


```python
del d # free memory
```


```python
df_targets = df_targets.merge(activity_in_session, on='user_session', how='left')
df_targets['activity_count'] = df_targets['activity_count'].fillna(0)
df_targets.head()
```

## Save new data structure for modeling


```python
df_targets.to_csv('training_data.csv')
```


```python
df_targets.info()
```


```python
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
from sklearn.utils import resample
from sklearn import metrics
```

## Resampling training set


```python
is_purcahase_set = df_targets[df_targets['is_purchased']== 1]
is_purcahase_set.shape[0]
```


```python
not_purcahase_set = df_targets[df_targets['is_purchased']== 0]
not_purcahase_set.shape[0]
```


```python
n_samples = 500000
is_purchase_downsampled = resample(is_purcahase_set,
                                replace = False, 
                                n_samples = n_samples,
                                random_state = 27)
not_purcahase_set_downsampled = resample(not_purcahase_set,
                                replace = False,
                                n_samples = n_samples,
                                random_state = 27)
```


```python
downsampled = pd.concat([is_purchase_downsampled, not_purcahase_set_downsampled])
downsampled['is_purchased'].value_counts()
```


```python
features = downsampled.loc[:,['brand', 'price', 'event_weekday', 'category_code_level1', 'category_code_level2', 'activity_count']]
```

## Encode categorical variables


```python
features.loc[:,'brand'] = LabelEncoder().fit_transform(downsampled.loc[:,'brand'].copy())
features.loc[:,'event_weekday'] = LabelEncoder().fit_transform(downsampled.loc[:,'event_weekday'].copy())
features.loc[:,'category_level1'] = LabelEncoder().fit_transform(downsampled.loc[:,'category_level1'].copy())
features.loc[:,'category_level2'] = LabelEncoder().fit_transform(downsampled.loc[:,'category_level2'].copy())

is_purchased = LabelEncoder().fit_transform(downsampled['is_purchased'])
features.head()
```


```python
print(list(features.columns))
```

## Split the data


```python
X_train, X_test, y_train, y_test = train_test_split(features, 
                                                    is_purchased, 
                                                    test_size = 0.3, 
                                                    random_state = 42)
```

## Train the model
Choose learning rate of 0.1 on XGBClassifier.


```python
from xgboost import XGBClassifier
model = XGBClassifier(learning_rate=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```


```python
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("fbeta:",metrics.fbeta_score(y_test, y_pred, average='weighted', beta=0.5))
```

## Feature importance
Show feature importance using plot_importance


```python
plot_importance(model, max_num_features=10, importance_type ='gain')
plt.rcParams['figure.figsize'] = (40,10)
plt.show()
```


```python

```
