# Timeseries prediction with XGBoost 

![XGBoost](/images/XGBoost_logo.png)

XGBoost is undoubtedly one of the most popular algorithm in ML and Kaggle. It is no surprise if one of Kaggle's Grandmasters posted a while back this:

![Bojan_Tunguz](images/Bojan_XGBoost.png)

## What is XGBoost

XGBoost stands for Extreme Gradient Boosting. More specifically, XGBoost is a decision-tree based ensemble ML algorithm that uses gradient boosting. 

A nice article about XGBoost is [this](https://www.geeksforgeeks.org/xgboost/)

XGBoost can be used for many cases, from regression to classification problems. 

In this post I will use XGBoost for predicting the Close value of the Apple stock by using historical data. I will then predict a number for each day of the test data. This will be then a regression problem. 

## XGBOOST IMPLEMENTATION

We will import some libraries such as pandas for data manipulation, pandas_datareader to grab the live data for the Apple stock, visualization libs like matplotlib and seaborn and finally from the XGBoost lib, we will import the XGBRegressor. 

```python
import pandas as pd
import pandas_datareader as pddr
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor as xgbreg


import warnings
warnings.filterwarnings("ignore")
```

Let's grab the data for the Apple stock from 1980. 

```python
start_date = dt.datetime(1980, 1, 1)
end_date = dt.datetime.now()
stock_sticker = 'AAPL'

df = pddr.DataReader(stock_sticker, 'yahoo', start_date, end_date)
```

Let's have a peak at the data

```python

df.head()

```

![df_head](images/df_head_xgboost.png)

```python

df.tail()

```

![df_tail](images/df_tail_xgboost.png)

The data dates back from December 1980 till yesterday (time of this post).

Let's visualize some of the data:

```python

plt.rcParams.update({'figure.figsize': (17, 3), 'figure.dpi':300})
fig, ax = plt.subplots()
sns.lineplot(data=df.tail(1825), x=df.tail(1825).index, y='Close')
plt.grid(linestyle='-', linewidth=0.3)
ax.tick_params(axis='x', rotation=90)

```

![plot](images/apple_plot.png)

When working with timeseries, you want to split the data between training and test. Normally from the data obtained, we decide a cut-off date, from where the test data will be taken. Data prior to that date will be the train data.
Let's do that:

```python

data_split = '02-01-2022'
train_data = df.loc[df.index < data_split]
test_data = df.loc[df.index >= data_split]

fig, ax = plt.subplots(figsize=(15,5))
train_data['Close'].plot(ax=ax, label='Train data', title="Data split")
test_data['Close'].plot(ax=ax, label='Test Data')
ax.axvline(data_split, color='black', ls='dotted')
plt.show()

```

![Data_split](images/xgboost_data_split.png)

The blue line is the train data and the orange one is the test data. 

