# Accelerate Pandas for quicker data manipulation

![Pandas](/images/Pandas_logo.jpg)

In this post I will go over different ways on how to manipulate data in pandas and compare them to verify which one is the quickest one.
In ML and AI, data can be huge in size and oftentimes it is good practice to apply correct methods when manipulating it and make your code as efficient as possible.

## The data

For this particular topic I will just create a ficticious dataset of people with columns:

- Age => values between 18 and 80
- Weight in kg => values between 40 and 150
- Height in cm => values between 140 and 210

First some imports

```python
import pandas as pd
import numpy as np
from tqdm import tqdm
```

Let's create the function to get the dataframe

```python
def create_dataset(size):
  df = pd.DataFrame()
  df["age"] = np.random.randint(18, 80, size)
  df["weight_kg"] = np.random.randint(40, 150, size)
  df["height_cm"] = np.random.randint(140, 210, size)
  return df
```

### Operations on the df

From the dataset we want to calcualte the BMI (Body Mass Index). The calculation is pretty straightforward with kg and m. The formula is just weight/height^2.
For example a person that weights 67kg and is 170 cm, the formula should be 67/1.70x1.70=23.18

We will look at three different ways of running this calcualtion on the dataset. 
Let's create the function to calculated the BMI

```python
def calculate_BMI(row):
  bmi_calculation = row["weight_kg"]/(row["height_cm"]*row["height_cm"])
  return bmi_calculation*10_000
```

And below the three different functions to invoke the function ***calculate_BMI***

### Iterrows
With iterrows we will simply loop through the dataframe and do the calculation at each row. 
```python
def iterrows_method(df):
  for index, row in tqdm(df.iterrows()):
    df.loc[index, "BMI"] = calculate_BMI(row)
  return df
```
### Apply method
With apply we are able to apply a function on the axis you want, by default 0 so the rows. 
```python
def apply_method(df):
  df["BMI"] = df.apply(calculate_BMI, axis=1)
  return df
```
### Np.where
With numpy and where method we are able to do some action based on some conditions.
```python
def np_where_method(df):
  df["BMI"] = np.where(df["weight_kg"]>0, (df["weight_kg"]/(df["height_cm"]*df["height_cm"]))*10_000, 0)
  return df
```

We will first try with a small dataframe of 100k rows and compare the three methods' speeds. 
In Jupyter Notebook you can time a cell execution time with %%timeit. 
We will first create a dataframe by calling the create_dataset method passing the size we want it to be and then run the selected method for the BMI calculation.

### Iterrows at 100k

```python
df = create_dataset(100_000)

%%timeit
iterrows_method(df)
```
100000it [01:20, 1244.84it/s]

100000it [01:20, 1236.19it/s]

100000it [01:19, 1256.01it/s]

100000it [01:19, 1256.17it/s]

100000it [01:17, 1282.12it/s]

100000it [01:17, 1288.55it/s]
1 loop, best of 5: 1min 17s per loop

A bit over 1 min to process 100k rows. Kinda slow. Can we do better?

### Apply at 100k

```python
df = create_dataset(100_000)

%%timeit
apply_method(df)
```
1 loop, best of 5: 1.73 s per loop

Now this is faster. But can we do better?

### NP.Where at 100k

```python
df = create_dataset(100_000)

%%timeit
np_where_method(df)
```
100 loops, best of 5: 2.53 ms per loop

That's a big improvement! 

![DataFrame](/images/BMI_Example.PNG)

### Bigger DataFrames

Let's do the same but with 500k. 

### Iterrows

```python
df = create_dataset(500_000)

%%timeit
iterrows_method(df)
```
500000it [27:55, 298.48it/s]

500000it [27:39, 301.23it/s]

83551it [04:21, 319.63it/s]

I actually had to interrupt the execution because it was taking way too long. 27 min seems excessive. 

### Apply 
```python
df = create_dataset(500_000)

%%timeit
apply_method(df)

```
1 loop, best of 5: 8.82 s per loop

That's a huge difference now.

### np.where

### Apply 
```python
df = create_dataset(500_000)

%%timeit
np_where_method(df)
```
100 loops, best of 5: 8.25 ms per loop

Looks like np.where is the fastest method to do such data manipulation. For sure iterrows is to avoid :D 



