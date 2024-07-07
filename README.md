# Population of all US Cities 2024

## Overview
This notebook uses regression modeling to predict the annual population change of US cities based on their population in 2024 and 2020, population density, and area.

## Objectives
- Predict the annual population change.
- Calculate the $R^2$ value and visualize the results with `matplotlib`.

## Tools Used
- numpy
- pandas
- scikit-learn
- matplotlib
- pickle

## Dataset
This dataset provides detailed information about the population of 300 US cities for the years 2024 and 2020. It includes:
- US City
- US State
- Popuation 2024 (x1)
- Population 2020 (x2)
- Annual change (y)
- Density (x3)
- Area (x4)

## Model
We will use the KNeighborsRegressor model for this task. KNeighborsRegressor is suitable for understanding the relationship between the dependent variable (annual population change) and the independent variables (population in 2024 and 2020, population density, and area).

## Credits

**Dataset Author:**
* Ibrar Hussain

**Model Author:**  
* Kevin Thomas

**Date:**  
* 07-06-24  

**Version:**  
* 1.0


```python
import os
import requests
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import pickle
import matplotlib.pyplot as plt
```


```python
# Download and extract the dataset
url = 'https://www.kaggle.com/api/v1/datasets/download/dataanalyst001/population-of-all-us-cities-2024?datasetVersionNumber=2'
local_filename = 'archive.zip'
response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(local_filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f'Download completed: {local_filename}')
else:
    print(f'Failed to download the file. Status code: {response.status_code}')
if response.status_code == 200:
    with zipfile.ZipFile(local_filename, 'r') as zip_ref:
        zip_ref.extractall('.')
    print('Unzipping completed')
else:
    print('Skipping unzipping due to download failure')

```

    Download completed: archive.zip
    Unzipping completed



```python
# Load the dataset
data = pd.read_csv('Population of all US Cities 2024.csv')
```


```python
# Observe data
data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rank</th>
      <th>US City</th>
      <th>US State</th>
      <th>Population 2024</th>
      <th>Population 2020</th>
      <th>Annual Change</th>
      <th>Density (/mile2)</th>
      <th>Area (mile2)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>New York</td>
      <td>New York</td>
      <td>8097282</td>
      <td>8740292</td>
      <td>-0.0195</td>
      <td>26950</td>
      <td>300.46</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Los Angeles</td>
      <td>California</td>
      <td>3795936</td>
      <td>3895848</td>
      <td>-0.0065</td>
      <td>8068</td>
      <td>470.52</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Chicago</td>
      <td>Illinois</td>
      <td>2638159</td>
      <td>2743329</td>
      <td>-0.0099</td>
      <td>11584</td>
      <td>227.75</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Houston</td>
      <td>Texas</td>
      <td>2319119</td>
      <td>2299269</td>
      <td>0.0021</td>
      <td>3620</td>
      <td>640.61</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Phoenix</td>
      <td>Arizona</td>
      <td>1662607</td>
      <td>1612459</td>
      <td>0.0076</td>
      <td>3208</td>
      <td>518.33</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Observe data size
data.shape
```




    (300, 8)




```python
# Drop unnecessary features
data = data.drop(["Rank", "US City", "US State"], axis=1)
data
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Population 2024</th>
      <th>Population 2020</th>
      <th>Annual Change</th>
      <th>Density (/mile2)</th>
      <th>Area (mile2)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8097282</td>
      <td>8740292</td>
      <td>-0.0195</td>
      <td>26950</td>
      <td>300.46</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3795936</td>
      <td>3895848</td>
      <td>-0.0065</td>
      <td>8068</td>
      <td>470.52</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2638159</td>
      <td>2743329</td>
      <td>-0.0099</td>
      <td>11584</td>
      <td>227.75</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2319119</td>
      <td>2299269</td>
      <td>0.0021</td>
      <td>3620</td>
      <td>640.61</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1662607</td>
      <td>1612459</td>
      <td>0.0076</td>
      <td>3208</td>
      <td>518.33</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>295</th>
      <td>110878</td>
      <td>110094</td>
      <td>0.0018</td>
      <td>2982</td>
      <td>37.18</td>
    </tr>
    <tr>
      <th>296</th>
      <td>110803</td>
      <td>111899</td>
      <td>-0.0025</td>
      <td>1961</td>
      <td>56.49</td>
    </tr>
    <tr>
      <th>297</th>
      <td>110801</td>
      <td>108889</td>
      <td>0.0043</td>
      <td>3036</td>
      <td>36.49</td>
    </tr>
    <tr>
      <th>298</th>
      <td>110463</td>
      <td>111477</td>
      <td>-0.0023</td>
      <td>5979</td>
      <td>18.48</td>
    </tr>
    <tr>
      <th>299</th>
      <td>110055</td>
      <td>109782</td>
      <td>0.0006</td>
      <td>4825</td>
      <td>22.81</td>
    </tr>
  </tbody>
</table>
<p>300 rows × 5 columns</p>
</div>




```python
# Split into X, y
X = data.drop("Annual Change", axis=1)
y = data["Annual Change"]
```


```python
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2)
```


```python
# Train the model
model = KNeighborsRegressor(
    algorithm="auto", 
    leaf_size=20, 
    metric="euclidean", 
    n_neighbors=3, 
    weights="distance")
model.fit(X_train, y_train)
model.score(X_test, y_test)
```




    0.7921830174519234




```python
# Plot the results
y_pred = model.predict(X_test)
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Annual Change')
plt.ylabel('Predicted Annual Change')
plt.title('Actual vs Predicted Annual Change')
plt.show()
```


    
![png](population-of-all-us-cities-2024_files/population-of-all-us-cities-2024_11_0.png)
    



```python
# Save model 
pickle.dump(model, open("model.pkl", "wb"))
```


```python
# Load the saved model
loaded_model = pickle.load(open("model.pkl", "rb"))
```


```python
# Inference
washington_dc_data = np.array([[8097282, 8740292, 26950, 300.46]])
columns = ['Population 2024', 'Population 2020', 'Density (/mile2)', 'Area (mile2)']
washington_dc_data_df = pd.DataFrame(washington_dc_data, columns=columns)
predicted_metrics = loaded_model.predict(washington_dc_data_df)
print(f"Predicted Annual Change: {predicted_metrics}")
```

    Predicted Annual Change: [-0.00503874]

