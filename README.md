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

## Step 1: Data Preparation


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

# Load the dataset
data = pd.read_csv('Population of all US Cities 2024.csv')

# Observe data
data.head()

# Observe data size
data.shape
```

    Download completed: archive.zip
    Unzipping completed





    (300, 8)



## Step 2: Feature Engineering


```python
# Drop unnecessary features
data = data.drop(["Rank", "US City", "US State"], axis=1)
data

# Split into X, y
X = data.drop("Annual Change", axis=1)
y = data["Annual Change"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2)
```

## Step 3: Modeling


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




    0.7443018649960422



## Step 4: Visualization


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


    
![png](population-of-all-us-cities-2024_files/population-of-all-us-cities-2024_10_0.png)
    


## Step 5: Save & Load Model


```python
# Save model 
pickle.dump(model, open("model.pkl", "wb"))
```


```python
# Load the saved model
loaded_model = pickle.load(open("model.pkl", "rb"))
```

## Step 6: Inference


```python
# Inference
washington_dc_data = np.array([[8097282, 8740292, 26950, 300.46]])
columns = ['Population 2024', 'Population 2020', 'Density (/mile2)', 'Area (mile2)']
washington_dc_data_df = pd.DataFrame(washington_dc_data, columns=columns)
predicted_metrics = loaded_model.predict(washington_dc_data_df)
print(f"Predicted Annual Change: {predicted_metrics}")
```

    Predicted Annual Change: [-0.00375231]
