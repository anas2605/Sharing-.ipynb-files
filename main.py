#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

#file_path = 'C:/Users/MOHAMED ANAS/Downloads/auto+mpg/auto-mpg.data-original'
file_path = 'C:/Users/MOHAMED ANAS/Downloads/auto+mpg/auto-mpg.data'

columnnames = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 
    'model_year', 'origin', 'car_name']

df = pd.read_csv(file_path, delim_whitespace=True, names=columnnames)

print(df.head())


# In[2]:


dataRows=len(df.index)
print(dataRows)


# In[3]:


dataCols=len(df.columns)
print(dataCols)


# In[4]:


rawData=df


# In[5]:


rawData.tail(10)


# In[6]:


# Checking for missing values 
missing_val = df.isnull().sum()
print("Missing values in each column:")
print(missing_val)


# In[7]:


preprocessedData=df


# In[8]:


# Filling na values
preprocessedData['horsepower'].fillna(preprocessedData['horsepower'].mean(), inplace=True)
preprocessedData['mpg'].fillna(preprocessedData['mpg'].mean(), inplace=True)
print(preprocessedData.head())


# In[9]:


# 1. Average MPG
averageMPG = preprocessedData['mpg'].mean()
print("The average miles per gallon (MPG) is ",averageMPG)


# In[10]:


# 2. Most Common Vehicle Type
commonVehicleType = preprocessedData['car_name'].mode()[0]
print("The most common vehicle type is ",commonVehicleType)


# In[11]:


# 3. Most Frequently Occurring Cylinder Count
commonCylinderCount = preprocessedData['cylinders'].mode()[0]
print("The most frequently occurring cylinder count is ",commonCylinderCount)


# In[12]:


import math
# 4. Std. deviation function 
def standardDeviation(data):
    mean = sum(data) / len(data)    
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    # Return the standard deviation (square root of variance)
    return math.sqrt(variance)


# In[13]:


# 5. Correlation coefficient function
def correlationCoefficient(x, y):
    if len(x) != len(y):
        print("The two input vectors must have the same length.")
    # Calculate means of x and y
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    # Calculate the covariance numerator
    covariance = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    # Return the correlation coefficient
    return covariance / (len(x) * standardDeviation(x) * standardDeviation(y))


# In[14]:


mpg = preprocessedData['mpg']
horsepower = preprocessedData['horsepower']
weight = preprocessedData['weight']
# Correlation matrix
attributeCorrelations = [
    [correlationCoefficient(mpg, mpg), correlationCoefficient(mpg, horsepower), correlationCoefficient(mpg, weight)],
    [correlationCoefficient(horsepower, mpg), correlationCoefficient(horsepower, horsepower), correlationCoefficient(horsepower, weight)],
    [correlationCoefficient(weight, mpg), correlationCoefficient(weight, horsepower), correlationCoefficient(weight, weight)]
]
print(attributeCorrelations)


# In[15]:


#In this matrix:
#The diagonal values are all 1 because each attribute is perfectly correlated with itself.
#The other values represent the Pearson correlation between the pairs of attributes.
#Analysis:
#MPG vs. Horsepower: The correlation value of -0.778426 suggests a strong negative correlation between MPG and horsepower. As horsepower increases, MPG decreases.
#MPG vs. Weight: The correlation value of -0.832244 suggests a strong negative correlation between MPG and weight. Heavier cars generally have lower fuel efficiency (MPG).
#Horsepower vs. Weight: The correlation value of 0.864538 indicates a strong positive correlation. More powerful cars tend to be heavier.
#Conclusion:
#MPG is inversely related to both horsepower and weight.
#Horsepower and weight have a positive relationship, meaning more powerful cars are typically heavier.


# In[16]:


correlationDispMPG = correlationCoefficient(preprocessedData['displacement'], mpg)
print("The correlation between Displacement and MPG is ",correlationDispMPG)


# In[17]:


a = ['horsepower', 'weight', 'displacement', 'acceleration', 'cylinders']
d={}
for i in a:
    correlation_values = correlationCoefficient(mpg, preprocessedData[i])
    #print(i,correlation_values)
    d.update({i:correlation_values})
print(d)


# In[18]:


#Feature engineering
#1. vehicle age
from datetime import datetime 
current_year = datetime.today().year
preprocessedData['vehicle_age'] = current_year - (preprocessedData['model_year'])
print(df.loc[:, ['vehicle_age', 'model_year']])


# In[19]:


#2. Normalize or standardise the numerical features in the dataset.
numerical_features = ['mpg', 'horsepower', 'weight', 'displacement', 'acceleration', 'vehicle_age']
preprocessedData[numerical_features] = (preprocessedData[numerical_features] - preprocessedData[numerical_features].mean()) / preprocessedData[numerical_features].std()
print(preprocessedData[numerical_features].head())


# In[21]:


preprocessedData["make"]=preprocessedData['car_name'].str.split().str[0]
print(preprocessedData["make"].head())


# In[22]:


preprocessedData['make_label'] = pd.factorize(preprocessedData['make'])[0]
print(preprocessedData[['make', 'make_label']].head())


# In[23]:


import seaborn as sns
import matplotlib.pyplot as plt
# Plot the histogram for MPG
sns.histplot(preprocessedData['mpg'],color='orange')
plt.title('Distribution of MPG')
plt.xlabel('MPG')
plt.ylabel('Frequency')
plt.show()


# In[24]:


make_counts = preprocessedData['make'].value_counts()
top_10 = make_counts.head(10)
# Create the bar plot
sns.barplot(x=top_10.index, y=top_10.values)
plt.title('Top 10 Most Frequent Makes')
plt.xlabel('Make')
plt.ylabel('Frequency')
plt.xticks(rotation=90) 
plt.show()


# In[25]:


top_10_makes = make_counts.head(10).index
# Filter the dataset for these top 10 makes
top_10_makes_data = preprocessedData[preprocessedData['make'].isin(top_10_makes)]
# Create the box plot
sns.boxplot(x='make', y='mpg', data=top_10_makes_data)
plt.title('Box Plot of the MPG for Different Makes')
plt.xlabel('Make')
plt.ylabel('MPG')
plt.xticks(rotation=90)
plt.show()


# In[26]:


# Scatter plot for horsepower vs MPG
plt.subplot(2, 2, 1)  # (rows, columns, position)
sns.scatterplot(x='horsepower', y='mpg', data=preprocessedData)
plt.title('Horsepower vs MPG')
plt.xlabel('Horsepower')
plt.ylabel('MPG')

# Scatter plot for weight vs MPG
plt.subplot(2, 2, 2)
sns.scatterplot(x='weight', y='mpg', data=preprocessedData)
plt.title('Weight vs MPG')
plt.xlabel('Weight')
plt.ylabel('MPG')

# Scatter plot for displacement vs MPG
plt.subplot(2, 2, 3)
sns.scatterplot(x='displacement', y='mpg', data=preprocessedData)
plt.title('Displacement vs MPG')
plt.xlabel('Displacement')
plt.ylabel('MPG')

# Scatter plot for acceleration vs MPG
plt.subplot(2, 2, 4)
sns.scatterplot(x='acceleration', y='mpg', data=preprocessedData)
plt.title('Acceleration vs MPG')
plt.xlabel('Acceleration')
plt.ylabel('MPG')


# Show the plots
plt.tight_layout()
plt.show()


# In[27]:


from sklearn.model_selection import train_test_split

# Split the data into training and test sets (70% training, 15% validation, 15% testing)
trainingdata, temp = train_test_split(preprocessedData, test_size=0.30, random_state=42)
validationdata, testingdata = train_test_split(temp, test_size=0.50, random_state=42)
print("Training set size:",len(trainingdata))
print("Validation set size:",len(validationdata))
print("Test set size:",len(testingdata))


# In[29]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
X_train= trainingdata.drop(['mpg', 'car_name','make'], axis=1)  
Y_train = trainingdata['mpg'] 
model = LinearRegression()
model.fit(X_train, Y_train)


# In[30]:


X_valid= validationdata.drop(['mpg', 'car_name','make'], axis=1) 
Y_valid= validationdata['mpg']
y_predict_valid = model.predict(X_valid)
mean_sq_err_valid = mean_squared_error(Y_valid, y_predict_valid)
r_square_valid = r2_score(Y_valid, y_predict_valid)
print("Mean Squared Error of validation data is : ",mean_sq_err_valid)
print("R-squared value of validation data is: ",r_square_valid)


# In[31]:


X_test= testingdata.drop(['mpg', 'car_name','make'], axis=1) 
Y_test= testingdata['mpg']
y_predict_test = model.predict(X_test)
mean_sq_err_test = mean_squared_error(Y_test, y_predict_test)
r_square_test = r2_score(Y_test, y_predict_test)
print("Mean Squared Error of testing data is : ",mean_sq_err_test)
print("R-squared value of testing data is: ",r_square_test)


# In[32]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scale=scaler.fit_transform(X_train)
X_valid_scale=scaler.transform(X_valid)
X_test_scale=scaler.transform(X_test)


# In[34]:


ridge= Ridge()
#hyperparameter
param_grid = {'alpha': [0.1, 1, 10, 100, 1000]}
#GridSearchCV to find the best alpha
gridsearch = GridSearchCV(ridge, param_grid, scoring='neg_mean_squared_error', cv=5)
gridsearch.fit(X_train_scale, Y_train)
best_alpha = gridsearch.best_params_['alpha']
print("Best alpha from GridSearchCV is : ",best_alpha)


# In[35]:


#Training model with best alpha
best_alpha_model = gridsearch.best_estimator_
print(best_alpha_model)


# In[36]:


#Validation data prediction
y_predict_valid_2 = best_alpha_model.predict(X_valid_scale)
mean_sq_err_valid_2 = mean_squared_error(Y_valid, y_predict_valid_2)
r_square_valid_2 = r2_score(Y_valid, y_predict_valid_2)
print("Mean Squared Error of validation data is : ",mean_sq_err_valid_2)
print("R-squared value of validation data is: ",r_square_valid_2)


# In[39]:


# Testing data prediction
y_predict_test_2 = best_alpha_model.predict(X_test_scale)
mean_sq_err_test_2 = mean_squared_error(Y_test, y_predict_test_2)
r_square_test_2 = r2_score(Y_test, y_predict_test_2)
print("Mean Squared Error of testing data is : ",mean_sq_err_test_2)
print("R-squared value of testing data is: ",r_square_test_2)


# In[40]:


coefs = best_alpha_model.coef_
intercept = best_alpha_model.intercept_
equation = f"MPG = {intercept:.4f} "  # Start with intercept
for i, coef in enumerate(coefs):
    equation += f" + ({coef:.4f} * X{i + 1})"  # Add each feature term
    
print("Model Equation:")
print(equation)


# In[41]:


#Predict MPG function
def predictMPG(model, scaler, input_data):
    input_data_scaled = scaler.transform([input_data])
    predicted_mpg = model.predict(input_data_scaled)
    return predicted_mpg[0]

