#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing packages 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 


# In[2]:


# Pull in our dataset 
data = {

    'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],

    'BloodPressure': [120, 122, 126, 128, 130, 133, 135, 138, 142, 145, 150, 155, 160, 165, 170, 175]

}


# In[3]:


df = pd.DataFrame(data)

df_descriptive = df.describe()
print(df_descriptive)


# In[4]:


# scatter plot of Age vs Blood Pressure 
plt.figure(figsize=(8,6))
plt.scatter(df['Age'], df['BloodPressure'], color='darkgreen')
plt.xlabel('Age')
plt.ylabel('Blood Pressure')
plt.title("Age vs. Blood Pressure")
plt.show()


# In[5]:


# Linear regression model
X = df[['Age']]
y = df['BloodPressure']
regression = LinearRegression().fit(X,y)


# In[6]:


#
plt.plot(X, regression.predict(X), label = "Regression Line", color= "red")
plt.scatter(df['Age'], df['BloodPressure'], color='darkgreen')
plt.show()


# In[ ]:


slope = regression.coef_[0]
intercept = regression.intercept_
print(f"Regression model had slope of {slope:.2f} and intercept of {intercept:.2f}.")


# In[ ]:


# predictions 
new_ages = [30, 40, 50, 60]
df_ages = pd.DataFrame({'Age' : new_ages})
predicted_blood_pressures = regression.predict(df_ages)


# In[ ]:


for age, bp in zip(new_ages , predicted_blood_pressures):
    print(f"Predicted Blood Pressure at Age {age} is {bp:.2f}.")

