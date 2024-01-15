# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:35:35 2023

@author: simra
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Importing dataset
CAssamOD= pd.read_csv("C:\\Users\\simra\\OneDrive\\Desktop\\NM\\SEM III\\RIDS Project\\Copy of OD_ASCR(Cleaned).csv")
pd.set_option("display.max_columns" and "display.max_rows",None) 

#Data Cleaning: Dropping Null Values and Redundant Columns
CAssamOD['Travel Time ()'] = pd.to_timedelta(CAssamOD['Travel Time ()'], errors='coerce')
CAssamOD['Time of Survey'].fillna(CAssamOD["Time of Survey"].median(), inplace=True)
CAssamOD['Travel Distance(km)'].fillna(CAssamOD["Travel Distance(km)"].median(), inplace=True)
CAssamOD['Travel Time ()'].fillna(CAssamOD["Travel Time ()"].median(), inplace=True)
CAssamOD["Purpose"].fillna(CAssamOD["Purpose"].median(),inplace = True)
CAssamOD["Occupancy"].fillna(CAssamOD["Occupancy"].median(),inplace = True)
CAssamOD["Frequency"].fillna(CAssamOD["Frequency"].median(),inplace= True)
CAssamOD["Road Name"].fillna(CAssamOD["Road Name"].median(),inplace=True)
D=CAssamOD.drop(["Unnamed: 20","Unnamed: 21","Commodity type","S.no","Quantity of Goods","Vehicle Code","Location Code"],axis=1)

D.Frequency=D.Frequency.astype(int)
D["Type of Vehicle"]=D["Type of Vehicle"].astype(str)
D['Travel Time ()'] = D['Travel Time ()'].dt.total_seconds()

#EDA
print(CAssamOD.head())
print(CAssamOD.info())
print("The dimensions of the dataset are: ",D.shape)
print("The size of the dataset is: ",D.size)
stats=pd.DataFrame(D.describe())
print(stats)
print(CAssamOD.info())




AODP=CAssamOD[CAssamOD["Vehicle Code"]=="P"]
print(AODP.head())

AODDP=CAssamOD[CAssamOD["Road Name"]=="Dharapur-Palasbari Road"]
print(AODDP.head(20))
AODDP1=AODDP[["Vehicle Code","Type of Vehicle","Travel Distance(km)"]]

m=AODDP["Travel Distance(km)"].min()
ma=AODDP["Travel Distance(km)"].max()
print(AODDP1[AODDP1["Travel Distance(km)"]==m])
print(AODDP1[AODDP1["Travel Distance(km)"]==ma])


print(CAssamOD.groupby('Frequency').size())
print(CAssamOD.groupby('Vehicle Code').size())
print(CAssamOD.groupby('Purpose').size())
print(CAssamOD.groupby("Road Name").size())


AODDP=CAssamOD[CAssamOD["Road Name"]=="Dharapur-Palasbari Road"]
print(AODDP.head(20))
AODDP1=AODDP[["Vehicle Code","Type of Vehicle","Travel Distance(km)"]]
print(AODDP1)
print(AODDP1.groupby("Vehicle Code").size())
MONAOD=CAssamOD[CAssamOD["Day "]== "Monday"]
print(MONAOD.groupby("Vehicle Code").size())

#Plots and Graphs


#Scatter plot of Travel Distance vs. Travel Time
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Purpose', y='Day ', data=CAssamOD)
plt.title('Scatter plot of Purpose vs. Day')
plt.xlabel('Purpose')
plt.ylabel('Day')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(CAssamOD.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
CAssamOD=CAssamOD.drop(columns=["S.no"])
plt.figure(figsize=(12, 8))
sns.heatmap(CAssamOD.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
stats=pd.DataFrame(CAssamOD.describe())

#Histogram of Road Name w Type of Vehicle
ca= D['Type of Vehicle'] == 0

dff= D.drop(D[ca].index)
plt.figure(figsize=(10, 6))
sns.histplot(data=dff, x='Road Name', multiple='stack', hue='Type of Vehicle',bins=range(1, 27))
plt.title('Histogram of Road Name with Type of Vehicle')
plt.xlabel('Road Name')
plt.ylabel('Frequency')
plt.xticks(range(1,27))
plt.show()

#Scatter Plot of Travel Distance and Time of Survey
plt.figure(figsize=(10, 6))
plt.scatter(D['Time of Survey'],D['Travel Distance(km)'])
plt.title('Scatter Plot of Travel Distance vs Time of Survey')
plt.xlabel('Time of Survey')
plt.ylabel('Travel Distance (km)')
plt.xticks(range(1,25))
plt.legend()

#Histogram Of Type of Vehicle
order=["1","2",'3','4','5','6','7','8','9','10','11','12','13','14','15']
plt.figure(figsize=(10, 6))
sns.histplot(D["Type of Vehicle"],kde=True,palette="viridis")
plt.title('Histogram of Type of Vehicle')
plt.xlabel('Type of Vehicle')
plt.ylabel('Frequency')
plt.xticks( rotation=90)
plt.show()

#Barplot of Type of Vehicle vs Trip Frequency
plt.figure(figsize=(12, 6))
sns.barplot(x='Type of Vehicle', y='Frequency', data=D, palette='viridis')
plt.title('Bar Plot of Type of Vehicle vs. Trip Frequency')
plt.xlabel('Type of Vehicle')
plt.ylabel('Trip Frequency')
plt.xticks(rotation=90)
plt.show()

#Histogram of Road Name(can be replaced w any column)
plt.figure(figsize=(10, 6))
sns.histplot(D['Road Name'], kde=True, bins=range(1, 28))
plt.title('Histogram based on Road')
plt.xlabel('Road')
plt.ylabel('Frequency')
plt.xticks(range(1, 28), rotation=90)
plt.show()



#Testing Different Regression Models
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
D['Type of Vehicle'] = label_encoder.fit_transform(D['Type of Vehicle'])
x_exp = D["Type of Vehicle"]
y_exp = D["Travel Distance(km)"]
# 
# # Exponential function
# def exponential_func(x, a, b):
#     return a * np.exp(b * x)
# 
#params_exp, covariance_exp = curve_fit(exponential_func, x_exp, y_exp)
#x_fit_exp = np.linspace(min(x_exp), max(x_exp), 100)
#y_fit_exp = exponential_func(x_fit_exp, *params_exp)
# 
# plt.scatter(x_exp, y_exp, label='Data')
# plt.plot(x_fit_exp, y_fit_exp, label='Exponential Fit', color='red')
# plt.xlabel('Type of Vehicle')
# plt.ylabel('Travel Distance(km)')
# plt.legend()
# plt.title('Exponential Regression')
# plt.show()
# 
# =============================================================================
# =============================================================================
# # Quadratic function
# def quadratic_func(x, a, b, c):
#     return a * x**2 + b * x + c
# 
# params_quad, covariance_quad = curve_fit(quadratic_func, x_exp, y_exp)
# 
# y_fit_quad = quadratic_func(x_fit_exp, *params_quad)
# 
# plt.scatter(x_exp, y_exp, label='Data')
# plt.plot(x_fit_exp, y_fit_quad, label='Quadratic Fit', color='green')
# plt.xlabel('Type of Vehicle')
# plt.ylabel('Travel Distance(km)')
# plt.legend()
# plt.title('Quadratic Regression')
# plt.show()
# 
# =============================================================================
# =============================================================================
# # Logarithmic function
# 
# def logarithmic_func(x, a, b):
#     return a * np.log(x) + b
# 
# x_log = D["Type of Vehicle"]
# 
# params_log, covariance_log = curve_fit(logarithmic_func, x_log, y_exp, maxfev=2000, p0=[1, 4000])
# 
# y_fit_log = logarithmic_func(x_fit_exp, *params_log)
# 
# plt.scatter(x_log, y_exp, label='Data')
# plt.plot(x_fit_exp, y_fit_log, label='Logarithmic Fit', color='blue')
# plt.xlabel('Type of Vehicle')
# plt.ylabel('Travel Distance (km)')
# plt.legend()
# plt.title('Logarithmic Regression')
# plt.show()
# 
# =============================================================================

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = D[["Type of Vehicle"]]
y = D['Travel Distance(km)']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
plt.scatter(X_test.values.ravel(), y_test, label='Actual data')
plt.plot(X_test.values.ravel(), y_pred,"r-", label='Regression line')
plt.xlabel('Type of Vehicle')
plt.ylabel('Travel Distance(km)')
plt.legend()
plt.show()
slope = model.coef_[0]
intercept = model.intercept_

print(f"Slope (Coefficient): {slope}")
print(f"Intercept: {intercept}")










