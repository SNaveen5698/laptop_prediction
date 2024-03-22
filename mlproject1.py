import pandas as pd
df=pd.read_excel("C:\\Users\\yello\\Downloads\\Health_insurance_cost.xlsx")
print(df.head())
print(df.isnull().sum())
df["age"].fillna(df["age"].median(),inplace=True)
df["BMI"].fillna(df["BMI"].median(),inplace=True)
df["health_insurance_price"].fillna(df["health_insurance_price"].median(),inplace=True)
df["smoking_status"]=df["smoking_status"].map({"yes":1,"no":0})
df=pd.get_dummies(df,columns=["location","gender"],drop_first=True)
print(df.isnull().sum())
x=df.drop("health_insurance_price",axis=1)
y=df["health_insurance_price"]

from sklearn.model_selection import train_test_split
a,b,c,d=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
obj1=LinearRegression()
obj1.fit(a,c)
ycap1=obj1.predict(b)
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print("mse using linear regression",mean_squared_error(d,ycap1))
print("rmse using linear regression",np.sqrt(mean_squared_error(d,ycap1)))
print("mae using linear regression",mean_absolute_error(d,ycap1))
print("r2_score using linear regression",r2_score(d,ycap1))

from sklearn.tree import DecisionTreeRegressor
obj2=DecisionTreeRegressor()
obj2.fit(a,c)
ycap2=obj2.predict(b)
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print("mse using desion tree regression",mean_squared_error(d,ycap2))
print("rmse using desion tree regressor",np.sqrt(mean_squared_error(d,ycap2)))
print("mae using desioncion tree regression",mean_absolute_error(d,ycap2))
print("r2_score using desicion tree regression",r2_score(d,ycap2))

from sklearn.ensemble import RandomForestRegressor
obj3=RandomForestRegressor()
obj3.fit(a,c)
ycap3=obj3.predict(b)
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print("mse using RFR",mean_squared_error(d,ycap3))
print("mae using RFR",mean_absolute_error(d,ycap3))
print("rmse using RFR",np.sqrt(mean_squared_error(d,ycap3)))
print("r2_score usning RFR",r2_score(d,ycap3))
from sklearn.neighbors import KNeighborsRegressor
obj4=KNeighborsRegressor()
obj4.fit(a,c)
ycap4=obj4.predict(b)
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print("mse using KNN",mean_squared_error(d,ycap4))
print("rmse using KNN",np.sqrt(mean_squared_error(d,ycap4)))
print("mas usiing KNN",mean_absolute_error(d,ycap4))
print("r2score using KNN",r2_score(d,ycap4))

from sklearn.svm import SVR
obj5=SVR()
obj5.fit(a,c)
ycap5=obj5.predict(b)
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print("mse using svr",mean_squared_error(d,ycap5))
print("mae using svr",mean_absolute_error(d,ycap5))
print("rmse using svr",np.sqrt(mean_squared_error(d,ycap5)))
print("r2score using svr",r2_score(d,ycap5))