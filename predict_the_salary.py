# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

#import dataset
df = pd.read_csv("D:/PYTHON/test/adult.csv")

#get basic information
df.info()
stats = pd.DataFrame(df.describe())
df.isnull().sum()
df['education'].value_counts()
df['workclass'].value_counts()
df["educational-num"].value_counts()
df["marital-status"].value_counts()
df["occupation"].value_counts()
df["native-country"].value_counts()
df['income'].value_counts()

###EDA####
#count plot of every column

def plot(column):
  fig_1 = px.histogram(x=df[column])
  fig_1.show()
  
plot("capital-gain")

#countplot by group
def plot_2(col_1 , col_2):
  fig_2 = px.histogram(x=df[col_1] , color=df[col_2])
  fig_2.update_layout(barmode = "group")
  fig_2.show()
  
plot_2("age" , "hours-per-week")

#sunburtplot 
def plot_3(col_1_1 , col_2_1):
  fig_3 = px.sunburst(df , path=[col_1_1,col_2_1])
  fig_3.show()

plot_3("age" , "gender")

#treemap
def plot_4(col_x , col_y):
  fig_4 = px.treemap(df , path=[col_x , col_y])
  fig_4.show()

plot_4("race" , "education")

#columns to keep
keeps_cloumns = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss','hours-per-week']
df[keeps_cloumns]

#split data into feature and target data set
x = df[keeps_cloumns].values
y = df['income'].values

#create partition 
x_train , x_test ,y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 10)

#build the model 
gb_class = GradientBoostingClassifier(n_estimators = 200 , learning_rate = 0.5 , max_features=1 , subsample = 1 , max_depth = 3 , min_impurity_decrease = 0)
gb_class.fit(x_train , y_train)


#accuray
print('accuracy score (training) : ',gb_class.score(x_train,y_train))
print('accuracy score (test) : ',gb_class.score(x_test,y_test))


#test the model
person =[[66,186061,10,0,4356,40]]
person =pd.DataFrame(person)

predict = gb_class.predict(person)
print(predict)








