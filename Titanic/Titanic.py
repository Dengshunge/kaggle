# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

# 对缺失的年龄进行RF补充
def set_missing_ages(df):
    # 提取已有的数值特征出来，放进RF模型中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    
    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values
    
    y = known_age[:,0]
    x = known_age[:,1:]
    
    # 训练模型
    rfr = RandomForestRegressor()
    rfr.fit(x,y)
    #预测结果
    predictedAges = rfr.predict(unknown_age[:,1:])
    
    #使用得到的预测值来填补缺失的年龄
    df.loc[df.Age.isnull(),'Age'] = predictedAges
    
    return rfr

# 对缺失的Cabin进行填充
def set_Cabin_type(df):
    df.loc[df.Cabin.notnull(),'Cabin'] = 'yes'
    df.loc[df.Cabin.isnull(),'Cabin'] = 'no'
    
# 利用众数对缺失的Embarked进行填充
def set_missing_Embarked(df):
    df.Embarked.fillna('S',inplace = True)
    
# 对特征进行特征因子化 
def set_dummies(df):
    dummies_Cabin = pd.get_dummies(df.Cabin,prefix = 'Cabin')
    dummies_Sex = pd.get_dummies(df.Sex,prefix='Sex')
    dummies_Embarked = pd.get_dummies(df.Embarked,prefix='Embarked')
    dummies_Pclass = pd.get_dummies(df.Pclass,prefix='Pclass')
    df = pd.concat([df,dummies_Cabin,dummies_Embarked,dummies_Pclass,dummies_Sex],axis=1)
    df.drop(['Pclass','Sex','Embarked','Cabin'],axis=1,inplace=True)
    return df

def data_normalization(df):
    import sklearn.preprocessing as preprocessing
    scaler = preprocessing.StandardScaler()
    age_scale_param = scaler.fit(DataFrame(df.Age))
    df['Age_scaled'] = scaler.fit_transform(DataFrame(df.Age),age_scale_param)
    fare_scale_param = scaler.fit(DataFrame(df.Fare))
    df['Fare_scaled'] = scaler.fit_transform(DataFrame(df.Fare),fare_scale_param)
    df.drop(['Age','Fare'],inplace=True,axis=1)
    
def train_preprocessing(df):
    rfr = set_missing_ages(df)
    set_Cabin_type(df)
    set_missing_Embarked(df)
    df = set_dummies(df)
    data_normalization(df)
    return df,rfr

def test_preprocessing(df,rfr):
    # 对缺失的年龄进行填充
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    unknown_age = age_df[age_df.Age.isnull()].values
    predictedAges = rfr.predict(unknown_age[:,1:])
    df.loc[df.Age.isnull(),'Age'] = predictedAges
    
    set_Cabin_type(df)
    set_missing_Embarked(df)
    df = set_dummies(df)
    df.Fare.fillna(df.Fare.mean(),inplace=True)
    data_normalization(df)
    return df
    
data_train = pd.read_csv(r'F:\project\pycharm_project\ML\Titanic\train.csv')

# 对数据进行处理
data_train,rfr = train_preprocessing(data_train)

# 对数据进行LR建模
# 首先提取需要的特征
train_np = data_train.filter(regex = 'Survived|Age_.*\
                             |SibSp|Parch|Fare_.*|Cabin_.*\
                             |Embarked_.*|Sex_.*|Pclass_.*').values
clf = linear_model.LogisticRegression()
#clf.fit(train_np[:,1:],train_np[:,0])
#
## 对测试数据进行处理
#data_test = pd.read_csv(r'F:\project\pycharm_project\ML\Titanic\test.csv')
#data_test = test_preprocessing(data_test,rfr)
#
## 对测试数据进行预测
#test = data_test.filter(regex = 'Survived|Age_.*\
#                             |SibSp|Parch|Fare_.*|Cabin_.*\
#                             |Embarked_.*|Sex_.*|Pclass_.*').values
#prediction = clf.predict(test)
#result = DataFrame({'PassengerId':data_test.PassengerId.values,'Survived':\
#                    prediction})
#result.to_csv(r'F:\project\pycharm_project\ML\Titanic\prediction7-14.csv',index=False)

# 利用训练集做交叉验证
train_x,train_y = train_np[:,1:],train_np[:,0]
score = cross_val_score(clf,train_x,train_y,cv=10)
print(score.mean())



# 以下是画图分析
#plt.figure(1)
#
#plt.subplot2grid((2,3),(0,0))
#data_train['Survived'].value_counts().plot(kind='bar')
#plt.title('获救情况（1为获救）')
#plt.ylabel('人数')
#plt.xticks(rotation=360)# 使X轴的标签正着显示，不然会有旋转
#
#plt.subplot2grid((2,3),(0,1))
#data_train['Pclass'].value_counts().plot(kind='bar')
#plt.title('乘客等级')
#plt.ylabel('人数')
#plt.xticks(rotation=360)
#
#plt.subplot2grid((2,3),(0,2))
#data_train['Sex'].value_counts().plot(kind='bar')
#plt.title('性别')
#plt.ylabel('人数')
#plt.xticks(rotation=360)
#
#plt.subplot2grid((2,3),(1,0),colspan=2)
#data_train['Age'].where(data_train['Pclass'] == 1).plot(kind='kde')
#data_train['Age'].where(data_train['Pclass'] == 2).plot(kind='kde')
#data_train['Age'].where(data_train['Pclass'] == 3).plot(kind='kde')
#plt.xlabel('年龄')
#plt.ylabel('密度')
#plt.title('每个乘客等级的年龄分布情况')
#plt.legend(('头等舱','二等舱','三等舱'),loc='best')
#plt.grid(b=True)# 开启网格线
#
#plt.subplot2grid((2,3),(1,2))
#data_train['Embarked'].value_counts().plot(kind='bar')
#plt.title('上船地点')
#plt.ylabel('人数')
#plt.xticks(rotation=360)

#plt.figure()

#Survived_Pclass_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
#Survived_Pclass_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
#df = DataFrame({'获救':Survived_Pclass_0,'未获救':Survived_Pclass_1})
#df.plot(kind='bar',stacked = True)
#plt.title('各舱位获救与未获救人数对比图')
#plt.ylabel('人数')
#plt.xlabel('乘客等级')
#plt.xticks(rotation=360)

#Survived_Female = data_train.Survived[data_train.Sex == 'female'].value_counts()
#Survived_Male = data_train.Survived[data_train.Sex == 'male'].value_counts()
#df = DataFrame({'女性':Survived_Female,'男性':Survived_Male})
#df.plot(kind='bar',stacked = True)
#plt.title('性别获救与未获救人数对比图')
#plt.ylabel('人数')
#plt.xlabel('性别')
#plt.xticks(rotation=360)

#Survived_Embarked_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
#Survived_Embarked_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
#df = DataFrame({'获救':Survived_Embarked_0,'未获救':Survived_Embarked_1})
#df.plot(kind='bar',stacked = True)
#plt.title('上船地点与获救情况对比图')
#plt.ylabel('人数')
#plt.xlabel('上船地点')
#plt.xticks(rotation=360)
