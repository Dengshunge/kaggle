import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model

# 填充缺失的年龄
def set_missing_ages(df):
    age_df = df[['Age','Fare','Parch', 'SibSp', 'Pclass']]# 用这几个变量来对年龄进行预测
    known_age = age_df[age_df.Age.notnull()].values # 转化为array
    unknown_age = age_df[age_df.Age.isnull()].values
    train_X,train_Y = known_age[:,1:] , known_age[:,0]

    # 训练
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(train_X,train_Y)

    # 预测
    predict_Ages = rfr.predict(unknown_age[:,1:])

    # 用预测的数据填充缺失值
    df.loc[df.Age.isnull(),'Age'] = predict_Ages

    return df,rfr

def set_Cabin_type(df):
    df.loc[df.Cabin.notnull(),'Cabin'] = 'Yes'
    df.loc[df.Cabin.isnull(), 'Cabin'] = 'No'
    return df

data_train = pd.read_csv(r'C:\Users\Administrator\Desktop\Titanic\train.csv')
data_train,rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)

# 特征因子化
dummies_Cabin = pd.get_dummies(data_train['Cabin'],prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'],prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

# 拼接
df = pd.concat([data_train,dummies_Cabin,dummies_Embarked,dummies_Pclass,dummies_Sex],axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'],axis = 1, inplace = True)

# 数据归一化
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(pd.DataFrame(df['Age']))
df['Age_scaled'] = scaler.fit_transform(pd.DataFrame(df['Age']), age_scale_param)
fare_scale_param = scaler.fit(pd.DataFrame(df['Fare']))
df['Fare_scaled'] = scaler.fit_transform(pd.DataFrame(df['Fare']), fare_scale_param)

# LR模型
# 提取需要的特征
train_df = df.filter(regex = 'Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.values

y = train_np[:,0]
x = train_np[:,1:]

lr = linear_model.LogisticRegression()
lr.fit(x,y)

# 对结果进行预测
# 首先要对test进行格式的统一
data_test = pd.read_csv(r'C:\Users\Administrator\Desktop\Titanic\test.csv')
data_test.loc[(data_test.Fare.isnull()),'Fare'] = 0

tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].values
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

df_test['Age_scaled'] = scaler.fit_transform(pd.DataFrame(df_test['Age']), age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(pd.DataFrame(df_test['Fare']), fare_scale_param)

# 预测结果
test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = lr.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv(r'C:\Users\Administrator\Desktop\Titanic\prediction.csv', index=False)
