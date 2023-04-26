---
layout: single
title: "타이타닉 생존여부 확인 by 로지스틱 회귀모델"
---


### 로지스틱 회귀모델을 이용해서 타이타닉호 승선객의 생존여부 확인하기 
##### age 결측치 값은 다음과 같이 처리 
  1. 생존자(survived = 1 )는 생존자 나이의 평균으로 대체
  2. 생존자(survived = 0 )는 사망자 나이의 평균으로 대체 


```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
```


```python
titanic = pd.read_csv("train.csv")
titanic.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB



```python
titanic.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>




```python
#변수제거
titanic_cln = titanic.drop(["PassengerId","Name","Ticket","Cabin"],axis=1)
titanic_cln.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic_cln.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 8 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   Survived  891 non-null    int64  
     1   Pclass    891 non-null    int64  
     2   Sex       891 non-null    object 
     3   Age       714 non-null    float64
     4   SibSp     891 non-null    int64  
     5   Parch     891 non-null    int64  
     6   Fare      891 non-null    float64
     7   Embarked  889 non-null    object 
    dtypes: float64(2), int64(4), object(2)
    memory usage: 55.8+ KB



```python
#결측치 처리 

#생존자 나이 평균 
sur1_mean = titanic_cln[titanic_cln["Survived"]==1]["Age"].mean()

#사망자 나이 평균 
sur0_mean = titanic_cln[titanic_cln["Survived"]==0]["Age"].mean()

print(sur1_mean, sur0_mean)
```

    28.343689655172415 30.62617924528302



```python
titanic_cln[titanic_cln["Survived"]==1]["Age"].fillna(sur1_mean)
titanic_cln[titanic_cln["Survived"]==0]["Age"].fillna(sur0_mean)
```




    0      22.000000
    4      35.000000
    5      30.626179
    6      54.000000
    7       2.000000
             ...    
    884    25.000000
    885    39.000000
    886    27.000000
    888    30.626179
    890    32.000000
    Name: Age, Length: 549, dtype: float64




```python
#결측치 처리는 loc로 
titanic_cln.loc[titanic_cln["Survived"]==1,"Age"] = titanic_cln[titanic_cln["Survived"]==1]["Age"].fillna(sur1_mean)
titanic_cln.loc[titanic_cln["Survived"]==0,"Age"] = titanic_cln[titanic_cln["Survived"]==0]["Age"].fillna(sur0_mean)
```


```python
#age 부분 다 채워진거 확인
titanic_cln.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 8 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   Survived  891 non-null    int64  
     1   Pclass    891 non-null    int64  
     2   Sex       891 non-null    object 
     3   Age       891 non-null    float64
     4   SibSp     891 non-null    int64  
     5   Parch     891 non-null    int64  
     6   Fare      891 non-null    float64
     7   Embarked  889 non-null    object 
    dtypes: float64(2), int64(4), object(2)
    memory usage: 55.8+ KB



```python
titanic_cln["Embarked"].value_counts()
```




    S    644
    C    168
    Q     77
    Name: Embarked, dtype: int64




```python
print(titanic_cln["Embarked"].value_counts().index[0])
#s가 제일 많으니까 나머지 결측값에도 s를 넣어줄것
```

    S



```python
titanic_cln["Embarked"] = titanic_cln["Embarked"].fillna(titanic_cln["Embarked"].value_counts().index[0])
```


```python
titanic_cln.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 8 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   Survived  891 non-null    int64  
     1   Pclass    891 non-null    int64  
     2   Sex       891 non-null    object 
     3   Age       891 non-null    float64
     4   SibSp     891 non-null    int64  
     5   Parch     891 non-null    int64  
     6   Fare      891 non-null    float64
     7   Embarked  891 non-null    object 
    dtypes: float64(2), int64(4), object(2)
    memory usage: 55.8+ KB



```python
print(titanic_cln["Pclass"].value_counts())
print(titanic_cln["Sex"].value_counts())
print(titanic_cln["Embarked"].value_counts())
```

    3    491
    1    216
    2    184
    Name: Pclass, dtype: int64
    0    577
    1    314
    Name: Sex, dtype: int64
    S    646
    C    168
    Q     77
    Name: Embarked, dtype: int64



```python
#Pclass, sex, Emabarked 변수 타입 변환
```


```python
#sex 를 female -> 1 male -> 0 으로 변환 
titanic_cln["Sex"] = titanic_cln["Sex"].replace(["female","male"],[1,0])
titanic_cln.head()
#sex 변수 변환됨
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
#3개의 카테고리로 만들어줄것
titanic_cln["Pclass"]=titanic_cln["Pclass"].astype("category") 
titanic_cln["Embarked"]=titanic_cln["Embarked"].astype("category") 
```


```python
titanic_cln.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 8 columns):
     #   Column    Non-Null Count  Dtype   
    ---  ------    --------------  -----   
     0   Survived  891 non-null    int64   
     1   Pclass    891 non-null    category
     2   Sex       891 non-null    int64   
     3   Age       891 non-null    float64 
     4   SibSp     891 non-null    int64   
     5   Parch     891 non-null    int64   
     6   Fare      891 non-null    float64 
     7   Embarked  891 non-null    category
    dtypes: category(2), float64(2), int64(4)
    memory usage: 43.9 KB



```python
titanic_cln_dum = pd.get_dummies(titanic_cln)
titanic_cln_dum.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Pclass_1</th>
      <th>Pclass_2</th>
      <th>Pclass_3</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic_cln_dum = sm.add_constant(titanic_cln_dum, has_constant="add")
titanic_cln_dum.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>const</th>
      <th>const</th>
      <th>Survived</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Pclass_1</th>
      <th>Pclass_2</th>
      <th>Pclass_3</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
feature_columns = list(titanic_cln_dum.columns.difference(["Survived"]))


X = titanic_cln_dum[feature_columns]
y = titanic_cln_dum["Survived"]
```


```python
x_train, x_test, y_train, y_test = train_test_split(X,y,
                                                   train_size=0.7, test_size =0.3,
                                                   random_state=102)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
```

    (623, 13) (268, 13) (623,) (268,)



```python
model = sm.Logit(y_train, x_train)
results = model.fit(method="newton")
```

    Optimization terminated successfully.
             Current function value: 0.441385
             Iterations 7



```python
results.summary()
```




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>Survived</td>     <th>  No. Observations:  </th>  <td>   623</td>  
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td>   613</td>  
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>     9</td>  
</tr>
<tr>
  <th>Date:</th>            <td>Tue, 26 Jul 2022</td> <th>  Pseudo R-squ.:     </th>  <td>0.3386</td>  
</tr>
<tr>
  <th>Time:</th>                <td>15:58:31</td>     <th>  Log-Likelihood:    </th> <td> -274.98</td> 
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -415.74</td> 
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th> <td>2.171e-55</td>
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Age</th>        <td>   -0.0461</td> <td>    0.009</td> <td>   -4.941</td> <td> 0.000</td> <td>   -0.064</td> <td>   -0.028</td>
</tr>
<tr>
  <th>Embarked_C</th> <td>    0.3284</td> <td> 4.11e+07</td> <td> 7.99e-09</td> <td> 1.000</td> <td>-8.06e+07</td> <td> 8.06e+07</td>
</tr>
<tr>
  <th>Embarked_Q</th> <td>    0.1857</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan</td> <td>      nan</td>
</tr>
<tr>
  <th>Embarked_S</th> <td>   -0.3197</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan</td> <td>      nan</td>
</tr>
<tr>
  <th>Fare</th>       <td>    0.0048</td> <td>    0.004</td> <td>    1.348</td> <td> 0.178</td> <td>   -0.002</td> <td>    0.012</td>
</tr>
<tr>
  <th>Parch</th>      <td>   -0.0910</td> <td>    0.140</td> <td>   -0.649</td> <td> 0.516</td> <td>   -0.366</td> <td>    0.184</td>
</tr>
<tr>
  <th>Pclass_1</th>   <td>    1.0648</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan</td> <td>      nan</td>
</tr>
<tr>
  <th>Pclass_2</th>   <td>    0.1762</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan</td> <td>      nan</td>
</tr>
<tr>
  <th>Pclass_3</th>   <td>   -1.0466</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan</td> <td>      nan</td>
</tr>
<tr>
  <th>Sex</th>        <td>    2.5486</td> <td>    0.239</td> <td>   10.675</td> <td> 0.000</td> <td>    2.081</td> <td>    3.017</td>
</tr>
<tr>
  <th>SibSp</th>      <td>   -0.4101</td> <td>    0.136</td> <td>   -3.020</td> <td> 0.003</td> <td>   -0.676</td> <td>   -0.144</td>
</tr>
<tr>
  <th>const</th>      <td>    0.1944</td> <td> 3.45e+15</td> <td> 5.64e-17</td> <td> 1.000</td> <td>-6.75e+15</td> <td> 6.75e+15</td>
</tr>
<tr>
  <th>const</th>      <td>    0.1944</td> <td> 1.25e+15</td> <td> 1.56e-16</td> <td> 1.000</td> <td>-2.44e+15</td> <td> 2.44e+15</td>
</tr>
</table>




```python
results.params
```




    Age          -0.046104
    Embarked_C    0.328389
    Embarked_Q    0.185698
    Embarked_S   -0.319655
    Fare          0.004812
    Parch        -0.091043
    Pclass_1      1.064840
    Pclass_2      0.176165
    Pclass_3     -1.046573
    Sex           2.548639
    SibSp        -0.410117
    const         0.194432
    const         0.194432
    dtype: float64




```python
np.exp(results.params)
#pclass가 1인경우, 생존할 확률이 2배 높음
#변수의 범주가 나뉘지는 경우에는 카테고리로 나눠서 할것 
```




    Age            0.954943
    Embarked_C     1.388730
    Embarked_Q     1.204058
    Embarked_S     0.726400
    Fare           1.004824
    Parch          0.912979
    Pclass_1       2.900376
    Pclass_2       1.192634
    Pclass_3       0.351139
    Sex           12.789688
    SibSp          0.663573
    const          1.214621
    const          1.214620
    dtype: float64




```python
results.aic
```




    569.9662477300736




```python
y_pred = results.predict(x_test)
y_pred
```




    618    0.868298
    849    0.954464
    235    0.548751
    865    0.715086
    731    0.321682
             ...   
    259    0.627825
    760    0.089515
    557    0.812292
    638    0.358194
    504    0.966472
    Length: 268, dtype: float64




```python
def PRED(y,threshold):
    Y= y.copy()
    Y[y > threshold] = 1
    Y[y <= threshold] = 0
    return(Y.astype(int))

Y_pred = PRED(y_pred,0.5)
Y_pred
```




    618    1
    849    1
    235    1
    865    1
    731    0
          ..
    259    1
    760    0
    557    1
    638    0
    504    1
    Length: 268, dtype: int64




```python
cfmat = confusion_matrix(y_test, Y_pred)
print(cfmat)
```

    [[141  26]
     [ 26  75]]



```python
def acc(cfmat):
    acc =(cfmat[0,0]+cfmat[1,1]/np.sum(cfmat))
    return(acc)

acc(cfmat)
```




    141.27985074626866




```python
threshold = np.arange(0,1,0.1)
table = pd.DataFrame(columns=["ACC"])

for i in threshold:
    Y_pred = PRED(y_pred,i)
    cfmat= confusion_matrix(y_test, Y_pred)
    table.loc[i] = acc(cfmat)
    
table.index.name="threshold"
table.columns.name="performance"
table

#0.6일때 정확도가 가장 높아짐
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>performance</th>
      <th>ACC</th>
    </tr>
    <tr>
      <th>threshold</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>0.376866</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>52.358209</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>105.332090</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>121.313433</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>135.294776</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>141.279851</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>149.257463</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>161.190299</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>164.134328</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>166.082090</td>
    </tr>
  </tbody>
</table>
</div>




```python
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred, pos_label=1)
plt.plot(fpr,tpr)

auc=np.trapz(tpr,fpr)
print("AUC: ",auc)
```

    AUC:  0.859370368174542



    
![output_34_1](https://user-images.githubusercontent.com/91577581/234586339-53314b71-2a58-49ff-9d33-775b64ab98af.png)

    


# 회귀계수 축소모형을 이용해서 타이타닉호 승선객의 생존여부 확인 


```python
from sklearn.linear_model import Ridge,Lasso,ElasticNet
```


```python
lasso = Lasso(alpha = 0.01)
lasso.fit(x_train,y_train)
```




    Lasso(alpha=0.01)




```python
lasso.coef_
```




    array([-0.00602402,  0.        ,  0.        , -0.05284511,  0.00103671,
           -0.00212009,  0.08610017, -0.        , -0.16710225,  0.42002796,
           -0.04601298,  0.        ,  0.        ])




```python
pred_y_lasso = lasso.predict(x_test)
pred_Y_lasso = PRED(pred_y_lasso, 0.5)
```


```python
cfmat = confusion_matrix(y_test, pred_Y_lasso)
print(acc(cfmat))
```

    145.26865671641792



```python
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_y_lasso)
plt.plot(fpr,tpr)

auc=np.trapz(tpr,fpr)
print("AUC: ", auc)
```

    AUC:  0.862571885931108



    
![output_41_1](https://user-images.githubusercontent.com/91577581/234587708-6e9a6e67-d529-410e-b46f-85343df7da6f.png)

    



```python
threshold = np.arange(0,1,0.1)
table = pd.DataFrame(columns = ["ACC"])

for i in threshold:
    Y_pred = PRED(pred_y_lasso, i)
    cfmat = confusion_matrix(y_test, Y_pred)
    table.loc[i]=acc(cfmat)
    
    
table.index.name = "threshold"
table.columns.name="performance"
table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>performance</th>
      <th>ACC</th>
    </tr>
    <tr>
      <th>threshold</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>2.376866</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>21.373134</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>93.332090</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>116.320896</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>137.298507</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>145.268657</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>161.235075</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>165.145522</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>166.097015</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>166.041045</td>
    </tr>
  </tbody>
</table>
</div>




```python
alpha = np.logspace(-3,1,5)
alpha
```




    array([1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01])




```python
data = []
acc_table = []


for i,a in enumerate(alpha):
    lasso = Lasso(alpha=a).fit(x_train,y_train)
    data.append(pd.Series(np.hstack([lasso.intercept_, lasso.coef_])))
    y_pred = lasso.predict(x_test)
    y_pred =PRED(y_pred,0.5)
    cfmat = confusion_matrix(y_test,y_pred)
    acc_table.append((acc(cfmat)))
    
    
df_lasso = pd.DataFrame(data, index= alpha).T
df_lasso
#라소를 적용한 모형의 회귀계수
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.001</th>
      <th>0.010</th>
      <th>0.100</th>
      <th>1.000</th>
      <th>10.000</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.557221</td>
      <td>0.519114</td>
      <td>0.414021</td>
      <td>0.314982</td>
      <td>0.386838</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.006679</td>
      <td>-0.006024</td>
      <td>-0.004045</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.018281</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.077428</td>
      <td>-0.052845</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.000619</td>
      <td>0.001037</td>
      <td>0.002525</td>
      <td>0.002147</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.014105</td>
      <td>-0.002120</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.140224</td>
      <td>0.086100</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.190047</td>
      <td>-0.167102</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.458285</td>
      <td>0.420028</td>
      <td>0.025301</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>-0.045830</td>
      <td>-0.046013</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
acc_table_lasso = pd.DataFrame(acc_table, index= alpha).T
acc_table_lasso
#람다값이 0.01일때 정확도가 제일 높음
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.001</th>
      <th>0.010</th>
      <th>0.100</th>
      <th>1.000</th>
      <th>10.000</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>142.272388</td>
      <td>145.268657</td>
      <td>159.059701</td>
      <td>162.048507</td>
      <td>167.0</td>
    </tr>
  </tbody>
</table>
</div>



# Ridge


```python
ridge = Ridge(alpha = 0.01)
ridge.fit(x_train,y_train)
```




    Ridge(alpha=0.01)




```python
ridge.coef_
```




    array([-0.00674968,  0.04090873,  0.01910686, -0.06001559,  0.00057219,
           -0.0154551 ,  0.16156199,  0.01545708, -0.17701907,  0.46253901,
           -0.04580177,  0.        ,  0.        ])




```python
pred_y_ridge = ridge.predict(x_test)
pred_Y_ridge = PRED(pred_y_ridge, 0.5)
```


```python
cfmat = confusion_matrix(y_test, pred_Y_ridge)
print(acc(cfmat))
```

    142.27611940298507



```python
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_Y_ridge)
plt.plot(fpr,tpr)

auc=np.trapz(tpr,fpr)
print("AUC: ", auc)
```

    AUC:  0.7914863342621689



    
![output_51_1](https://user-images.githubusercontent.com/91577581/234587844-e2e137c0-7bdf-411c-987c-dbb38d476d1b.png)

    



```python
threshold = np.arange(0,1,0.1)
table = pd.DataFrame(columns = ["ACC"])

for i in threshold:
    Y_pred = PRED(pred_Y_ridge, i)
    cfmat = confusion_matrix(y_test, Y_pred)
    table.loc[i]=acc(cfmat)
    
    
table.index.name = "threshold"
table.columns.name="performance"
table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>performance</th>
      <th>ACC</th>
    </tr>
    <tr>
      <th>threshold</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>142.276119</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>142.276119</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>142.276119</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>142.276119</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>142.276119</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>142.276119</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>142.276119</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>142.276119</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>142.276119</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>142.276119</td>
    </tr>
  </tbody>
</table>
</div>



# ElasticNet


```python
elastic = ElasticNet(alpha = 0.1, l1_ratio=0.5)
elastic.fit(x_train,y_train)
```




    ElasticNet(alpha=0.1)




```python
elastic.coef_
```




    array([-0.00401088,  0.        ,  0.        , -0.        ,  0.0023124 ,
           -0.        ,  0.        ,  0.        , -0.00783898,  0.20741403,
           -0.02231027,  0.        ,  0.        ])




```python
pred_y_elastic = elastic.predict(x_test)
pred_Y_elastic = PRED(pred_y_ridge, 0.5)
#0.5 기준으로 1과 0 판단
```


```python
cfmat = confusion_matrix(y_test, pred_Y_elastic)
print(acc(cfmat))
```

    142.27611940298507



```python
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_Y_elastic)
plt.plot(fpr,tpr)

auc=np.trapz(tpr,fpr)
print("AUC: ", auc)
```

    AUC:  0.7914863342621689



    
![output_58_1](https://user-images.githubusercontent.com/91577581/234588089-2fd8de29-4391-44c8-b5d4-b2ecb16c6bf0.png)

    

