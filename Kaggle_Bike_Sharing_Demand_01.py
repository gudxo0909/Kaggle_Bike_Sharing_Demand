# import module

import pandas as pd
import numpy as np
import seaborn as sns

train = pd.read_csv('train.csv')
print(train.shape)
train.head()

test = pd.read_csv('test.csv')
print(test.shape)
test.head()

train.dtypes

# 문자형인 datetime열을 datetime 형태로 변경

train['datetime'] = pd.to_datetime(train['datetime'])
train['datetime'].head()

# 연, 월, 일, 시, 분, 초 변수 생성, 시계열 데이터에서 필요한 값 추출

train['Year'] = train['datetime'].dt.year
train['Month'] = train['datetime'].dt.month
train['Day'] = train['datetime'].dt.day
train['Hour'] = train['datetime'].dt.hour
train['Minute'] = train['datetime'].dt.minute
train['Second'] = train['datetime'].dt.second
train[['datetime', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']].head()

train['Dayofweek'] = train['datetime'].dt.dayofweek
train['Dayname'] = train['datetime'].dt.day_name()
train[['datetime', 'Dayofweek', 'Dayname']].head()

test['datetime'] = pd.to_datetime(test['datetime'])
test['Year'] = test['datetime'].dt.year
test['Month'] = test['datetime'].dt.month
test['Day'] = test['datetime'].dt.day
test['Hour'] = test['datetime'].dt.hour
test['Minute'] = test['datetime'].dt.minute
test['Second'] = test['datetime'].dt.second
test[['datetime', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']].head()

test['Dayofweek'] = test['datetime'].dt.dayofweek
test['Dayname'] = test['datetime'].dt.day_name()
test[['datetime', 'Dayofweek', 'Dayname']].head()

# 연도 데이터 시각화

sns.barplot(data = train, x = 'Year', y = 'count')

# 월 데이터 시각화

sns.barplot(data = train, x = 'Month', y = 'count')

# 일 데이터 시각화

sns.barplot(data = train, x = 'Day', y = 'count')
sns.countplot(data = test, x = 'Day')

'''train 데이터의 일 데이터가 19일 까지만 나타나 있음.
   count 변수가 없는 test 데이터를 확인한 결과, 20일 ~ 31일까지의 데이터가 있음.
   여기서 일 데이터가 train과 test를 나누는 변수라고 생각하지만 일 별 대여량의
   차이가 크지 않으므로 사용하지 않음.'''

# 시 데이터 시각화

sns.barplot(data = train, x = 'Hour' , y = 'count')
'''시 데이터를 시각화 해본 결과, 출. 퇴근 시간의 자전거 대여량이 많다는 것을 확인 할 수 있음.'''

# 분 데이터 시각화

sns.barplot(data = train, x = 'Minute', y = 'count')
train['Minute'].value_counts()
'''시각화를 한 결과 값이 0 하나뿐이라 value_counts() 기능을 이용해 확인한 결과
   정말 값이 0 하나임. 따라서 분석에 사용하지 않음.'''

# 초 데이터 시각화

sns.barplot(data = train, x = 'Second', y = 'count')
'''초 단위도 마찬가지로 값이 0 하나 뿐임.'''

# 요일 데이터 시각화

sns.barplot(data = train, x = 'Dayname', y = 'count')
'''요일 데이터 시각화 결과 요일 별 대여량에는 큰 차이가 없음.'''

# 한 번에 여러개의 그래프를 그리기 위해 matplotlib library 사용

import matplotlib.pyplot as plt

# 4개의 그래프를 만들기 위해 2행 2열의 틀 만들기

figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2)

figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2)
sns.barplot(data = train, x = 'Year', y = 'count', ax = ax1)
sns.barplot(data = train, x = 'Month', y = 'count', ax = ax2)
sns.barplot(data = train, x = 'Hour', y = 'count', ax = ax3)
sns.barplot(data = train, x = 'Dayname', y = 'count', ax = ax4)

# 그래프 사이즈 키우기

figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2)
figure.set_size_inches(14, 8)
sns.barplot(data = train, x = 'Year', y = 'count', ax = ax1)
sns.barplot(data = train, x = 'Month', y = 'count', ax = ax2)
sns.barplot(data = train, x = 'Hour', y = 'count', ax = ax3)
sns.barplot(data = train, x = 'Dayname', y = 'count', ax = ax4)

# 시간 데이터 세세한 시각화

plt.figure(figsize = (12, 4))
sns.pointplot(data = train, x = 'Hour', y = 'count', hue = 'holiday')

train['holiday'].value_counts()
'''시각화 결과 휴일이 1인 값, 즉, 휴일인 날의 점 부분에 막대가 많은 것으로 보아 휴일인 날의 데이터가 부족한 것을 알 수 있음.
   value_count()로 확인 한 결과 휴일의 데이터 분포가 불균형함을 확인.'''

plt.figure(figsize = (12, 4))
sns.pointplot(data = train, x = 'Hour', y = 'count', hue = 'workingday')
''' workingday 데이터는 holiday 데이터보다 훨씬 분포가 잘 어루어져있음을 확인.
    workingday가 1인 데이터, 즉, 근무일에 해당하는 데이터는 출퇴근 시간에 대여량이 많고,
    workingday가 0인 데이터, 즉, 휴일에는 점심 시간을 포함해 주로 오후 시간대에 대여량이 많음을 확인.
    이러한 정보들을 토대로 휴무 여부에 따라 자전거 대여소에 자전거 준비량을 시간대 별로 다르게 하는게 좋다는 추측을 해볼 수 있다.'''

# 요일을 기준으로 시간대별 대여량 확인

plt.figure(figsize = (12, 4))
sns.pointplot(data = train, x = 'Hour', y = 'count', hue = 'Dayname')
''' 같은 휴무날임에도 토요일과 일요일의 대여량에 차이가 있음. 토요일이 일요일보다 대여량이 많음.'''

# 계절 데이터 시각화

sns.barplot(data = train, x = 'season', y = 'count')
'''season 데이터 =>1 -> 봄, 2 -> 여름, 3 -> 가을, 4 -> 겨울
   여름과 가을에 대여량이 많고, 겨울의 대여량이 생각보다 많으며, 봄에는 대여량이 가장 적다.
   일반적으로는 겨울에 비해 따뜻한 봄에 대여량이 많을 것 같은데 의외의 결과가 나와서 season 데이터를 확인'''

train[['Month', 'season']]
train.loc[train['Month'] == 2, ['Month', 'season']]
'''월 별 season 데이터를 확인해보니 1월과 2월이 봄으로 입력되어 있음.
   일반적인 3 ~ 5월을 봄으로, 6 ~ 8월을 여름으로, 9 ~ 11월을 가을로, 12 ~ 2월을 겨울로 재정의'''

def general_season(month):
    if month in [3, 4, 5]:
        return 1
    elif month in [6, 7, 8]:
        return 2
    elif month in [9, 10, 11]:
        return 3
    else:
        return 4

train['season'] = train['Month'].apply(general_season)
train[['Month', 'season']]

# test 데이터에도 적용

test['season'] = test['Month'].apply(general_season)
test[['Month', 'season']]

sns.barplot(data = train, x = 'season', y = 'count')

# 날씨 데이터인 weather 데이터 시각화

sns.barplot(data = train, x = 'weather', y = 'count')
'''weather의 값이 작을수록 날씨가 좋은 것을 의미하는데 날씨가 좋지 않은 4 값의 대여량이
   3 값의 대여량 보다 많음. 따라서 weather가 4인 데이터 확인'''

train[train['weather'] == 4]
'''weather가 4인 데이터 확인 결과 값이 하나만 존재. 머신 러닝에 적용하기 어려움.'''
train.loc[train['weather'] == 4, 'weather'] = 3
'''따라서 weather가 4인 데이터를 weather이 3인 데이터로 변경'''

train[train['weather'] == 4]

sns.barplot(data = train, x = 'weather', y = 'count')

# test 데이터도 마찬가지로 처리

test[test['weather'] == 4]
'''test 데이터의 weather가 4인 값도 2개만 존재하므로 3으로 변경'''

test.loc[test['weather'] == 4, 'weather'] = 3
test[test['weather'] == 4]

# 온도에 해당하는 temp를 시각화(temp는 연속형 데이터이므로 barplot을 사용하기 어려움.따라서 seaborn의 distplot과 lmplot을 이용)

sns.lmplot(data = train, x = 'temp', y = 'count' , fit_reg = True) # 두 변수간의 선형관게를 파악하기 좋은 lmplot 사용
sns.distplot(train['temp']) # 정규분포도를 보기 좋은 distplot 사용

# 체감온도를 의미하는 atemp 데이터 시각화

sns.lmplot(data = train, x = 'atemp', y = 'count', fit_reg = True)
sns.distplot(train['atemp'])
''' temp 데이터와 atemp 데이터가 비슷한 형태를 보임. 따라서 어떤 관계가 있는지 알아보기 위해
    lmplot을 이용'''

sns.lmplot(data = train, x = 'temp' , y = 'atemp') # 선형선에 거의 일치
corr = train.corr()
corr['temp']
'''temp와의 상관관계를 확인한 결과 atemp와 0.984948이라는 높은 상관관계를 갖고 있음.
   다중공선성을 해결하기 위해 둘 중 목표 변수와 더 상관관계가 높은 데이터만 적용'''

corr['count'] # temp와 count의 상관계수는 0.394454, atemp와 count의 상관계수는 0.389784이므로 temp를 머신 러닝에 적용

# 습도와 풍속 그리고 목표 변수인 대여량의 데이터 분석

# 습도 데이터와 대여량 데이터의 관계

sns.lmplot(data = train, x = 'humidity', y = 'count', fit_reg = True)
'''뚜렷한 관계는 나타나지 않지만 선형선이 음의 방향으로 되어있으므로 습도가 높을수록 대여량이 적다고 해석할 수 있음.
   따라서 습도에 따라 자전거 대여소에 자전거 수를 조절하는 것이 효율적이라고 추측할 수 있음.'''
corr['count'] # count와의 상관분석 결과 습도와의 상관계수는 -0.317371로 약한 음의 관계가 있음.

# 풍속 데이터와 대여량 데이터의 관계

sns.lmplot(data = train, x = 'windspeed', y = 'count')
'''마찬가지로 뚜렷한 관계는 나타나지 않지만 선형선이 양의 방향으로 되어있음.
   따라서 풍속이 강할수록 대여량이 많을 것이라고 해석할 수도 있지만, 풍속이 40이상인 데이터의 수가 적어 명확하진 않음.'''

train['windspeed'].value_counts()
'''실제 데이터 수를 확인한 결과 0 값의 데이터 수가 가장 많음. 0 값은 바람이 불지 않는 날을 의미하는데
   일반적으로 바람이 불지 않는 날은 그렇게 많지 않아 데이터가 잘못 입력되었다는 가정 하에 재정의'''

# 풍속 데이터 처리

windspeed_0 = train[train['windspeed'] == 0]
print(windspeed_0.shape)
windspeed_0.head()

windspeed_not0 = train[train['windspeed'] != 0]
print(windspeed_not0.shape)
windspeed_not0.head()

# windspeed를 기준으로 훈련용 데이터셋, 테스트용 데이터셋 정리(x_train -> wsnot0, x_test -> ws0, y_train -> ws)

ws0 = windspeed_0[['season', 'workingday', 'weather', 'temp', 'humidity', 'Year', 'Month', 'Hour', 'Dayofweek']]
print(ws0.shape)
ws0.head()

wsnot0 = windspeed_not0[['season', 'workingday', 'weather', 'temp', 'humidity', 'Year', 'Month', 'Hour', 'Dayofweek']]
print(wsnot0.shape)
wsnot0.head()

ws = windspeed_not0['windspeed']
print(ws.shape)
ws.head()

# scikitlearn의 RandomForest 모델을 이용한 머신 러닝(단, 풍속 데이터가 분류형이 아닌 연속형 데이터이기 때문에 classifier(분류)모델이 아닌 regressor(회귀)모델을 사용)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model

model.fit(wsnot0, ws)

windspeed_0['windspeed'] = model.predict(ws0)
windspeed = pd.concat([windspeed_0, windspeed_not0], axis = 0)['windspeed'] # concat으로 windspeed_0 데이터와 windspeed_not0 데이터를 병합, axis = 0으로 설정하여 행을 기준으로 병합)
windspeed = train['windspeed']

# test도 train데이터와 마찬가지로 처리

test['windspeed'].value_counts()

windspeed_0_t = test[test['windspeed'] == 0]
windspeed_not0_t = test[test['windspeed'] != 0]
ws0_t = windspeed_0_t[['season', 'workingday', 'weather', 'temp', 'humidity', 'Year', 'Month', 'Hour', 'Dayofweek']]
wsnot0_t = windspeed_not0_t[['season', 'workingday', 'weather', 'temp', 'humidity', 'Year', 'Month', 'Hour', 'Dayofweek']]
ws_t = windspeed_not0_t['windspeed']
model.fit(wsnot0_t, ws_t)
windspeed_0_t['windspeed'] = model.predict(ws0_t)
windspeed_t = pd.concat([windspeed_0_t, windspeed_not0_t], axis = 0)['windspeed']
windspeed_t = test['windspeed']

# seaborn의 distplot을 이용한 count 데이터 시각화

sns.distplot(train['count'])
'''데이터가 왼쪽으로 쏠려있어 데이터를 분석하기에 좋지 않음. 로그값을 적용해 정규분포를 고르게 만들어 분석을 하고 예측값을 도출한 다음
   지수값을 적용해 복귀 시키는 방벙블 사용'''

import numpy as np

train['log_count'] = np.log(train['count'] + 1) # 0의 값에 로그를 적용하면 무한대의 값이 나오고 로그를 적용했을 때 0값이 나오지 않도록 하기위해 1을 더함
train[['count', 'log_count']].head()

sns.distplot(train['log_count']) # log_count 분포도를 시각화한 결과 데이터가 비교적 고르게 분포
