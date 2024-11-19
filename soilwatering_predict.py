import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False


"""
    선형회귀 모델의 변경
    Pn = sigma(n, i<=WLS) Di - D(i-1)/n-1
    WLS는 물이 부족하다는 신호를 보낸 경우이다
"""

# Step 1: 더미 토양 수분 데이터 생성
np.random.seed(0)
days = np.arange(1, 31)
soil_moisture = np.random.uniform(low=0, high=100, size=30)  # Random soil moisture values

# Step 2: 데이터 전처리
# 수분이 30이하로 떨어지면 수분공급이 필요하다고 간주
water_need_days = days[soil_moisture < 30]

# Step 3: 예측 모델
# 수분이 낮은 날을 수분공급이 요구되는 날로 간주
X = water_need_days.reshape(-1, 1)
y = np.roll(water_need_days, -1)[:-1] - water_need_days[:-1]  # 수분 공급한 날 사이의 기간

model = LinearRegression()
model.fit(X[:-1], y)  # 가장 최신 데이터 추출

# 수분 공급 주기 예측
predicted_cycle = model.predict(X)

# Step 4: 출력
print("수분 공급 주기 예측:", predicted_cycle)

# Plot
plt.scatter(days, soil_moisture, label='토양 습도')
plt.plot(X, model.predict(X), color='red', label='예측된 수분 공급 주기')
plt.axhline(y=30, color='green', linestyle='--', label='수분 공급 한계점')
plt.xlabel('일')
plt.ylabel('토양 습도')
plt.title('토양 습도 수준에 따른 수분 공급 주기 예측')
plt.legend()
plt.show()
