import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 模拟数据
data = [i + (i % 5) for i in range(100)]
df = pd.Series(data)

# 拟合 ARIMA 模型
model = ARIMA(df, order=(2, 1, 0))  # ARIMA(p=2, d=1, q=0)
model_fit = model.fit()

# 预测
forecast = model_fit.predict(start=90, end=105)
df.plot()
forecast.plot()
plt.show()
