#  UNICORN STARTUP VALUATION PREDICTION MODEL

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("startups.csv", encoding='latin-1')

df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

df.rename(columns={
    'Valuation($B)': 'Valuation',
    'Select Investors': 'Investors',
    'Date Joined': 'Date'
}, inplace=True)

df['Valuation'] = df['Valuation'].str.replace('$', '').str.replace('B', '').astype(float)

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Year Joined'] = df['Date'].dt.year

df['Num_Investors'] = df['Investors'].fillna('').apply(lambda x: len(x.split(',')))

df.dropna(subset=['Valuation', 'Year Joined'], inplace=True)

sns.scatterplot(x='Year Joined', y='Valuation', data=df)
plt.title("Valuation vs Year Joined")
plt.xlabel("Year Joined")
plt.ylabel("Valuation ($B)")
plt.show()

X = df[['Year Joined', 'Num_Investors']]
y = df['Valuation']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

sample_input = pd.DataFrame([[2021, 4]], columns=['Year Joined', 'Num_Investors'])
prediction = model.predict(sample_input)
print(f"\n🔮 Predicted Unicorn Valuation: ${round(prediction[0], 2)} Billion")
