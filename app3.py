import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("startups.csv", encoding='latin-1')

# Data preprocessing
df['Valuation'] = df['Valuation ($B)'].str.replace('$', '', regex=False).astype(float)
df['Year Joined'] = pd.to_datetime(df['Date Joined'], errors='coerce').dt.year
df['Num_Investors'] = df['Select Investors'].fillna('Unknown').apply(lambda x: len(x.split(',')))
df.dropna(subset=['Valuation', 'Year Joined', 'Industry', 'Country', 'City'], inplace=True)

# Log transform the valuation for modeling
df['Log_Valuation'] = np.log1p(df['Valuation'])

# Features and target
X = df[['Year Joined', 'Num_Investors', 'Industry', 'Country', 'City']]
y = df['Log_Valuation']

categorical_cols = ['Industry', 'Country', 'City']
numerical_cols = ['Year Joined', 'Num_Investors']

# Preprocessor
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

# Model pipeline
model = RandomForestRegressor(n_estimators=100, random_state=42)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
pipeline.fit(X_train, y_train)

# Streamlit UI
st.title("🦄 Unicorn Startup Valuation Predictor (Enhanced)")

st.markdown("### 🎯 Enter Startup Details Below")

year = st.slider("Year Joined", int(df['Year Joined'].min()), int(df['Year Joined'].max()), 2020)
num_investors = st.slider("Number of Investors", 1, 50, 5)
industry = st.selectbox("Industry", sorted(df['Industry'].unique()))
country = st.selectbox("Country", sorted(df['Country'].unique()))
city = st.selectbox("City", sorted(df['City'].unique()))

# Predict
input_df = pd.DataFrame([{
    'Year Joined': year,
    'Num_Investors': num_investors,
    'Industry': industry,
    'Country': country,
    'City': city
}])

log_pred = pipeline.predict(input_df)[0]
actual_pred = np.expm1(log_pred)

st.success(f"💰 Predicted Valuation: **${actual_pred:.2f} Billion**")

# Download button
st.download_button("📥 Download Prediction", data=input_df.assign(Predicted_Valuation=actual_pred).to_csv(index=False), file_name="unicorn_prediction.csv")

# Feature Importance Plot
st.markdown("### 🔍 Feature Importance")

# Fit again to access feature names
preprocessor.fit(X_train)
feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_feature_names = np.concatenate([feature_names, numerical_cols])

importances = model.feature_importances_

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
feat_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
feat_df.sort_values(by='Importance', ascending=True).tail(20).plot.barh(x='Feature', y='Importance', ax=ax, color='teal')
plt.title("Top Feature Importances")
st.pyplot(fig)

# Evaluation metric
st.markdown("### 📉 Model RMSE on Test Data")
y_test_pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_test_pred)))
st.code(f"RMSE: ${rmse:.2f} Billion")
