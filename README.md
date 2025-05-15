from google.colab import files
uploaded = files.upload()
# Step 1: Load data
df = pd.read_csv('house_prices_large.csv')

# Step 2: Select features and target
X = df[['Area', 'Bedrooms', 'Bathrooms', 'Parking', 'Age']]
y = df['Price']

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 4: Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Predict and evaluate
predictions = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R² Score:", r2_score(y_test, predictions))

# Step 6: Plot results
plt.scatter(y_test, predictions, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.show()
