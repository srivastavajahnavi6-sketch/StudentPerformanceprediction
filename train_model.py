
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("StudentPerformanceFactors.csv")

# Only 3 input features
X = data[["Hours_Studied", "Attendance", "Sleep_Hours"]]
y = data["Exam_Score"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "student_model.pkl")
print("Model Saved Successfully!")

# Predictions
predictions = model.predict(X_test)

print("Model Trained Successfully!")
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))






