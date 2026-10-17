import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load student dataset
data = pd.read_csv("student_data.csv")

print("STUDENT PERFORMANCE PREDICTION")
print("--------------------------------")
print(data.head())

# Select input and output
X = data[["study_hours", "previous_marks"]]
y = data["final_marks"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Machine Learning model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)

print("\nModel trained successfully!")
print("Mean Absolute Error:", round(error, 2))

# Predict for a new student
hours = float(input("\nEnter study hours per day: "))
marks = float(input("Enter previous marks out of 100: "))

student = pd.DataFrame({
    "study_hours": [hours],
    "previous_marks": [marks]
})

predicted_marks = model.predict(student)[0]
predicted_marks = max(0, min(100, predicted_marks))

print("\n----- PREDICTION RESULT -----")
print("Study Hours:", hours)
print("Previous Marks:", marks)
print("Predicted Final Marks:", round(predicted_marks, 2))
report = pd.DataFrame({
    "Study Hours": [hours],
    "Previous Marks": [marks],
    "Predicted Final Marks": [round(predicted_marks, 2)]
})

report.to_csv("prediction_report.csv", index=False)

print("Report saved as prediction_report.csv")
print("-----------------------------")
import matplotlib.pyplot as plt

plt.scatter(
    data["study_hours"],
    data["final_marks"],
    label="Actual Marks"
)

plt.scatter(
    [hours],
    [predicted_marks],
    color="red",
    label="Predicted Marks"
)

plt.xlabel("Study Hours Per Day")
plt.ylabel("Final Exam Marks")
plt.title("Student Performance Prediction")
plt.legend()
plt.grid(True)
plt.show()
