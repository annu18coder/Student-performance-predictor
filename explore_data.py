import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("data/student_data.csv")

# 1. hmne check kia ki data load ho rha h ya nhi
print(df.head())

# 2. hmne dekha ki dataset kitna bara h - rows and columns k no btata h
print(df.shape)         # output: (100, 6)  -> 100 rows and 6 columns

# 3 hme dekhna hai ki Har column ka data type kya hai, aur data missing toh nahi hai?
print(df.info())
# isko hm simply aise bhi likh skte h
# df.info()

"""
describe() - 
        "Hamare marks aur attendance ke numbers generally kaise distributed hain?"

For example:
- Average study hours kitne hain?
- Average final marks?
- Lowest marks?
- Highest marks?
- Data kitna spread hai?

        sirf numerical columns ke statistical summary deta hai."""
print(df.describe())

# 4. overall null value deta h 0(false) aur 1(true haa hai null value uske basis pr)
print(df.isnull().sum())

# 5. same overall duplicated value deta h 0 and 1 k form m
print(df.duplicated().sum())


# 6. code for showing the graph distributionn
plt.hist(df["study_hours"])
plt.title("Study Hours Distribution")
plt.xlabel("Study Hours")
plt.ylabel("Number of Students")

# plt.show()

# 7. Ab hum study hours aur final marks ke beech relationship dekhenge.
'''
Ye ML ke liye bahut important hai because humara question hai:
"Kya study hours badhne par final marks bhi generally badhte hain?"

Iske liye histogram nahi, scatter plot use karenge. 📈
'''

plt.scatter(df["study_hours"], df["final_marks"])
# plt.show()

plt.scatter(df["attendance"], df["final_marks"])
# plt.show()

plt.scatter(df["previous_marks"], df["final_marks"])
# plt.show()

plt.scatter(df["assignment_score"], df["final_marks"])
# plt.show()

plt.scatter(df["test_score"], df["final_marks"])
# plt.show()

# Inference
'''
Ye perfect upward line nahi hai.
Kyun?
10 hours wale students mein bhi marks different hain, aur 1 hour wale students mein bhi variation hai.
So hum bolenge:
Study hours and final marks have a positive relationship, but it's not perfectly linear.
Isliye study hours vs final marks ka graph upward hai, but dots perfectly straight line mein nahi hain
'''

# 8. Finding correlation between variables
correlation = df.corr()
# print(correlation)

# 9. Heatmap
sns.heatmap(correlation, annot=True, cmap="coolwarm")
# plt.show()

# Dataset exploration and understanding completed ✔️

# 10. Model bnane k lie ab hme model ko batana hai:
''' Tumhe kaunse inputs dekhkar output predict karna hai? '''

# phase 5: Features vs Target 
X = df[["study_hours", "attendance", "previous_marks", "assignment_score", "test_score"]]
Y = df["final_marks"]
print(X.head())
print(Y.head())

# phase 6: Train/Test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# phase 7: Linear Regression
model = LinearRegression()              # Linear Regression ka ek model/object bna h
model.fit(X_train, Y_train)
print(model.coef_)              # [2.   0.15 0.25 0.15 0.25]

# finding intercept 
print(model.intercept_)         #  -2.842170943040401e-14

# Prediction
model.predict(X_test)
predictions = model.predict(X_test)

print(Y_test)
print(predictions)

# systematically checking the accuracy of predicted values
mae = mean_absolute_error(Y_test, predictions)
print(mae)

mse = mean_squared_error(Y_test, predictions)
print(mae)

rmse = mean_squared_error(Y_test, predictions)** 0.5
print(rmse)

r2 = r2_score(Y_test, predictions)
print(r2)

# Prediction through second model - Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=1)
rf_model.fit(X_train, Y_train)                  # output is output ka R2
rf_prediction = rf_model.predict(X_test)

# 11 Saving the model
joblib.dump(rf_model, "random_forest_model.pkl")

#  Loading the model
loaded_model = joblib.load("random_forest_model.pkl")

print("X_test:")
print(X_test)
print("Actual: ")
print(Y_test.values)
print("Predictions: ")
print(rf_prediction)

# Calculating the errors of Random Forest predictions
rf_mae = mean_absolute_error(Y_test, rf_prediction)
print(rf_mae)           # 3.1672250000000037

rf_mse = mean_squared_error(Y_test, rf_prediction)
print(rf_mse)           # 23.862547812499958

# to check ki kis value k karan zyada error aa rha h or which is affecting more
errors = Y_test - rf_prediction
squared_errors = errors ** 2
print(squared_errors)

rf_rmse = mean_squared_error(Y_test, rf_prediction) ** 0.5
print(rf_rmse)                 # 3.794616222360299

print("MSE:", rf_mse)
print("RMSE:", rf_rmse)
print("RMSE²:", rf_rmse ** 2)

# R² = R-squared
rf_r2 = r2_score(Y_test, rf_prediction)
print(rf_r2)

# Model Comparison
print("Linear Regression")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)

print("\nRandom Forest")
print("MAE:", rf_mae)
print("MSE:", rf_mse)
print("RMSE:", rf_rmse)
print("R²:", rf_r2)


# Testing on a new input
# new_student = [[7, 85, 72, 80, 75]]
# use of [[]] - Inner [ ] → ek student ke 5 features
# - Outer [ ] → students ka collection / rows
# Isliye shape:
# (1, 5)
new_student = pd.DataFrame([{
    "study_hours": 7,
    "attendance": 85,
    "previous_marks": 72,
    "assignment_score": 80,
    "test_score": 75
}])

prediction = model.predict(new_student)
print("Linear Regression Based Predicted Performance:", prediction)

rf_prediction = rf_model.predict(new_student)
print("Random Forest Based Predicted Performance: ", rf_prediction)

load_prediction = loaded_model.predict(new_student)
print("Loaded Model Prediction: ", load_prediction)
