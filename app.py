from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load data and train the model
data = pd.read_csv("heart_disease.csv")
features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
target = "target"

# Split data
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Define a function to make predictions
def predict_heart_disease(input_data):
    input_df = pd.DataFrame([input_data], columns=features)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    return "Heart Disease" if prediction == 1 else "No Heart Disease"

# Web route for the homepage
@app.route('/')
def index():
    return render_template('index.html', features=features)

# Web route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    input_data = {feature: float(request.form[feature]) for feature in features}
    result = predict_heart_disease(input_data)
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
