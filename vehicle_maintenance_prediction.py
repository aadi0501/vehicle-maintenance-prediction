import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, request, render_template_string

# Initialize Flask app
app = Flask(__name__)

# Load the dataset and automatically calculate Temperature Difference
data = pd.read_csv(r"https://raw.githubusercontent.com/aadi0501/vehicle-maintenance-prediction/refs/heads/main/engine_data.csv")

# Automatically calculate Temperature Difference
data['Temperature_difference'] = data['lub oil temp'] - data['Coolant temp']

# Prepare features and labels
X = data.drop(columns=['Engine Condition'])
y = data['Engine Condition']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")

# HTML Template for the web page
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Engine Condition Prediction</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            background: #f7f7f7;
            color: #333;
            margin: 0;
            padding: 0;
        }
        h1, h2 {
            text-align: center;
            color: #333;
        }
        .container {
            width: 100%;
            max-width: 800px;
            margin: 50px auto;
            background-color: #fff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        form {
            display: flex;
            flex-direction: column;
        }
        label {
            margin: 10px 0 5px;
            font-size: 1rem;
            color: #555;
        }
        input {
            padding: 10px;
            font-size: 1rem;
            margin-bottom: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            outline: none;
            transition: border 0.3s;
        }
        input:focus {
            border-color: #007bff;
        }
        button {
            padding: 12px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 1.2rem;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #0056b3;
        }
        .result {
            margin-top: 20px;
            text-align: center;
            font-size: 1.5rem;
            font-weight: 500;
        }
        .result.success {
            color: #28a745;
        }
        .result.warning {
            color: #dc3545;
        }
        .form-group {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }
        .form-group input {
            width: 100%;
        }
        .header {
            background-color: #007bff;
            color: white;
            padding: 15px;
            text-align: center;
            border-radius: 8px 8px 0 0;
        }
        .footer {
            margin-top: 20px;
            text-align: center;
            font-size: 0.9rem;
            color: #777;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-cogs"></i> Engine Condition Prediction</h1>
        </div>

        <h2>Enter Engine Parameters</h2>
        <form method="POST" action="/predict">
            <div class="form-group">
                <label for="engine_rpm">Engine RPM</label>
                <input type="number" name="engine_rpm" step="0.01" placeholder="Enter engine RPM" required>
            </div>
            <div class="form-group">
                <label for="lub_oil_pressure">Lub Oil Pressure</label>
                <input type="number" name="lub_oil_pressure" step="0.01" placeholder="Enter Lub Oil Pressure" required>
            </div>
            <div class="form-group">
                <label for="fuel_pressure">Fuel Pressure</label>
                <input type="number" name="fuel_pressure" step="0.01" placeholder="Enter Fuel Pressure" required>
            </div>
            <div class="form-group">
                <label for="coolant_pressure">Coolant Pressure</label>
                <input type="number" name="coolant_pressure" step="0.01" placeholder="Enter Coolant Pressure" required>
            </div>
            <div class="form-group">
                <label for="lub_oil_temp">Lub Oil Temperature</label>
                <input type="number" name="lub_oil_temp" step="0.01" placeholder="Enter Lub Oil Temperature" required>
            </div>
            <div class="form-group">
                <label for="coolant_temp">Coolant Temperature</label>
                <input type="number" name="coolant_temp" step="0.01" placeholder="Enter Coolant Temperature" required>
            </div>
            <button type="submit"><i class="fas fa-check-circle"></i> Predict</button>
        </form>

        {% if result %}
        <div class="result {% if 'Normal' in result %}success{% else %}warning{% endif %}">
            <h3>{{ result }}</h3>
        </div>
        {% endif %}
    </div>

    <div class="footer">
        <p>&copy; 2024 Engine Condition Predictor | Designed with <i class="fas fa-heart"></i> by You</p>
    </div>
</body>
</html>
"""

# Flask route for home page
@app.route('/')
def home():
    return render_template_string(html_template)

# Flask route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        engine_rpm = float(request.form['engine_rpm'])
        lub_oil_pressure = float(request.form['lub_oil_pressure'])
        fuel_pressure = float(request.form['fuel_pressure'])
        coolant_pressure = float(request.form['coolant_pressure'])
        lub_oil_temp = float(request.form['lub_oil_temp'])
        coolant_temp = float(request.form['coolant_temp'])

        # Calculate Temperature Difference
        temp_difference = lub_oil_temp - coolant_temp

        # Prepare input for the model
        input_data = np.array([engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp, temp_difference]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Display the result
        result = "Normal" if prediction == 1 else "Check Engine"
        return render_template_string(html_template, result=f"The engine condition is: {result}")
    except Exception as e:
        return render_template_string(html_template, result=f"Error: {str(e)}")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
