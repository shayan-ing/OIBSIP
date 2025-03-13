from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load('car_price_predictor.pkl')

# Function to preprocess user input
def preprocess_input(year, present_price, driven_kms, fuel_type, selling_type, transmission, owner):
    # Calculate Age (assuming the model was trained with 2025 as the current year)
    age = 2025 - year

    # Create a DataFrame from user input
    user_input = pd.DataFrame({
        'Year': [year],
        'Present_Price(In Lakhs)': [present_price],
        'Driven_kms': [driven_kms],
        'Fuel_Type': [fuel_type],
        'Selling_type': [selling_type],
        'Transmission': [transmission],
        'Owner': [owner],
        'Age': [age]
        
    })

    # Log the input data for debugging
    print("Processed Input Data:")
    print(user_input)

    return user_input

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        # Get form data
        year = int(request.form['year'])
        present_price = float(request.form['present_price'])
        driven_kms = int(request.form['driven_kms'])
        fuel_type = request.form['fuel_type']
        selling_type = request.form['selling_type']
        transmission = request.form['transmission']
        owner = int(request.form['owner'])

        # Preprocess the input
        user_input = preprocess_input(year, present_price, driven_kms, fuel_type, selling_type, transmission, owner)

        # Make a prediction
        prediction = model.predict(user_input)[0]
        print(f"Predicted Selling Price: {prediction}")

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)