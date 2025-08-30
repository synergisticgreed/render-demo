from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        cgpa = float(request.form['CGPA'])
        iq = float(request.form['IQ'])
        attendance = float(request.form['Attendance'])

        # Add dummy value for serial number (first feature)
        serial_number = 0  # Can be any number; model doesn't really use it
        final_features = np.array([[serial_number, cgpa, iq, attendance]])

        # Make prediction
        prediction = model.predict(final_features)
        output = 'Placed' if prediction[0] == 1 else 'Not Placed'

        return render_template('index.html', prediction_text='Prediction: {}'.format(output))
    
    except Exception as e:
        return render_template('index.html', prediction_text='Error: {}'.format(e))

if __name__ == "__main__":
    app.run(debug=True)
