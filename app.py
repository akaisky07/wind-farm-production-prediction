from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the machine learning model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        u10 = float(request.form['u10'])
        v10 = float(request.form['v10'])
        u100 = float(request.form['u100'])
        v100 = float(request.form['v100'])

        # Perform the prediction using the model
        production = model.predict([[u10, v10, u100, v100]])[0]

        return jsonify({'production': production})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

