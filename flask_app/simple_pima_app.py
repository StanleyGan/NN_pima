from sklearn.externals import joblib
import numpy as np
from flask import Flask, request

model = None
app = Flask(__name__)

def load_model():
    global model
    model = joblib.load('model.pkl')

@app.route('/')
def home_endpoint():
    return "Hello World"

@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for single sample
    if request.method == "POST":
        data = request.get_json()
        data = np.array(data)[np.newaxis,:]
        scaled_data = model.scaler_x.transform(data)
        prediction = model.predict(scaled_data)

    return str(prediction[0])

if __name__ == "__main__":
    load_model()
    app.run(host='0.0.0.0', port=5000)
