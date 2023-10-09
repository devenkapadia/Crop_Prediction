from flask import Flask, jsonify, request
import pickle
import pandas as pd
import numpy as np
from flask_cors import CORS

pipe = pickle.load(open('pipe.pkl', 'rb'))
app = Flask((__name__))
CORS(app, origins="http://localhost:3000")


@app.route('/')
def home():
    val = [12,23]
    return jsonify({'prediction': val})


@app.route('/predict', methods=['POST'])
def predict():
    # data = [40, 40, 40, 45, 70, 8, 100]
    data = request.json
    test_input = np.array([data])
    val = pipe.predict(test_input)[0]
    print(val)
    return jsonify({'prediction': val})


if __name__ == '__main__':
    app.run(debug=True)
