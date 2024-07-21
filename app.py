from flask import Flask, render_template, request, jsonify
import numpy as np
from neural_network import initialize_parameters, train, predict

app = Flask(__name__)

# Sample dataset (XOR logic gate)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

input_dim = 2
hidden_dim = 4
output_dim = 1
iterations = 10000
learning_rate = 0.1

W1, b1, W2, b2 = train(X, Y, input_dim, hidden_dim, output_dim, iterations, learning_rate)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    data = request.json
    input_data = np.array(data['input']).reshape(1, -1)
    prediction = predict(input_data, W1, b1, W2, b2)
    return jsonify({'prediction': int(prediction[0, 0])})

if __name__ == '__main__':
    app.run(debug=True)
