import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# initializing the flask app
app=Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

# default page of our web app
@app.route('/')
def home():
    return render_template('index.html')

# For predicting the output
@app.route('/predict',methods=['POST'])
def predict():
    # For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 3)

    return render_template('index.html', prediction_text='Salary prediction based on experience is :{}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)