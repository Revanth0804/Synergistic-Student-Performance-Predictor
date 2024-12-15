import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('student_performance_model', 'rb'))

def replace_values(lst, old_value, new_value):
    for i in range(len(lst)):
        if lst[i] == old_value:
            lst[i] = new_value

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    data = []
    for x in request.form.values():
        data.append(x)

    replace_values(data, 'Male', 0)
    replace_values(data, 'Female', 1)
    replace_values(data, 'GP', 0)
    replace_values(data, 'MS', 1)
    replace_values(data, 'Urban', 0)
    replace_values(data, 'rural', 1)
    replace_values(data, 'alive', 0)
    replace_values(data, 'dead', 1)
    replace_values(data, 'Home maker', 0)
    replace_values(data, 'health', 1)
    replace_values(data, 'other', 2)
    replace_values(data, 'services', 3)
    replace_values(data, 'teacher', 4)
    replace_values(data, 'course', 0)
    replace_values(data, 'home', 1)
    replace_values(data, 'other', 2)
    replace_values(data, 'reputation', 3)
    replace_values(data, 'mother', 0)
    replace_values(data, 'father', 1)
    replace_values(data, 'other', 2)
    replace_values(data, 'yes', 0)
    replace_values(data, 'no', 1)
    replace_values(data, 'none', 0)
    replace_values(data, 'primary_education', 1)
    replace_values(data, 'secondary_education', 2)
    replace_values(data, 'higher_education', 3)

    data.append(0)
    data.append(1)
    data.append(0)
    data.append(1)
    # print('The data is:',data)
    

    int_features = [int(x) for x in data]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    # prediction=prediction.upper()
    # output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='The performance of the student is {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)