import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import model_helper

COLS = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
def rawToFrame(raw):
    frame = pd.DataFrame([raw],columns=COLS)
    return frame


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        raw_input = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['diabetes_pedigree']),
            float(request.form['age'])
        ]
        
        pr = model_helper.predict(rawToFrame(raw_input))
        if pr[0][1] > 0.5:
            prediction = f"Diabetic({pr[0][1]*100}%)"
        else:
            prediction = f"Non-Diabetic({pr[0][0]*100}%)"

        #lIME
        liEx = model_helper.limeExplain(np.array(raw_input))
        exp_list = liEx.as_list()
        lime_labels, lime_data = zip(*exp_list)
        #SHAP
        shap_values = model_helper.get_shap_values(rawToFrame(raw_input))

        #Making labels in same order 
        labels = []
        shapValsOrdered = []

        for raw_label in lime_labels:
            splitted = raw_label.split(' ')
            for i in splitted:
                if i in COLS:
                    labels.append(i)
                    shapValsOrdered.append(shap_values[COLS.index(i)])
                    continue

        return render_template('index.html', prediction=prediction, lime_data=lime_data, lime_labels=labels, lime_explanation=True, \
                               shap_data = shapValsOrdered, shap_labels = labels, shap_explanation=True  )
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
