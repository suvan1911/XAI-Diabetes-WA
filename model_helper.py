import pickle
import pandas as pd 

COLS = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
def rawToFrame(raw):
    frame = pd.DataFrame([raw],columns=COLS)
    return frame

with open('model2.pickle','rb') as f:
    model = pickle.load(f)
    df = pickle.load(f)
    X_train = pickle.load(f)

from lime.lime_tabular import LimeTabularExplainer
limeExplainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=[0, 1], mode='classification')
def limeExplain(inp):
    exp = limeExplainer.explain_instance(inp, model.predict_proba)
    return exp

import shap 
explainer = shap.Explainer(model)

def get_shap_values(frame):
    shap_values = explainer.shap_values(frame)
    return shap_values[:,:,1][0]


def predict(frame):
    prediction = model.predict_proba(frame)
    return prediction





