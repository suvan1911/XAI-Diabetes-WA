import pickle
import pandas as pd

with open('model.pickle','rb') as f:
    model = pickle.load(f)
    df = pickle.load(f)
    X_train = pickle.load(f)

from lime.lime_tabular import LimeTabularExplainer
limeExplainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=[0, 1], mode='classification')
def limeExplain(inp):
    exp = limeExplainer.explain_instance(inp, model.predict_proba)
    return exp

import dice_ml
dice_data = dice_ml.Data(dataframe=df,
                        continuous_features=['DiabetesPedigreeFunction', 'BMI','Insulin','SkinThickness','BloodPressure','Glucose','Age','Pregnancies'],
                        outcome_name = 'Outcome'
                        )

dice_model = dice_ml.Model(model=model, backend='sklearn')
exp = dice_ml.Dice(dice_data, dice_model)

features_to_vary = ['BMI','Glucose','BloodPressure']
permitted_range =   {'BMI':[18,35],
                     'Glucose': [70,250],
                     'BloodPressure':[40,120],
                     }

def genCounterfactual(inp):
    dice_exp = exp.generate_counterfactuals(inp,
                                            total_CFs=1, 
                                            desired_class="opposite", 
                                            features_to_vary=features_to_vary,
                                            permitted_range=permitted_range)
    
    return dice_exp


import shap 
explainer = shap.Explainer(model)

def get_shap_values(frame):
    shap_values = explainer.shap_values(frame)
    return shap_values[:,:,1][0]


def predict(frame):
    prediction = model.predict_proba(frame)
    return prediction





