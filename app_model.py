from flask import Flask, jsonify, request
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from src.utils.utils import *

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return '''
    
    ML Boat – Titanic Survival Prediction API
    ¿Jack cabía en la tabla?

    '''

@app.route('/api/v1/predict', methods=['GET'])
def predict():

    age = request.args.get('age', None)
    sex = request.args.get('sex', None)
    pclass = request.args.get('pclass', None)
    fare = request.args.get('fare', None)

    model = joblib.load('src/model/modelo_pipeline.joblib')
    X = predict_survival(model, int(age), str(sex.strip()), int(pclass), int(fare), column_order=None)
    prediction = model.predict(X)[0]
    print(prediction)

    if prediction == 0:
        return 'MUERE'
    else:
        return 'VIVES'

@app.route('/api/v1/retrain', methods=['GET'])
def retrain(): # Ligado al endpoint '/api/v1/retrain/', metodo GET
    if os.path.exists('src/data_retrain/titanic_new.csv'):
        data = pd.read_csv('src/data_retrain/titanic_new.csv')

        set_train, set_test = train_test_split(data,test_size = 0.20,random_state=42)

        model = joblib.load('src/model/modelo_pipeline.joblib')
        model.fit(set_train, set_train['Survived'])

        y_pred = model.predict(set_test.drop(columns=['Survived']))
        
        report = classification_report(set_test['Survived'], y_pred)

        # Reentrenamos con todos los datos
        model.fit(data, data['Survived'])
        joblib.dump(model, 'src/model/modelo_retrain.joblib')
            
        return f"Model retrained. Check new classification report", report 
        
    else:
        return f"<h2>New data for retrain NOT FOUND. Nothing done!</h2>"

if __name__ == '__main__':
    app.run(debug=True)