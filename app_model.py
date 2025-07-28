from flask import Flask, jsonify, request
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from xgboost import XGBClassifier

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return 'ML Boat – Titanic Survival Prediction API'

@app.route('/api/v1/predict', methods=['GET'])
def predict():

    age = request.args.get('age', None)
    sex = request.args.get('sex', None)
    pclass = request.args.get('pclass', None)
    fare = request.args.get('fare', None)

    # función Deivid

    variables = [age, sex, pclass, fare]

    return f'Prediccion {variables}'

@app.route('/api/v1/retrain', methods=['GET'])
def retrain(): # Ligado al endpoint '/api/v1/retrain/', metodo GET
    if os.path.exists('src/data_retrain/titanic_new.csv'):
        data = pd.read_csv('src/data_retrain/titanic_new.csv')

        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['Survived']),
                                                        data['Survived'],
                                                        test_size = 0.20,
                                                        random_state=42)

        model = joblib.load('../model/modelo_pipeline.joblib')
        model.fit(X_train, y_train)
 
        model.fit(data.drop(columns=['Survived']), data['Survived'])
        y_pred = model.predict(data)   

        joblib.dump(model, 'modelo_pipeline.joblib')
            
        return f"Model retrained. Check new classification report and Confusion Matrix Display:", 
        classification_report(data["Survived"], model.predict(data)), ConfusionMatrixDisplay.from_predictions(data["Survived"], y_pred) 
        
    else:
        return f"<h2>New data for retrain NOT FOUND. Nothing done!</h2>"

if __name__ == '__main__':
    app.run(debug=True)