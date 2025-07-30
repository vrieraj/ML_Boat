## 🚤 ML Boat – Titanic Survival Prediction API

**MLBoat** es un proyecto de aprendizaje automático que predice la probabilidad de supervivencia de un pasajero del Titanic a partir de tres variables clave: **sexo**, **edad** y **tarifa pagada**. Utiliza un modelo de clasificación basado en **XGBoost** y expone sus predicciones mediante una API REST construida con **FastAPI**.

Este proyecto está diseñado como un ejercicio práctico para aprender a entrenar modelos de ML, evaluar su rendimiento y desplegarlos como servicios web.

---


## 🧠 Variables utilizadas

- `sex` (str): "male" o "female"
- `age` (float): Edad del pasajero
- `fare` (float): Tarifa pagada por el billete

---

## 📡 Ejemplo de solicitud

- **Endpoint** `/api/v1/predict`

``` bash

http://localhost:8000/api/v1/predict?age=29&sex=female&pclass=2&fare=35.5
```

---


## 🔁 Retrain del modelo con nuevos datos

El sistema permite reentrenar el modelo con nuevos datos de entrenamiento proporcionados por el usuario.

**Requisitos:**

- El archivo debe llamarse exactamente `titanic_new.csv`.

- Debe estar ubicado en la ruta: `src/data_retrain/titanic_new.csv`.

- El archivo debe contener las siguientes columnas:

``` bash
PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
```

Si el archivo `titanic_new.csv` existe y es válido, el modelo se reentrena automáticamente y se guarda el nuevo modelo en src/model/modelo_pipeline.joblib.
