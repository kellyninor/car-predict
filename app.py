from flask import Flask
from flask_restx import Api, Resource, fields
from joblib import load
import pandas as pd
import importlib

app = Flask(__name__)
api = Api(app, version='1.0', title='Car Prediction API', description='API for predicting car prices- Team 32')

# Cargar el preprocesador y el modelo entrenado
# Importa dinámicamente el preprocesador desde preprocessor.py
preprocessor_module = importlib.import_module('preprocessor')
CarDataPreprocessor = preprocessor_module.CarDataPreprocessor

# Cargar el preprocesador y el modelo entrenado
preprocessor = CarDataPreprocessor()
modelo = load('best_model.pkl')

# Definir la estructura del input
prediction_input = api.model('Data de Entrada', {
    'Year': fields.Integer(required=True),
    'Mileage': fields.Integer(required=True),
    'State': fields.String(required=True),
    'Make': fields.String(required=True),
    'Model': fields.String(required=True)
})

class PredictAPI(Resource):
    @api.expect(prediction_input)
    def post(self):
        # Obtener los datos de entrada
        data = api.payload

        # Convertir los datos de entrada en un DataFrame de Pandas
        input_data = pd.DataFrame([data])

        # Preprocesar los datos de entrada
        processed_input_data = preprocessor.transform(input_data)

        print(processed_input_data)

        # Realizar la predicción
        prediction = modelo.predict(processed_input_data)

        # Devolver la predicción como JSON
        return {'prediction': prediction.tolist()}

api.add_resource(PredictAPI, '/predict')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)