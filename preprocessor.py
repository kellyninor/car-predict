from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
import pandas as pd

class CarDataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Cargar los datos de entrenamiento desde el archivo CSV
        self.dataTraining = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTrain_carListings.zip')
        self.minmax_scaler = MinMaxScaler()
    
    def fit(self, X, y=None):
        return self    
    
    
    def transform(self, X):
        X['Age'] = X.apply(self.calculate_age, axis=1)
        X['Manufacturer_Class'] = X.apply(self.calculate_manufacturer_class, axis=1)
        X['Model_Class'] = X.apply(self.calculate_model_class, axis=1)
        # Aplicar transformación Box-Cox y normalización MinMax a las columnas 'Age' y 'Mileage'
        X[['Age', 'Mileage']] = self.transform_minmax(X[['Age', 'Mileage']])    
        # Eliminar columnas no necesarias
        X.drop(['State', 'Make', 'Model', 'Year'], axis=1, inplace=True) 
        return self.create_input_data(X)
    
    def transform_minmax(self, X):         
        # Aplicar normalización MinMax
        X = pd.DataFrame(self.minmax_scaler.fit_transform(X), columns=X.columns)  
        # Eliminar columnas no necesarias  
        return X   
  
    def calculate_age(self, data):
        max_year = self.dataTraining['Year'].max()
        age = max_year - data['Year']
        if age == 0:
            age = 1
        return age
    
    def calculate_manufacturer_class(self, data):
        mean_price_manufacturer = self.dataTraining.groupby('Make')['Price'].mean().reset_index()
        class_1 = []
        class_2 = []
        class_3 = []
        limit_1 = mean_price_manufacturer['Price'].quantile(1/3)
        limit_2 = mean_price_manufacturer['Price'].quantile(2/3)
        for index, row in mean_price_manufacturer.iterrows():
            if row['Price'] <= limit_1:
                class_1.append(row['Make'])
            elif row['Price'] <= limit_2:
                class_2.append(row['Make'])
            else:
                class_3.append(row['Make'])
        class_1 = pd.Categorical(class_1)
        class_2 = pd.Categorical(class_2)
        class_3 = pd.Categorical(class_3)
        if data['Make'] in class_1:
            return '1'
        elif data['Make'] in class_2:
            return '2'
        else:
            return '3'
    
    def calculate_model_class(self, data):
        mean_price_by_model = self.dataTraining.groupby('Model')['Price'].mean().reset_index()
        tercile_1 = mean_price_by_model['Price'].quantile(1/3)
        tercile_2 = mean_price_by_model['Price'].quantile(2/3)
        class_1_models = mean_price_by_model[mean_price_by_model['Price'] <= tercile_1]['Model'].tolist()
        class_2_models = mean_price_by_model[(mean_price_by_model['Price'] > tercile_1) & (mean_price_by_model['Price'] <= tercile_2)]['Model'].tolist()
        class_3_models = mean_price_by_model[mean_price_by_model['Price'] > tercile_2]['Model'].tolist()
        class_1_models = pd.Categorical(class_1_models)
        class_2_models = pd.Categorical(class_2_models)
        class_3_models = pd.Categorical(class_3_models)
        if data['Model'] in class_1_models:
            return '1'
        elif data['Model'] in class_2_models:
            return '2'
        else:
            return '3'
    
    def create_input_data(self, data):
        record = {
            'Age': data['Age'],
            'Mileage': data['Mileage'],
            'Manufacturer_Class_1': 0,
            'Manufacturer_Class_2': 0,
            'Manufacturer_Class_3': 0,
            'Model_Class_1': 0,
            'Model_Class_2': 0,
            'Model_Class_3': 0
        }
        manufacturer_class = int(data['Manufacturer_Class'])
        record[f'Manufacturer_Class_{manufacturer_class}'] = 1
        model_class = int(data['Model_Class'])
        record[f'Model_Class_{model_class}'] = 1
        record_df = pd.DataFrame([record])
        return record_df