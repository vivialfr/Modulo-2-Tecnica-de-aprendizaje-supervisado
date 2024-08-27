'''
Momento de Retroalimentación: 
Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework. 

Viviana Alanis Fraige | A01236316
8/26/2024
'''

# Librerias a utilizar
import pandas as pd
import numpy as np

'''IMPORTAR DATOS A UTILIZAR: se utilzara el dataset de cancer para predecir 
el nivel de cancer de un paciente'''
# Cargar el conjunto de datos
df = pd.read_csv('Cancer_Data.csv')
df.head()


'''PREPROCESAMIENTO DE DATOS'''
# Eliminar la columna 'id' ya que no me es util para este analisis
df = df.drop(columns=['id'])

# Convertir la columna 'diagnosis' la cual es mi target a valores binarios: 'M' = 1, 'B' = 0
# donde M es maligno y B es benigno
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Tratar valores NaN (rellenar con la media de la columna)
df = df.fillna(df.mean())

# Seleccionar caracteristicas 
features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
             'smoothness_mean', 'compactness_mean', 'concavity_mean', 
             'concave points_mean', 'radius_worst', 'texture_worst', 
             'perimeter_worst', 'area_worst', 'smoothness_worst', 
             'compactness_worst', 'concavity_worst', 'concave points_worst', 
             'symmetry_worst']

X = df[features].values  # caracteristicas
y = df['diagnosis'].values  # la columna 'diagnosis' es el target

# Normalizar los datos
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

'''ENTRENAMIENTO Y PRUEBA'''
# Dividir en conjunto de entrenamiento y prueba
def train_test_split(X, y, test_size=0.2, random_state=5):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_index = int(X.shape[0] * (1 - test_size))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

X_train, X_test, y_train, y_test = train_test_split(X, y)


'''IMPLEMENTANDO REGRESION LINEAL'''
# FunciOn para calcular los coeficientes regresiOn lineal
def coefficients(X, y, learning_rate=0.001, epochs=1000):
    n_samples, n_features = X.shape
    X_b = np.c_[np.ones(n_samples), X]  
    
    b = np.zeros(X_b.shape[1])
    
    for epoch in range(epochs):
        y_pred = X_b @ b
        error = y_pred - y
        gradients = (2 / n_samples) * X_b.T @ error
        
        # Actualizar los coeficientes
        b -= learning_rate * gradients
        
        # Controlar errores numericos
        if np.any(np.isnan(b)) or np.any(np.isinf(b)):
            print("error")
            break
        
        if epoch % 100 == 0:
            mse = np.mean(error ** 2)
            print(f'Epoch {epoch}: MSE = {mse}')
    
    return b

def predict(X, b):
    X_b = np.c_[np.ones(X.shape[0]), X]  
    return X_b @ b

def accuracy(actual, predicted):
    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    return correct / len(actual) * 100

# Calcular los coeficientes
b = coefficients(X_train, y_train)

'''REALIZAR PREDICCIONES'''
# Realizar predicciones en el conjunto de prueba
y_pred_continuous = predict(X_test, b)

# Convertir predicciones a clasificacion binaria
threshold = 0.5
y_pred_binary = [1 if p >= threshold else 0 for p in y_pred_continuous]

'''EVALUAR EL MODELO'''
accuracy_value = accuracy(y_test, y_pred_binary)
print(f'Precisión: {accuracy_value}%')

