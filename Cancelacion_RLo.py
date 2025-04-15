import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Función para generar dataset
def generar_dataset():
    np.random.seed(42)  # Fijar semilla para reproducibilidad
    
    # Generación de variables independientes
    duracion_llamada = np.random.randint(1, 61, 100)  # Duración en minutos (1 a 60)
    plan_contratado = np.random.choice([0, 1], size=100)  # 0 = No, 1 = Sí
    historial_pago = np.random.rand(100)  # Valores entre 0 y 1
    
    # Generación de variable objetivo (cancelación de servicio)
    cancelacion = np.where((duracion_llamada < 30) & (plan_contratado == 0) & (historial_pago < 0.5), 1, 0)
    
    # Creación del DataFrame
    dataset = pd.DataFrame({
        "Duracion_Llamada": duracion_llamada,
        "Plan_Contratado": plan_contratado,
        "Historial_Pago": historial_pago,
        "Cancelacion": cancelacion
    })
    
    return dataset

# Entrenar el modelo de regresión logística
def entrenar_modelo():
    dataset = generar_dataset()
    
    X = dataset[["Duracion_Llamada", "Plan_Contratado", "Historial_Pago"]]  # Variables independientes
    y = dataset["Cancelacion"]  # Variable objetivo/categorica
    
    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear y entrenar el modelo
    modelo = LogisticRegression(class_weight='balanced')
    modelo.fit(X_train, y_train)
    
    
    return modelo, X_test, y_test

# Función para predecir si un cliente cancelará su servicio
def predecir_cancelacion(duracion_llamada, plan_contratado, historial_pago):
    modelo, _, _ = entrenar_modelo()
    datos_entrada = np.array([[duracion_llamada, plan_contratado, historial_pago]])
    prediccion = modelo.predict(datos_entrada)[0]
    
    return "Sí" if prediccion == 1 else "No"

# Función para obtener la matriz de confusión
def obtener_matriz_confusion():
    modelo, X_test, y_test = entrenar_modelo()
    y_pred = modelo.predict(X_test)
    
    matriz = confusion_matrix(y_test, y_pred)
    
    # Graficar la matriz de confusión
    plt.figure(figsize=(5, 5))
    plt.imshow(matriz, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusión")
    plt.colorbar()
    
    clases = ["No Canceló", "Canceló"]
    plt.xticks([0, 1], clases)
    plt.yticks([0, 1], clases)
    
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matriz[i, j]), ha="center", va="center", color="black")
    
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    
    # Guardar la imagen en base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    matriz_base64 = base64.b64encode(img.getvalue()).decode()

    plt.close()
    
    return matriz_base64

def obtener_dataset_html():
    dataset = generar_dataset()
    return dataset.to_html(classes='table table-bordered', index=False)

# Función para calcular métricas de evaluación
def obtener_metricas():
    modelo, X_test, y_test = entrenar_modelo()
    y_pred = modelo.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    return accuracy, precision, recall
