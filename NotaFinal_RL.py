import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io 
import base64
from sklearn.linear_model import LinearRegression

data = {
    "Study Hours": [10, 15, 12, 8, 14, 5, 16, 7, 11, 13, 9, 4, 18, 3, 17, 6, 14, 2, 20, 1],
    "Final Grade": [3.8, 4.2, 3.6, 3, 4.5, 2.5, 4.8, 2.8, 3.7, 4, 3.2, 2.2, 5, 1.8, 4.9, 2.7, 4.4, 1.5, 5, 1]
}

df = pd.DataFrame(data)
df = pd.DataFrame(data)
x = df[["Study Hours"]]
y = df[["Final Grade"]]

model = LinearRegression()
model.fit(x,y)
def calculateGrade(hours):
    result = model.predict([[hours]])[0]
    return result
    
#generar el grafico 
def generate_plot():
    """Genera un gráfico de regresión lineal y lo convierte en imagen base64"""
    plt.figure(figsize=(8, 6))
    plt.scatter(df["Study Hours"], df["Final Grade"], color="blue", label="Datos reales")
    plt.plot(df["Study Hours"], model.predict(x), color="red", linewidth=2, label="Regresión lineal")
    plt.xlabel("Horas de Estudio")
    plt.ylabel("Calificación Final")
    plt.title("Regresión Lineal: Horas de Estudio vs Calificación")
    plt.legend()

    # Guardar la imagen en memoria
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    grafica = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return grafica