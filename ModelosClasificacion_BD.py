import sqlite3

conn = sqlite3.connect("data/modelos.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS modelos_clasificacion (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre_modelo TEXT NOT NULL,
    descripcion TEXT NOT NULL,
    fuente_informacion TEXT NOT NULL,
    contenido_grafico TEXT NOT NULL
)
""")

modelos = [
    (
        "Regresión Logística",
        "La regresión logística es un método matemático que analiza relaciones entre datos para predecir resultados limitados (como sí/no), siendo clave en machine learning por su precisión y simplicidad. Es rápida, interpretable y flexible, pudiendo manejar múltiples variables. Utiliza una función sigmoidea para transformar variables en probabilidades entre 0 y 1, determinando así el resultado más probable de forma eficiente.",
        "https://aws.amazon.com/es/what-is/logistic-regression/",
        "https://d1.awsstatic.com/sigmoid.bfc853980146c5868a496eafea4fb79907675f44.png"
    ),
    (
        "K-Nearest Neighbors (KNN)",
        "K-Nearest Neighbors (KNN) es un algoritmo de aprendizaje supervisado que clasifica nuevos datos comparándolos con los k ejemplos más cercanos del conjunto de entrenamiento, usando métricas de distancia como euclidiana o Manhattan. En clasificación, asigna la clase más frecuente entre los vecinos; en regresión, el promedio de sus valores. Al no generar un modelo previo, se considera 'perezoso' y basado en instancias.",
        "https://www.ibm.com/mx-es/topics/knn",
        "https://media2.dev.to/dynamic/image/width=1000,height=420,fit=cover,gravity=auto,format=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fqlg7ojo2w4rx027cqaqn.png"
    ),
    (
        "Árboles de Decisión",
        "Los árboles de decisión son modelos predictivos que resuelven problemas de clasificación y regresión mediante una estructura jerárquica de nodos y ramas. Comienzan con un nodo raíz, dividen los datos en subconjuntos homogéneos basados en características clave (usando métricas como ganancia de información) y terminan en nodos hoja con resultados finales. Aunque son interpretables y visuales, tienden al sobreajuste.",
        "https://www.ibm.com/es-es/think/topics/decision-trees",
        "https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/cdp/cf/ul/g/df/de/Decision-Tree.png"
    ),
    (
        "Random Forest",
        "El Random Forest mejora la precisión en clasificación y regresión combinando múltiples árboles de decisión, cada uno entrenado con muestras y características aleatorias para evitar sobreajuste. Mediante votación mayoritaria o promedios, ofrece flexibilidad, manejo de datos faltantes y evaluación de variables importantes.",
        "https://www.ibm.com/mx-es/think/topics/random-forest",
        "https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/cdp/cf/ul/g/50/f9/ICLH_Diagram_Batch_03_27-RandomForest.png"
    ),
    (
        "Support Vector Machine (SVM)",
        "Las Máquinas de Vectores de Soporte (SVM) son algoritmos supervisados que identifican el hiperplano óptimo para separar clases con el máximo margen, usando vectores de soporte para mejorar la generalización. Pueden resolver problemas lineales y no lineales mediante kernels (polinómico, RBF, sigmoide), transformando los datos a espacios de mayor dimensión. Aunque su entrenamiento es computacionalmente costoso, destacan por su precisión y resistencia al sobreajuste.",
        "https://www.ibm.com/mx-es/think/topics/support-vector-machine",
        "https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/creative-assets/s-migr/ul/g/8f/27/3-1_svm_optimal-hyperplane_max-margin_support-vectors-2-1.png"
    ),
    (
        "Gradient Boosting (XGBoost, AdaBoost, etc.)",
        "El Boosting entrena secuencialmente modelos simples (como árboles pequeños), donde cada nuevo modelo corrige los errores del anterior enfocándose más en los datos mal clasificados. Estos 'aprendices débiles' se combinan al final para formar un predictor fuerte y preciso. AdaBoost ajusta pesos, Gradient Boosting trabaja con residuos, y XGBoost optimiza el proceso para mayor velocidad y escalabilidad.",
        "https://www.ibm.com/es-es/topics/boosting",
        "https://almablog-media.s3.ap-south-1.amazonaws.com/image_28_7cf514b000.png"
    ),
    (
        "Naive Bayes",
        "Naive Bayes es un clasificador probabilístico basado en el teorema de Bayes que asume independencia entre características (aunque rara vez sea cierto), lo que lo hace simple y eficiente. Incluye variantes como Gaussian, Multinomial y Bernoulli. A pesar de su simplicidad, ofrece buen rendimiento incluso con pocos datos o alta dimensionalidad, siendo útil en análisis de sentimientos o diagnóstico médico.",
        "https://www.ibm.com/es-es/think/topics/naive-bayes",
        "https://nicolovaligi.com/articles/naive-bayes-tensorflow/tf_iris.png"
    )

]

cursor.execute("SELECT COUNT(*) FROM modelos_clasificacion")
cantidad = cursor.fetchone()[0]

if cantidad == 0:
    cursor.executemany("""
    INSERT INTO modelos_clasificacion (nombre_modelo, descripcion, fuente_informacion, contenido_grafico)
    VALUES (?, ?, ?, ?)
    """, modelos)
    print("Datos insertados.")
else:
    print("Los datos ya existen. No se insertaron nuevamente.")



DB_PATH = "data/modelos.db"

def obtener_modelos():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, nombre_modelo FROM modelos_clasificacion")
    modelos = cursor.fetchall()
    conn.close()
    return modelos

def obtener_modelos_detalle():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT nombre_modelo, descripcion, fuente_informacion, contenido_grafico FROM modelos_clasificacion")
    modelos = cursor.fetchall()
    conn.close()
    return modelos

def obtener_modelo_por_id(modelo_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM modelos_clasificacion WHERE id = ?", (modelo_id,))
    modelo = cursor.fetchone()
    conn.close()
    return modelo


#def eliminar_modelos_duplicados():
#    conn = sqlite3.connect(DB_PATH)
#    cursor = conn.cursor()
#
#    cursor.execute("""
#        DELETE FROM modelos_clasificacion
#        WHERE id NOT IN (
#            SELECT MIN(id)
#            FROM modelos_clasificacion
#            GROUP BY nombre_modelo
#        )
#    ")

#    conn.commit()
#    conn.close()
#    print("Modelos duplicados eliminados correctamente.")

#eliminar_modelos_duplicados()