from flask import Flask, render_template, request, send_file
import sqlite3
import NotaFinal_RL
import Salario_RL
import Cancelacion_RLo
from Supermercado_NC import predecir_cliente, guardar_datos_prediccion, obtener_dataset_pickle, descargar_csv_pickle, descargar_excel_pickle, obtener_dataset_picklen, descargar_csv_picklen, descargar_excel_picklen, obtener_metricass, obtener_matriz_confusion_base64
from ModelosClasificacion_BD import obtener_modelo_por_id, obtener_modelos, obtener_modelos_detalle
from Cancelacion_RLo import obtener_matriz_confusion, obtener_metricas


app = Flask(__name__, template_folder='Templates')

# Home 
@app.route("/")
def home():
    return render_template("inicioflask.html")
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Ruta para hora actual
@app.route("/hello/<name>")
def hello_there(name):
    from datetime import datetime
    import re
    
    now = datetime.now()
    match_object = re.fullmatch("[a-zA-Z]+", name)
    clean_name = match_object.group(0) if match_object else "Friend"
    
    content = f"Hello everyone!!!!!, {clean_name} ! Hour: {now}"
    return content 
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Ruta ejemplo de html 
@app.route("/examplehtml/")
def examplehtml():
    return render_template("EjemplosClase/example.html")
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Ruta para calcular la regresión lineal
@app.route("/linearegresion/", methods=["GET", "POST"])
def calculategrades():
    predictResult = None
    grafica = NotaFinal_RL.generate_plot()

    if request.method == "POST":
        hours = float(request.form["hours"])
        predictResult = NotaFinal_RL.calculateGrade(hours)

    return render_template("EjemplosClase/linearRegresionGrades.html", result=predictResult, plot_url=grafica)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------SEMANA 3------------------------------------------------------------------------
#-------------Libro Casos de uso---------#
@app.route("/Casodeuso/")
def Casodeuso():
    return render_template('Semana3/casodeuso.html')
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------SEMANA 4------------------------------------------------------------------------
#----------------Calculadora y Grafica---------------#
@app.route("/SalarioMensual/", methods=["GET", "POST"])
def calcularsalarios():
    predictResult = None
    grafico = Salario_RL.generate_plot()

    if request.method == "POST":
        salario = float(request.form["salario"])
        predictResult = Salario_RL.calcularsalario(salario)

    return render_template("Semana4/HtRegresion.html", result=predictResult, plot_url=grafico)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------SEMANA 5------------------------------------------------------------------------
#---------------Mapa Mental y explicacion-------------#
@app.route("/RegresionLogistica/")
def RegresionLogistica():
    return render_template("Semana5/RegresionLogistica.html")
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------SEMANA 6------------------------------------------------------------------------
#----------------Menu Navegacion---------------#
@app.route("/Telecomunicaciones/")
def menu():
    return render_template("Semana6/MenuNavegacionLogistica.html")

#----------------Ver Dataset---------------#
@app.route("/Telecomunicaciones/Dataset")
def dataset():
    from Cancelacion_RLo import obtener_dataset_html
    dataset_html = obtener_dataset_html()
    return render_template("Semana6/DatasetLogistica.html", dataset_html=dataset_html)

#----------------Calculadora para Predecir---------------#
@app.route("/Telecomunicaciones/Predecir", methods=["GET", "POST"])
def predecir():
    resultado = None

    if request.method == "POST":
        duracion_llamada = float(request.form["duracion_llamada"])
        plan_contratado = int(request.form["plan_contratado"])
        historial_pago = float(request.form["historial_pago"])
        
        # Usamos el modelo de regresión logística para predecir
        resultado = Cancelacion_RLo.predecir_cancelacion(duracion_llamada, plan_contratado, historial_pago)

    return render_template("Semana6/PredecirLogistica.html", resultado=resultado)

#----------------Resultados Metricas---------------#
@app.route('/Telecomunicaciones/Resultados')
def mostrar_resultados():
    matriz_confusion = obtener_matriz_confusion()
    accuracy, precision, recall = obtener_metricas()
    return render_template('Semana6/ResultadosLogistica.html', matriz_confusion=matriz_confusion, accuracy=accuracy, precision=precision, recall=recall)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------# -------------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------SEMANA 7-------------------------------------------------------------------------
#----------------Pagina Inicio con lista de todos los modelos---------------#
@app.route("/modelo/")
def index():
    modelos = obtener_modelos()
    modelos_detalle = obtener_modelos_detalle()
    return render_template("Semana7/index.html", modelos=modelos, modelos_detalle=modelos_detalle)

#----------------Pagina de Despiegle segun modelo seleccionado---------------#
@app.route("/modelo/<int:modelo_id>")
def modelo(modelo_id):
    modelos = obtener_modelos()  
    modelo = obtener_modelo_por_id(modelo_id)
    return render_template("Semana7/modelo.html", modelo=modelo, modelos=modelos)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ----------------------------------------------SEMANA 8--------------------------------------------------------------------
# ---------Menu Navegacion Principal---------#
@app.route("/Supermercado/")
def menusup():
    return render_template("Semana8/MenuNavegacionNcc.html")

# ---------Calculadora Predecir e ingresar nuevos datos---------#
@app.route("/Supermercado/Predecir", methods=["GET", "POST"])
def supermerpredecir():
    prediccion = None
    if request.method == "POST":
        frecuencia_visita = float(request.form["frecuencia_visita"])
        monto_gasto = float(request.form["monto_gasto"])
        categoria_compra = request.form["categoria_compra"]
        prediccion = predecir_cliente(frecuencia_visita, monto_gasto, categoria_compra)
        guardar_datos_prediccion(frecuencia_visita, monto_gasto, categoria_compra, prediccion)
    return render_template("Semana8/PredecirSup.html", prediccion=prediccion)

# ---------Menu Navegacion entre los dos Datasets---------#
@app.route("/Supermercado/Dataset/")
def menudat():
    return render_template("Semana8/MenuDataset.html")

# ---------Dataset Modelo Entrenamiento-------------------#
@app.route('/Supermercado/Dataset/Modelo', methods=['GET', 'POST'])
def datasetm_ncc():
    inicio_html, fin_html = obtener_dataset_pickle()

    if request.method == 'POST':
        if 'csv' in request.form:
            csv_file = descargar_csv_pickle()
            return send_file(
                csv_file,
                mimetype='text/csv',
                as_attachment=True,
                download_name='dataset_entrenamiento.csv'
            )
        elif 'excel' in request.form:
            excel_file = descargar_excel_pickle()
            return send_file(
                excel_file,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name='dataset_entrenamiento.xlsx'
            )

    return render_template('Semana8/DsNccM.html', inicio_html=inicio_html, fin_html=fin_html)


# ---------Dataset Modelo Nuevo creado con nuevos registros-------------------#
@app.route('/Supermercado/Dataset/Nuevo', methods=['GET', 'POST'])
def datasetn_ncc():
    dataset_html = obtener_dataset_picklen()

    if request.method == 'POST':
        if 'csv' in request.form:
            csv_file = descargar_csv_picklen()
            return send_file(
                csv_file,
                mimetype='text/csv',
                as_attachment=True,
                download_name='dataset_nuevo.csv'
            )
        elif 'excel' in request.form:
            excel_file = descargar_excel_picklen()
            return send_file(
                excel_file,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name='dataset_nuevo.xlsx'
            )

    return render_template('Semana8/DsNccN.html', dataset_html=dataset_html)


# ---------Resultados de metricas de modelo-------------------#
@app.route('/Supermercado/Resultados')
def resultados():
    accuracy, precision, recall = obtener_metricass()
    matriz_confusion = obtener_matriz_confusion_base64()

    return render_template('Semana8/ResultadosSup.html',accuracyy=accuracy, precisionn=precision,recalll=recall,matriz_confusion=matriz_confusion)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------