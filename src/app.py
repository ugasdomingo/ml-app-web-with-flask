from flask import Flask, request, render_template_string, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Cargar el modelo al arrancar la app
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'modelo_diabetes.pkl')
with open(MODEL_PATH, 'rb') as f:
    artefacto = pickle.load(f)

modelo   = artefacto['modelo']
scaler   = artefacto['scaler']
features = artefacto['features']

HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Predictor de Diabetes</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; }
        h1   { color: #2c3e50; }
        label{ display: block; margin-top: 12px; font-weight: bold; color: #555; }
        input{ width: 100%; padding: 8px; margin-top: 4px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
        button{ margin-top: 20px; padding: 12px 30px; background: #2980b9; color: white;
                border: none; border-radius: 4px; font-size: 16px; cursor: pointer; }
        button:hover { background: #1a6a9a; }
        .resultado { margin-top: 25px; padding: 15px; border-radius: 6px; font-size: 18px; font-weight: bold; }
        .positivo  { background: #fde8e8; color: #c0392b; border: 1px solid #e74c3c; }
        .negativo  { background: #e8f8f0; color: #1e8449; border: 1px solid #27ae60; }
        .info      { font-size: 12px; color: #888; margin-top: 5px; }
    </style>
</head>
<body>
    <h1>Predictor de Diabetes</h1>
    <p>Introduce los datos del paciente para obtener una predicción.</p>
    <form method="POST" action="/predict">
        <label>Embarazos (Pregnancies)</label>
        <input type="number" name="Pregnancies" min="0" max="20" step="1" value="1" required>

        <label>Glucosa en plasma (Glucose) <span class="info">mg/dL</span></label>
        <input type="number" name="Glucose" min="50" max="250" step="1" value="120" required>

        <label>Presión arterial (BloodPressure) <span class="info">mm Hg</span></label>
        <input type="number" name="BloodPressure" min="40" max="140" step="1" value="70" required>

        <label>Grosor pliegue cutáneo (SkinThickness) <span class="info">mm</span></label>
        <input type="number" name="SkinThickness" min="5" max="100" step="1" value="20" required>

        <label>Insulina sérica (Insulin) <span class="info">mu U/ml</span></label>
        <input type="number" name="Insulin" min="10" max="900" step="1" value="80" required>

        <label>Índice de masa corporal (BMI)</label>
        <input type="number" name="BMI" min="10" max="70" step="0.1" value="25.0" required>

        <label>Función de pedigrí de diabetes (DiabetesPedigreeFunction)</label>
        <input type="number" name="DiabetesPedigreeFunction" min="0.0" max="2.5" step="0.001" value="0.5" required>

        <label>Edad (Age)</label>
        <input type="number" name="Age" min="18" max="100" step="1" value="30" required>

        <button type="submit">Predecir</button>
    </form>

    {% if resultado is not none %}
    <div class="resultado {{ clase_css }}">
        {{ mensaje }}
        <div style="font-size:14px; font-weight:normal; margin-top:5px;">
            Probabilidad de diabetes: {{ probabilidad }}
        </div>
    </div>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML, resultado=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        valores = [float(request.form.get(f, 0)) for f in features]
        X = np.array([valores])
        X_sc = scaler.transform(X)

        pred  = modelo.predict(X_sc)[0]
        proba = modelo.predict_proba(X_sc)[0][1]

        if pred == 1:
            mensaje   = 'POSITIVO — El modelo predice diabetes'
            clase_css = 'positivo'
        else:
            mensaje   = 'NEGATIVO — El modelo no detecta diabetes'
            clase_css = 'negativo'

        return render_template_string(
            HTML,
            resultado=pred,
            mensaje=mensaje,
            clase_css=clase_css,
            probabilidad=f'{proba:.1%}'
        )
    except Exception as e:
        return render_template_string(HTML, resultado=-1,
                                      mensaje=f'Error: {e}',
                                      clase_css='negativo',
                                      probabilidad='—')

# Endpoint JSON para uso programático
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    valores = [float(data.get(f, 0)) for f in features]
    X_sc = scaler.transform(np.array([valores]))
    pred  = int(modelo.predict(X_sc)[0])
    proba = float(modelo.predict_proba(X_sc)[0][1])
    return jsonify({'prediccion': pred, 'probabilidad_diabetes': proba})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
