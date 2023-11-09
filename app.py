from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import torch
from torchvision import transforms
from PIL import Image
import io
import timm
import json

# Iniciar la aplicación Flask
app = Flask(__name__, static_folder='static')
CORS(app)

# Cargar el mapeo de índices a nombres de clases
with open('idx_to_class.json') as f:
    idx_to_class = json.load(f)

# Definir las transformaciones que se aplicarán a cada imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD)
])

# Función para cargar el modelo completo
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

# Cargar el modelo (reemplaza 'model_path' con la ruta de tu archivo de modelo guardado)
model_path = 'TomatoDiseasesModel_1.pth'
model = load_model(model_path)

@app.route('/')
def index():
    return app.send_static_file('index.html')

# Definir la ruta de la API para la predicción
@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        # Asegurarse de que se envíe un archivo con la solicitud
        if 'file' not in request.files:
            return jsonify({'error': 'no file'}), 400
        
        # Leer el archivo de imagen
        file = request.files['file']
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'}), 400
        
        try:
            img_bytes = file.read()
            image = Image.open(io.BytesIO(img_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = transform(image).unsqueeze(0)

            # Hacer la predicción
            with torch.no_grad():
                outputs = model(image)
                outputs = outputs[:, -1, -1, :]
                _, predicted = torch.max(outputs.data, 1)
            
            # Convertir la predicción a una respuesta JSON
            prediction = predicted.item()
            class_name = idx_to_class[str(prediction)]
            return jsonify({'class_id': prediction, 'class_name': class_name})

        except Exception as e:
            print(e)
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()