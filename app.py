from flask import Flask, request, render_template, jsonify, send_file
import torch
import cv2
import numpy as np
import pickle
import os

app = Flask(__name__)

# Configurações
MODEL_PATH = 'best_model.pkl'  # Atualize com o caminho real do seu arquivo pickle
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Carregar o modelo
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)
model.to(DEVICE)
model.eval()

def image_view_and_classify(image_path, model):
    # Carregar a imagem
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Não foi possível carregar a imagem - {image_path}")

    try:
        # Redimensionar a imagem para visualização e processamento
        imageview = cv2.resize(image, (512, 512))
        image = cv2.resize(image, (128, 128))
    except Exception as e:
        raise ValueError(f"Erro ao redimensionar a imagem: {e}")

    # Pré-processamento da imagem para a rede neural
    image = image / 255.0
    image = image.transpose(2, 0, 1)  # Reordenar para formato (C, H, W)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Adicionar batch dimension

    # Enviar a imagem para o dispositivo (GPU ou CPU)
    image = image.to(DEVICE)

    # Classificação da imagem
    with torch.no_grad():
        result = model(image)
        probabilities = torch.nn.functional.softmax(result, dim=1)  # Obter as probabilidades
        max_prob, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.item()
        max_prob = max_prob.item()

    # Gerar a resposta baseada na classificação
    if predicted_class == 0:
        resposta = 'Eu acho que vi um gatinho'
    else:
        resposta = 'Olha que catiorinho mais lindo'

    # Gerar a linha com a probabilidade
    probabilidade_str = f'Probabilidade: {max_prob*100:.2f}%'

    # Desenhar a resposta na imagem
    x, y, w, h = 0, 0, 512, 45
    cv2.rectangle(imageview, (x, y), (x + w, y + h * 2), (0, 0, 0), -1)  # Estender o retângulo para duas linhas
    cv2.putText(imageview, resposta, (x + int(w / 10), y + int(h / 2)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (200, 100, 200), 1)
    cv2.putText(imageview, probabilidade_str, (x + int(w / 10), y + int(h * 3 / 2)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (200, 100, 200), 1)

    # Salvar a imagem com a resposta
    result_image_path = 'static/result_image.jpg'
    cv2.imwrite(result_image_path, imageview)
    return result_image_path

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400

    if file:
        # Salvar o arquivo enviado
        image_path = 'static/uploaded_image.jpg'
        file.save(image_path)

        try:
            # Visualizar e classificar a imagem
            result_image_path = image_view_and_classify(image_path, model)
            return send_file(result_image_path, mimetype='image/jpeg')
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Erro desconhecido'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
