from flask import jsonify, request
from app import app
from model_handler import ModelHandler

handler = ModelHandler()

@app.route('/emotion', methods=['POST'])
def classify_text():
  data = request.get_json()
  text = data['text']
  #print(text)

  return handler.inference(text)
