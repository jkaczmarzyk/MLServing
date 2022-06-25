from flask import jsonify, request
from app import app
import torch
import model_functions as mf
from transformers import BertTokenizer
import pandas as pd
from torch.utils.data import DataLoader



hyperparameters = dict(
    MAX_LENGTH = 300,
    TRAIN_BATCH_SIZE = 100,
    VALID_BATCH_SIZE = 50,
    epochs = 1,
    out_path = './models',
    log_freq=10,
    learn_rate = 1e-06
)
test_params = {
    'batch_size': hyperparameters['VALID_BATCH_SIZE'],
    'shuffle':True,
    'num_workers':0
}
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


@app.route('/emotion', methods=['POST'])
def classify_text():
  data = request.get_json()
  text = data['text']
  print(text)
  #model inference here
  
  filename = './app/tinybert.pth'
  mdl_name = 'google/bert_uncased_L-2_H-128_A-2'
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


  df = [text,1]
  df = pd.DataFrame(df).T
  print(df)
  df.columns = ['text','list']
  

  testing_set = mf.CustomDataSet(df, tokenizer, hyperparameters['MAX_LENGTH'])

  testing_loader = DataLoader(testing_set, **test_params)


  # potential improvement: not load class each time
  bert_model = mf.BERTClass(mdl_name)
  bert_model.load_state_dict(torch.load(filename,map_location=device))
  bert_model.to(device)
  preds = mf.inference(bert_model,testing_loader,device)
  preds = [str(p) for p in preds]
  prediction = {'prediction':preds}
  

  return prediction
