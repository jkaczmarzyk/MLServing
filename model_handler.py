from app import app
import torch
import model_functions as mf
from transformers import BertTokenizer
import pandas as pd
from torch.utils.data import DataLoader
from yaml import load, Loader

class ModelHandler():

    def __init__(self) -> None:


        with open('./config.yml','r') as yml_file:
            self.config = load(yml_file,Loader=Loader)


        print(self.config)
        self.MAX_LENGTH = self.config['model']['max-length']
        self.VALID_BATCH_SIZE = self.config['model']['valid-batch-size']


        self.test_params = {
            'batch_size': self.VALID_BATCH_SIZE,
            'shuffle': self.config['model']['test-parameters']['shuffle'],
            'num_workers': self.config['model']['test-parameters']['number-workers']
        }

        self.tokenizer = BertTokenizer.from_pretrained(self.config['model']['tokenizer'])

        self.filename = self.config['model']['filename']
        self.mdl_name = self.config['model']['model-name']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bert_model = mf.BERTClass(self.mdl_name)
        self.bert_model.load_state_dict(torch.load(self.filename,map_location=self.device))
        self.bert_model.to(self.device)

    def inference(self, text):
        df = [text,1]
        df = pd.DataFrame(df).T
        print(df)
        df.columns = ['text','list']


        testing_set = mf.CustomDataSet(df, self.tokenizer, self.MAX_LENGTH)

        testing_loader = DataLoader(testing_set, **self.test_params)

        preds = mf.inference(self.bert_model,testing_loader,self.device)
        preds = [str(p) for p in preds]
        prediction = {'prediction':preds}

        return prediction