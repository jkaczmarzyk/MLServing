import torch
from torch.utils.data import Dataset
import transformers

def inference(model,test_loader,device):

  model.eval()
  preds = []


  with torch.no_grad():
    for _, data in enumerate(test_loader,0):
        ids = data['ids'].to(device,dtype=torch.long)
        mask = data['mask'].to(device,dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device,dtype=torch.long)
        targets = data['targets'].to(device,dtype=torch.float)
        s = targets.size(dim=0)
        targets = targets.resize(s,1)
        outputs = model(ids,mask,token_type_ids)
        o = torch.sigmoid(outputs).float().cpu().detach().numpy()
        for pred in o:
          preds.append(pred[0])


    return preds

class BERTClass(torch.nn.Module):

    def __init__(self,model_name):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained(model_name)
        self.l2 = torch.nn.Dropout(0.2)
        self.l3 = torch.nn.Linear(128,1)
    
    def forward(self,ids,mask, token_type_ids):
        _ , output_1 = self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)

        return output

class CustomDataSet(Dataset):

    def __init__(self,dataframe,tokenizer,max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.text
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)
    
    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
            'ids': torch.tensor(ids,dtype=torch.long),
            'mask': torch.tensor(mask,dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }