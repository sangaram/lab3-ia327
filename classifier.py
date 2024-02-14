from transformers import BertTokenizer, BertModel
import torch

class ClassificationModel(torch.nn.Module):
    '''
    The classification model, based on BERT.

    You can change the architecture of the model (e.g. add layers after BERT for classification), as well as the BERT embeddings you wish to use (and how to process them) with the vectorize_input and forward methods, using masks to access these embeddings easily.
    '''
    def __init__(self, device, rels_dict):
        super(ClassificationModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        
        self.bert_model = BertModel.from_pretrained('bert-base-cased')
        self.hidden_size = self.bert_model.config.hidden_size
        self.classification_layer = torch.nn.Linear(self.hidden_size, len(rels_dict))
        self.device = device

    def forward(self, inputs):
        input_ids, input_mask, output_tokens_mask = inputs
        input_ids, input_mask, output_tokens_mask = input_ids.to(self.device), input_mask.to(self.device), output_tokens_mask.to(self.device)
        bert_output = self.bert_model(input_ids, attention_mask=input_mask)
        bert_CLS_output = bert_output[1] #BERT embedding of the CLS classification token (first token)
        #tokens_mask_output = torch.bmm(output_tokens_mask.float().unsqueeze(1), bert_output[0]).squeeze(1) #summed embeddings of the tokens corresponding to the output mask. You can use several masks in order to access several interesting embedding separately.
        
        classif_input = bert_CLS_output
        
        logits = self.classification_layer(classif_input)
        return logits
    
    def vectorize_input(self, text):
        '''
        For each text, it produces: 
        - input_ids, a tensor containing the tokenized version of the input text
        - input_mask, the padding tensor (indicates if the token at each index is part of the text or just filling to make all vectors the same size)
        - output_tokens_mask, a tensor indicating the indexes of tokens of which you want to use the embedding for classification

        You are welcome to modify this function, particularly the output tokens mask
        '''
        max_seq_len = 512 #this is the maximum context length for BERT
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > max_seq_len - 2:
            tokens = tokens[:(max_seq_len - 2)]
        tokens = tokens + ["[SEP]"]

        tokens =  ["[CLS]"]+ tokens
        input_mask = [1] * len(tokens)

        output_tokens_mask = [0] * len(tokens)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
        input_mask = input_mask + ([self.tokenizer.pad_token_id] * padding_length)
        output_tokens_mask = output_tokens_mask+ ([self.tokenizer.pad_token_id] * padding_length)

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(output_tokens_mask) == max_seq_len

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        output_tokens_mask = torch.tensor(output_tokens_mask, dtype=torch.long)

        return(input_ids, input_mask, output_tokens_mask)