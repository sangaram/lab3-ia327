from utils import clean
from torch.utils.data import Dataset
import random

def read_corpus(file):
    with open(file, "r", encoding="utf-8") as f:
        line = f.readline()
        raw_data = []
        while line:
            if line.replace("\n","") != "":
                raw_data.append(line.replace("\n","").split("\t"))
            line = f.readline()
    return raw_data

def prepare_text(subject, object, sentence):
    '''
    YOUR CODE GOES HERE

    Prepares the text that will be given as input to the model.
    The sentence expresses the relation between the subject and the object (which necessarily appears in the sentence).
    
    The model will try to predict what kind of relationship holds between the subject and the object.
    Thus, your input will be a text (1 sentence) that gives as much relevant information to the model as possible.
    
    '''

    return subject + " " + sentence + " " + object

def prepare_data(rels_dict, mode = "train"):
    '''
    Prepares the dataset that will be given to the model.
    subj and obj are used for printing your results in the end, and are not used during training. Only input_sentence and label will used.
    '''
    assert mode in {"train", "test"}
    raw_corpus = read_corpus("silver-train.tsv")
    if mode == "test":
        raw_corpus = read_corpus("silver-test.tsv")
    
    data = []
    for subj, obj, sentence, rel in raw_corpus:
        rel_id = rels_dict[rel]
        input_sentence = prepare_text(subj, obj, sentence)
        if mode == "train":
            data.append((input_sentence, rel_id, subj, obj))
        else:
            data.append((input_sentence, rel_id, subj, obj))
    return data

class FactsDataset(Dataset):
    def __init__(self, data, model):
        '''
        This class implements a Dataset class adapted to the task
        '''
        self.model = model
        self.examples = []
        for element in data:
            input_text, label, subj, obj = element
            vectorized_text_tuple = self.model.vectorize_input(input_text)
            self.examples.append((vectorized_text_tuple, label, subj, obj))

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)