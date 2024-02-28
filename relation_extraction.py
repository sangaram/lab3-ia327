from utils import run_evaluation
from corpus_building import prepare_data, FactsDataset
from torch.utils.data import DataLoader, RandomSampler
from classifier import ClassificationModel
import torch.optim as optim
import torch
import numpy as np
from pathlib import Path
import time
import tqdm

"""
=== Purpose ===
The objective of this lab is to perform Relation Extraction.
That is, from two entities and a sentence, we want to predict the relation that holds between the two entities.

You will perform this task by fine-tuning a BERT model on a dataset you will create from an automatically produced set of facts.

=== Provided Data ===

We provide:

1. silver-train.tsv: a collection of facts where each element is a quadruplet (subject, object, sentence, relation), that you will use for training.
2. silver-test.tsv: a similar collection, where all relations are masked, that will be used for testing.
3. classifier.py, corpus_building.py, and relation_extraction.py: templates for your code.
4. utils.py with evaluate() function to evaluate your approach on your development set.

=== Task ===

Your task is to build a dataset and train a model for the task of relation extraction.

You need to complete the functions prepare_data and prepare_text (in corpus_building.py) to build the dataset that will be given to the model.
prepare_text shall output a single sentence that will be used as model input.

This sentence will be vectorized with the vectorize_input function from the ClassificationModel class (in classifier.py).
You can modify this function, as well as the architecture of the model if you wish to use advanced techniques using masks to access specific BERT embeddings.

Finally, you can also modify the training parameters in the train() function.


=== Development ===

The initialize_data function will automatically create a development set, that will be used to evaluate your model during training. After training, you can also use the run_evaluation function from utils.py to evaluate your model on this dataset. 
We use the F2-measure for scoring, which gives more weight to recall.


=== Evaluation ===

The output of your model on the test set will be evaluated using the same F2-measure, but with our gold standard.
The results of your model on the dev set help you to make your model as good as possible.


### === Submission ===

1. Take your code, any necessary resources to run the code, and the output of your code (2 files: results_dev.tsv and results_test.tsv) on the dev/test datasets.
2. ZIP these files in a file called firstName_lastName.zip
3. Submit it on eCampus before the deadline.


### === Contact ===

Don't hesitate to ask your questions by sending an email to: zacchary.sadeddine@telecom-paris.fr
"""

def initialize_model(rels_dict):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = ClassificationModel(device, rels_dict)
	model = model.to(device)
	print("Model Loaded")
	return model, device

def initialize_data(rels_dict, model):
	corpus = prepare_data(rels_dict)
	dev_cut = round(len(corpus)*0.8) #used for train / dev
	train_dataset = FactsDataset(corpus[:dev_cut], model)
	dev_dataset = FactsDataset(corpus[dev_cut:], model)

	test_data = prepare_data(rels_dict, mode="test")
	test_dataset = FactsDataset(test_data, model)
	print("Datasets created", len(train_dataset), len(dev_dataset), len(test_dataset))

	return train_dataset, dev_dataset, test_dataset

def compute_class_weights(labels):
	#idx_to_rel = {idx: rel for (rel, idx) in rels_dict.items()}
	#columns = ["subject", "object", "sentence", "relation"]
	#df = pd.read_table(data_path, header=None, names=columns, sep='\t')
	#rels, counts = np.unique(df['relation'].dropna().to_numpy(), return_counts=True)
	#assert set(rels) == set(rels_dict.keys()), "Error, unexpected relations encountered in the relation dict"
	_, counts = np.unique(labels, return_counts=True)
	probs = counts / counts.sum()
	#class_weights_dict = {rel: 1. / prob for (rel, prob) in zip(rels, probs)}
	#class_weights = [class_weights_dict[idx_to_rel[idx]] for idx in range(len(rels))]
	class_weights = torch.tensor(1 / probs, dtype=torch.float32)
	return class_weights

def train(train_dataset, dev_dataset, model, class_weights, device):
	'''
	A few parameters you can play with. The higher the batch size, the higher memory usage.
	'''
	num_epochs = 5
	learning_rate=2e-5
	batch_size = 20
	update_every_n_steps = 10

	train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
	dev_loader = DataLoader(dev_dataset, sampler=RandomSampler(dev_dataset), batch_size=batch_size)
	optimizer = optim.Adam(model.parameters(),lr=learning_rate)

	optimizer.zero_grad()

	loss = torch.nn.CrossEntropyLoss(weight=class_weights)
	training_losses, dev_losses = [], []
	train_progress = tqdm.trange(0, num_epochs * len(train_loader))
	for epoch in range(num_epochs):
		training_loss = 0
		dev_loss = 0
		model.train()
		print("Training starts")
		for step, (examples, labels, _, _) in enumerate(train_loader):
			labels = labels.to(device)

			outputs = model(examples)

			batch_loss = loss(outputs, labels)
			training_loss += batch_loss.item()
			batch_loss.backward()
			if (step + 1) % update_every_n_steps == 0:
				optimizer.step()
				optimizer.zero_grad()
				del outputs, batch_loss
				torch.cuda.empty_cache()
			train_progress.update(1)
			
		optimizer.step()
		optimizer.zero_grad()
		del outputs, batch_loss
		torch.cuda.empty_cache()

		#Every step, we check the loss on the dev set (helps detecting overfitting)
		model.eval()
		print("Starting evaluation ...")
		eval_progress = tqdm.trange(0, len(dev_loader))
		with torch.no_grad():
			for step, (examples, labels, _, _) in enumerate(dev_loader):
				labels = labels.to(device)
				outputs = model(examples)

				batch_loss = loss(outputs, labels)
				dev_loss += batch_loss.item()

				del outputs, batch_loss
				torch.cuda.empty_cache()
				eval_progress.update(1)

			print("Training Loss - Epoch " + str(epoch) + " - " + str(training_loss))
			print("Dev Loss - Epoch " + str(epoch) + " - " + str(dev_loss))
			training_losses.append(training_loss)
			dev_losses.append(dev_loss)

		savepath = Path(f"./checkpoints/{time.strftime("%d_%m_%Y_%H_%M_%S")}_checkpoint.pt")
		torch.save({
			"model_classifier_state_dict": model.classification_layer.state_dict(),
			"optimize_state_dict": optimizer.state_dict(),
			"num_epochs": num_epochs,
			"epoch": epoch,
			"learning_rate": learning_rate,
			"batch_size": batch_size,
			"update_every_n_steps": update_every_n_steps
		}, savepath)
		
	print(training_losses)
	print(dev_losses)
	return model

def infer_write(dataset, model, invert_rels_dict, mode="dev"):
	print("Inference starts - " + mode)
	assert mode in {"dev", "test"}
	with open("results-"+mode+".tsv", 'wt', encoding="utf-8") as output:
		with torch.no_grad():
			model.eval()
			loader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=1)
			for example, _, subj_entity, obj_entity in loader:
				outputs = model(example)
				pred = torch.argmax(outputs, dim = -1).item()
				pred_rel = invert_rels_dict[pred]
				output.write(subj_entity[0] +"\t" + pred_rel + "\t" + obj_entity[0] + "\n")
	print("Inference over")
	
def run():
	rels_dict = {'no_rel': 0, '<label>': 1, '<dateCreated>': 2, '<location>': 3, '<actor>': 4, '<birthDate>': 5, '<deathDate>': 6, '<director>': 7, '<memberOf>': 8, '<locationCreated>': 9, '<birthPlace>': 10, '<author>': 11, '<deathPlace>': 12}
	invert_rels_dict = {id: rel for rel,id in rels_dict.items()}

	model, device = initialize_model(rels_dict)
	train_dataset, dev_dataset, test_dataset = initialize_data(rels_dict, model)
	#class_weights = compute_class_weights("./silver-train.tsv", rels_dict).to(device)
	labels = list(map(lambda x: x[1], train_dataset))
	class_weights = compute_class_weights(labels).to(device)
	
	model = train(train_dataset, dev_dataset, model, class_weights, device)
	infer_write(dev_dataset, model, invert_rels_dict)
	run_evaluation()
	infer_write(test_dataset, model, invert_rels_dict, mode="test")

if __name__ == "__main__":
	run()