import pdb
import json
import pandas as pd
import itertools
from datasets import load_dataset, Dataset, DatasetDict

def load_laptop(split, sep_token):
	'''
	Loads a designated split from SemEval14-laptop, returns Dataset object
	'''
	idx_dict = {'positive': 0, 'negative': 1, 'neutral': 2}

	json_path = f"./data/SemEval14-laptop/{split}.json"
	with open(json_path, 'r') as file:
		dataset = json.load(file)
	text = []
	label = []
	for _, v in dataset.items():
		text.append(f"{v['sentence']} {sep_token} {v['term']}")
		label.append(idx_dict[v['polarity']])
	dataset = {'text': text, 'label': label}
	return Dataset.from_dict(dataset)

def load_res(split, sep_token):
	'''
	Loads a designated split from SemEval14-res, returns Dataset object
	'''
	idx_dict = {'positive': 0, 'negative': 1, 'neutral': 2}

	json_path = f"./data/SemEval14-res/{split}.json"
	with open(json_path, 'r') as file:
		dataset = json.load(file)
	text = []
	label = []
	for _, v in dataset.items():
		text.append(f"{v['sentence']} {sep_token} {v['term']}")
		label.append(idx_dict[v['polarity']])
	dataset = {'text': text, 'label': label}
	return Dataset.from_dict(dataset)

def load_acl(split):
	'''
	Loads a designated split from acl_sup, returns Dataset object
	'''
	idx_dict = {'Uses': 0, 'Future': 1, 'CompareOrContrast': 2, 'Motivation': 3, 'Extends': 4, 'Background': 5}

	json_path = f"./data/acl_sup/{split}.jsonl"
	text = []
	label = []
	with open(json_path, 'r') as file:
		for line in file:
			data = json.loads(line)
			text.append(data['text'])
			label.append(idx_dict[data['label']])
	dataset = {'text': text, 'label': label}
	return Dataset.from_dict(dataset)

def load_agnews():
	'''
	Loads agnews_sup, splits it into train and test set, returns DatasetDict object
	'''
	csv_path = f"./data/agnews_sup/agnews_sup.csv"
	full_dataset = pd.read_csv(csv_path, names=['label', 'title', 'description'])
	label = [x - 1 for x in full_dataset['label'].tolist()] # labels: 1, 2, 3, 4 -> 0, 1, 2, 3
	full_dataset = Dataset.from_dict({'text': full_dataset['description'], 'label': label})
	train_test_split = full_dataset.train_test_split(test_size=0.1, seed=2022)
	return train_test_split
	

def get_dataset(dataset_name, sep_token):
	'''
	dataset_name: str, the name of the dataset
	sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
	'''
	dataset = None

	if isinstance(dataset_name, list):
		base_label = 0
		aggregated_train = {'text': [], 'label': []}
		aggregated_test = {'text': [], 'label': []}

		for name in dataset_name:
			cur_dataset = get_dataset(name, sep_token)

			train_label = [x + base_label for x in cur_dataset['train']['label']]
			test_label = [x + base_label for x in cur_dataset['test']['label']]

			aggregated_train['text'] += cur_dataset['train']['text']
			aggregated_train['label'] += train_label
			aggregated_test['text'] += cur_dataset['test']['text']
			aggregated_test['label'] += test_label

			base_label += max(cur_dataset['train']['label']) + 1
		
		return DatasetDict({'train': Dataset.from_dict(aggregated_train), 
							'test': Dataset.from_dict(aggregated_test)})

	assert isinstance(dataset_name, str), "Invalid dataset_name"

	dataset_name, version = dataset_name.split('_')
	assert version in ['sup', 'fs'], "Invalid version"

	if dataset_name == 'restaurant':
		dataset = DatasetDict({'train': load_res('train', sep_token), 
						 	   'test': load_res('test', sep_token)})
		
	elif dataset_name == 'laptop':
		dataset = DatasetDict({'train': load_laptop('train', sep_token), 
						 	   'test': load_laptop('test', sep_token)})
		
	elif dataset_name == 'acl':
		dataset = DatasetDict({'train': load_acl('train'), 
						 	   'test': load_acl('test')})
	
	elif dataset_name == 'agnews':
		dataset = load_agnews()
	
	if version == 'fs':
		seed = 8151
		num_labels = max(dataset['train']['label']) + 1
		if num_labels <= 5:
			dataset['train'] = dataset['train'].shuffle(seed=seed).select(range(32))
		else:
			dataset['train'] = dataset['train'].shuffle(seed=seed)
			selected_idx = [[] for _ in range(num_labels)]
			for idx, label in enumerate(dataset['train']['label']):
				if len(selected_idx[label]) < 8:
					selected_idx[label].append(idx)
			selected_idx = list(itertools.chain(*selected_idx))
			dataset['train'] = dataset['train'].select(selected_idx).shuffle(seed=seed)

	return dataset

if __name__ == "__main__":
	dataset_names = ['restaurant_sup', 'laptop_sup', 'acl_sup', 'agnews_sup']
	# fewshot_dataset_names = ['restaurant_fs', 'laptop_fs', 'acl_fs', 'agnews_fs']
	# for dataset_name in dataset_names:
	# 	dataset = get_dataset(dataset_name, '<sep>')
	# for fewshot_dataset_name in fewshot_dataset_names:
	# 	dataset = get_dataset(fewshot_dataset_name, '<sep>')
	dataset = get_dataset(dataset_names, '<sep>')
	pdb.set_trace()