from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

class DataLoaderHF():
	def __init__(self, model_name, max_len, batch_size, sos_token):
		self.model_name = model_name
		self.batch_size = batch_size
		self.max_len = max_len
		self.sos_token = sos_token

		# 使用Hugging Face的AutoTokenizer进行
		self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

		print('Dataset initializing start')

	def make_dataset(self):
		# 使用Hugging Face加载Multi30k数据集
		dataset = load_dataset("bentrevett/multi30k")

		print('Dataset loaded')

		# 返回训练集，验证集和测试集
		train_data = dataset['train']
		valid_data = dataset['validation']
		test_data = dataset['test']

		return train_data, valid_data, test_data

	def tokenize_function(self, example):
		source_text = [text for text in example['de']]
		target_text = [self.sos_token + text for text in example['en']]
		
		input_ids = self.tokenizer(source_text, padding="max_length", truncation=True, max_length=self.max_len)['input_ids']
		labels = self.tokenizer(target_text, padding='max_length', truncation=True, max_length=self.max_len)['input_ids']

		return {
			'input_ids': input_ids,
			'labels': labels,
		}

	def make_iter(self, train_data, valid_data, test_data):
		# 使用Dataloader生成批次数据
		train_data = train_data.map(self.tokenize_function, batched=True, num_proc=6)
		valid_data = valid_data.map(self.tokenize_function, batched=True, num_proc=6)
		test_data = test_data.map(self.tokenize_function, batched=True, num_proc=6)

		train_data.set_format(type='torch', columns=['input_ids', 'labels'])
		valid_data.set_format(type='torch', columns=['input_ids', 'labels'])
		test_data.set_format(type='torch', columns=['input_ids', 'labels'])

		train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=6)
		valid_dataloader = DataLoader(valid_data, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=6)
		test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=6)

		print('Data Initializing done')
		
		return train_dataloader, valid_dataloader, test_dataloader