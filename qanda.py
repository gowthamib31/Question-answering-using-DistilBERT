import torch
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
squad_dataset = load_dataset("squad")
nltk.download('punkt')
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch
import numpy as np
import ipywidgets as widgets



