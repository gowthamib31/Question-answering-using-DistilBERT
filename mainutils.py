
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
import torch

model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def answer_question(question, context):
    inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
    start_logits, end_logits = model(*inputs).start_logits, model(*inputs).end_logits
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits)
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index+1]))
    return answer