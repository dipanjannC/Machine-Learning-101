import torch
import tensorflow as tf
import tensorflow_hub as hub
import torch
import transformers
from transformers import *
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from transformers import BartTokenizer, BartForConditionalGeneration

torch_device = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'

print("Downloading bert-large-uncased-whole-word-masking-finetuned-squad pre-trained model")
QA_MODEL = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
QA_TOKENIZER = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
QA_MODEL.to(torch_device)
QA_MODEL.eval()

print("Downloading facebook bart-large-cnn model")
SUMMARY_TOKENIZER = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
SUMMARY_MODEL = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
SUMMARY_MODEL.to(torch_device)
SUMMARY_MODEL.eval()

print("Downloading biobert_v1.1_pubmed pre-trained model")
para_model = AutoModel.from_pretrained('monologg/biobert_v1.1_pubmed')
para_tokenizer = AutoTokenizer.from_pretrained('monologg/biobert_v1.1_pubmed', do_lower_case=False)

print("Done. All pre-trained models loaded into docker image")
