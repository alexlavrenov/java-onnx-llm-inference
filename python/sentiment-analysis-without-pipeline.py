from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch

# we are using same default model from pipeline example
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")

# uncomment to see input to model which we prepared using tokenizer
# print(inputs)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)

# uncomment to see model output
# print(outputs)

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
for input_sentense, prediction in zip(raw_inputs, predictions):
    print(input_sentense, prediction)

print(model.config.id2label)
