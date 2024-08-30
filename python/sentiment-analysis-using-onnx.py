from transformers import AutoTokenizer
import torch
import onnxruntime as ort

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

onnx_model_path = "../.onnx/distilbert-base-uncased-finetuned-sst-2-english/model.onnx"
session = ort.InferenceSession(onnx_model_path)

# Prepare the inputs for ONNX Runtime
onnx_inputs = {
    "input_ids": inputs["input_ids"].numpy(),
    "attention_mask": inputs["attention_mask"].numpy()
}

# Run the model
outputs = session.run(None, onnx_inputs)

# uncomment to see raw results
# print(outputs)

logits_tensor = torch.tensor(outputs[0], dtype=torch.float32)

predictions = torch.nn.functional.softmax(logits_tensor, dim=1)
for input_sentense, prediction in zip(raw_inputs, predictions):
    print(input_sentense, prediction)

