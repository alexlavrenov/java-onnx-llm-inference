import json
import os

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AutoConfig
import torch

# Load the pre-trained model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model_dir = os.path.normpath("../.onnx/" + model_name)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
onnx_model_path = os.path.join(model_dir, "model.onnx")

tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

# Dummy input for the model (batch size 1, sequence length 8)
dummy_input = tokenizer.encode_plus(
    "This is a sample input for ONNX export",
    return_tensors="pt"
)

# Define the input and output names
input_names = ["input_ids", "attention_mask"]
output_names = ["output"]

# Convert the model to ONNX
onnx_model_path = os.path.join(model_dir, "model.onnx")
torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    onnx_model_path,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}, "attention_mask": {0: "batch_size", 1: "sequence_length"}},
    opset_version=11
)

# Export the configuration
config_dict = config.to_dict()
with open(os.path.join(model_dir, "config.json"), "w") as f:
    json.dump(config_dict, f, indent=2)

print("Configuration exported to JSON format.")

# Export the tokenizer
tokenizer.save_pretrained(model_dir)
print("Tokenizer exported.")