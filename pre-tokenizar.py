from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

entrada = "hello world"
tokens = tokenizer.encode(entrada, return_tensors="pt")[0].tolist()
print(tokens)  # exemplo: [101, 7592, 2088, 102]
