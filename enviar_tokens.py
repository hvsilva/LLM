from transformers import AutoTokenizer
import requests

tokenizer = AutoTokenizer.from_pretrained("onnx/gpt2")
entrada = "Programa Java"
tokens = tokenizer.encode(entrada)

print("Tokens gerados:", tokens)

res = requests.post("http://localhost:8080/onnx-transformer/prever", json={"tokens": tokens})
print("Resposta da API:")
print(res.json())
