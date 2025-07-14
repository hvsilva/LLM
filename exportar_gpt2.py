from transformers import AutoTokenizer
from optimum.exporters.onnx import main_export
from pathlib import Path

main_export(
    model_name_or_path="gpt2",
    task="text-generation",  # ou 'causal-lm'
    output=Path("onnx/gpt2")
)

# Salvar tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.save_pretrained("onnx/gpt2")
