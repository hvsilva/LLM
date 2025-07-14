from transformers import AutoTokenizer, AutoModel
from optimum.exporters.onnx import main_export
from pathlib import Path

# Modelo a exportar
model_name = "bert-base-uncased"

# Exporta o modelo ONNX
main_export(
    model_name_or_path=model_name,
    task="feature-extraction",  # ou 'text-classification'
    output=Path("onnx/bert-base")
)

# Salva o tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained("onnx/bert-base")
