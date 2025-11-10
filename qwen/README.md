
# Qwen2.5 - MIGraphX Inference Demo


---

## Contents

- [`export.py`](export.py): Download a Qwen ONNX model, convert to MIGraphX, and generate config.
- [`mgx_inference.py`](mgx_inference.py): Run inference using the MIGraphX backend (GPU, ROCm/HIP).
- [`torch_inference.py`](torch_inference.py): Pure PyTorch inference (for comparison).

---

## Quickstart

### 1. Export Model to MIGraphX

```bash
python export.py \
    --output_dir ./migraphx_model \
    --past_seq_length 2048 \
    --seq_length 1500 \
    --batch_size 1 \
    --repo_id AlanTurner/Qwen2.5-0.5B-Instruct-ONNX \
    --cache_dir ./hf_cache
```

This will:
- Download ONNX model checkpoint & config (`model.onnx`, `genai_config.json`) from Hugging Face Hub.
- Convert to MIGraphX IR (`model.mxr`) and write a config file (`config.json`).
- Place outputs in `./migraphx_model/`

---

### 2. Inference with MIGraphX (`mgx_inference.py`)

```bash
python mgx_inference.py \
    --model_path ./migraphx_model/model.mxr \
    --config_path ./migraphx_model/config.json
```

This will:
- Load the MIGraphX model and config.
- Prepare demonstration input tensors.
- Run inference asynchronously on the current HIP (GPU) stream.
- Synchronize and exit.

All device handling and tensor allocations (input_ids, mask, past_key_values, etc) are handled internally.

**See `mgx_inference.py` for how to interact with the MIGraphX Python APIs.**

---


## Notes

- **Requirements:**  
  - [MIGraphX](https://github.com/ROCm/migraphx) Python package.
  - [Huggingface Hub](https://pypi.org/project/huggingface-hub/) for downloading ONNX models.
  - PyTorch (for example usages and as a comparison).

- **ONNX Model**:  
  This repo defaults to Qwen2.5-0.5B-Instruct-ONNX: [https://huggingface.co/AlanTurner/Qwen2.5-0.5B-Instruct-ONNX](https://huggingface.co/AlanTurner/Qwen2.5-0.5B-Instruct-ONNX)

- See each script for further customization or to adapt for batch inference, sequence generation, etc.

---


