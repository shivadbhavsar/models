import os
from huggingface_hub import hf_hub_download
import json
import migraphx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="./migraphx_model")
parser.add_argument("--past_seq_length", type=int, default=2048)
parser.add_argument("--seq_length", type=int, default=1500)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--repo_id", type=str, default="AlanTurner/Qwen2.5-0.5B-Instruct-ONNX")
parser.add_argument("--cache_dir", type=str, default="./hf_cache")

if __name__ == "__main__":
    args = parser.parse_args()

    repo_id = args.repo_id
    cache_dir = args.cache_dir

    onnx_path = hf_hub_download(repo_id=args.repo_id, filename="model.onnx", cache_dir=cache_dir)
    onnx_data_path = hf_hub_download(repo_id=args.repo_id, filename="model.onnx.data", cache_dir=cache_dir)
    config_path = hf_hub_download(repo_id=args.repo_id, filename="genai_config.json", cache_dir=cache_dir)

    genai_config = json.load(open(config_path, "r"))

    head_size = genai_config["model"]["decoder"]["head_size"]
    required_configs = ["bos_token_id", "eos_token_id", "pad_token_id", "context_length", "vocab_size"]

    config = {}
    for c in required_configs:
        config[c] = genai_config["model"][c]

    ## bin/driver perf <path> --fill1 input_ids 
    # --fill1 attention_mask 
    # --dim-param "@past_sequence_length" "2048" 
    # --input-dim @input_ids 1 2000 
    # --input-dim @attention_mask 1 2000

    output_dir = args.output_dir
    past_seq_length = args.past_seq_length
    seq_length = args.seq_length
    batch_size = args.batch_size
    input_ids_shape = [batch_size, seq_length]
    attention_mask_shape = [batch_size, seq_length]

    past_sequence_length = migraphx.shape.dynamic_dimension(past_seq_length, past_seq_length)

    mgx_model = migraphx.parse_onnx(onnx_path, dim_params={"past_sequence_length": past_sequence_length}, map_input_dims={"input_ids": input_ids_shape, "attention_mask": attention_mask_shape})

    # # print(mgx_model.get_parameter_names())
    # # print(mgx_model.get_parameter_shapes())

    mgx_model.compile(migraphx.get_target("gpu"), offload_copy=False)

    os.makedirs(output_dir, exist_ok=True)
    migraphx.save(mgx_model, os.path.join(output_dir, "model.mxr"))

    config["head_size"] = head_size
    config["past_sequence_length"] = past_seq_length
    config["sequence_length"] = seq_length
    config["batch_size"] = batch_size

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f)

    print(f"Model saved to {output_dir}")
