import numpy as np
import migraphx
import json
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="./migraphx_model/model.mxr")
parser.add_argument("--config_path", type=str, default="./migraphx_model/config.json")

TYPE_MAP = {
    "bool_type": np.bool,
    "uint8_type": np.uint8,
    "int8_type": np.int8,
    "int16_type": np.int16,
    "int32_type": np.int32,
    "int64_type": np.int64,
    "float_type": np.float32,
    "double_type": np.float64,
    "half_type": np.float16,
}

def load_model(model_path):
    return migraphx.load(model_path)


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

# TRT demo repeats input_ids instead of padding
def generate_input_ids(batch_size, sequence_length):
    real_input_ids = [151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
        151645,    198, 151644,    872,    198,  58104,    304,  10200,  39507,
          9625,     11,   2055,  26161,  16176,    438,    264, 151645,    198,
        151644,  77091,    198]
    real_input_ids = np.array(real_input_ids, dtype=np.int64)
    real_size = real_input_ids.size

    # repeat the real_input_ids to the sequence_length
    # host_input_ids = np.repeat(real_input_ids, sequence_length // len(real_input_ids))
    host_input_ids = np.empty(sequence_length, dtype=np.int64)
    for i in range(sequence_length):
        host_input_ids[i] = real_input_ids[i % real_size]
    host_input_ids = host_input_ids.reshape(1, sequence_length)
    host_input_ids = host_input_ids.repeat(batch_size, axis=0)
    return migraphx.to_gpu(migraphx.argument(host_input_ids))


# trt is using position_ids, not sure what the equivalent mask is 
def generate_attention_mask(batch_size, sequence_length):
    attention_mask = np.ones((batch_size, sequence_length), dtype=np.int64)
    return migraphx.to_gpu(migraphx.argument(attention_mask))

def generate_past_key_values(batch_size, past_sequence_length, head_size):
    past_key_values = np.zeros((batch_size, 2, past_sequence_length, head_size), dtype=np.float16)
    return migraphx.to_gpu(migraphx.argument(past_key_values))

def allocate_logits_output(batch_size, sequence_length, vocab_size):
    return migraphx.allocate_gpu(migraphx.shape(type="half_type", lens=[batch_size, sequence_length, vocab_size]))


def generate_inputs(input_names, config):
    batch_size = config["batch_size"]
    sequence_length = config["sequence_length"]
    past_sequence_length = config["past_sequence_length"]
    head_size = config["head_size"]
    vocab_size = config["vocab_size"]
    inputs = {}
    for name in input_names:
        if name == "input_ids":
            inputs[name] = generate_input_ids(batch_size, sequence_length)
        elif name == "attention_mask":
            inputs[name] = generate_attention_mask(batch_size, sequence_length)
        elif "past_key_values" in name:
            inputs[name] = generate_past_key_values(batch_size, past_sequence_length, head_size)
        elif "#output_" in name:
            inputs[name] = allocate_logits_output(batch_size, sequence_length, vocab_size)
        else:
            raise ValueError(f"Unexpected input name: {name}")
    return inputs

def run_model(model, inputs, stream):
    outs = model.run_async(inputs, stream, "ihipStream_t")
    return outs


if __name__ == "__main__":
    args = parser.parse_args()
    model_path = args.model_path
    config_path = args.config_path

    print("Loading model...")
    model = load_model(model_path)
    input_names = model.get_parameter_names()

    print("Loading config...")
    config = load_config(config_path)
    
    print("Generating inputs...")
    inputs = generate_inputs(input_names, config)

    print("Running model...")
    curr_stream = torch.cuda.current_stream()
    outputs = run_model(model, inputs, curr_stream.cuda_stream)

    print("Synchronizing...")
    curr_stream.synchronize()
