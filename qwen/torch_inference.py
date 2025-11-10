#!/usr/bin/env python3
"""
Script to load and run Qwen2.5 0.5B model
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name="Qwen/Qwen2.5-0.5B", device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load Qwen2.5 model and tokenizer
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        model, tokenizer
    """
    print(f"Loading model: {model_name}")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_p=0.9, top_k=50):
    """
    Generate text from a prompt
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input text prompt
        max_length: Maximum length of generated text
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
    
    Returns:
        Generated text
    """
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and return
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def chat_completion(model, tokenizer, messages, max_length=512, temperature=0.7):
    """
    Chat completion using Qwen's chat format
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        messages: List of message dicts with 'role' and 'content'
                  Example: [{"role": "user", "content": "Hello!"}]
        max_length: Maximum length of generated text
        temperature: Sampling temperature
    
    Returns:
        Assistant's response
    """
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response


def main():
    """Main function demonstrating usage"""
    
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    print("\n" + "="*80)
    print("Example 1: Simple Text Generation")
    print("="*80)
    prompt = "Once upon a time in a distant galaxy"
    print(f"Prompt: {prompt}")
    generated = generate_text(model, tokenizer, prompt, max_length=100)
    print(f"\nGenerated:\n{generated}")
    
    # print("\n" + "="*80)
    # print("Example 2: Chat Completion")
    # print("="*80)
    # messages = [
    #     {"role": "system", "content": "You are a helpful AI assistant."},
    #     {"role": "user", "content": "What are the key features of Python programming language?"}
    # ]
    # print(f"User: {messages[1]['content']}")
    # response = chat_completion(model, tokenizer, messages, max_length=200)
    # print(f"\nAssistant: {response}")
    
    # print("\n" + "="*80)
    # print("Example 3: Multi-turn Conversation")
    # print("="*80)
    # messages = [
    #     {"role": "system", "content": "You are a helpful AI assistant."},
    #     {"role": "user", "content": "Tell me about machine learning."},
    # ]
    # response1 = chat_completion(model, tokenizer, messages, max_length=150)
    # print(f"User: {messages[1]['content']}")
    # print(f"Assistant: {response1}")
    
    # # Add assistant response and new user message
    # messages.append({"role": "assistant", "content": response1})
    # messages.append({"role": "user", "content": "Can you give me a simple example?"})
    
    # response2 = chat_completion(model, tokenizer, messages, max_length=150)
    # print(f"\nUser: {messages[3]['content']}")
    # print(f"Assistant: {response2}")
    
    # print("\n" + "="*80)
    # print("Example 4: Interactive Chat (enter 'quit' to exit)")
    # print("="*80)
    
    # conversation = [
    #     {"role": "system", "content": "You are a helpful AI assistant."}
    # ]
    
    # while True:
    #     user_input = input("\nYou: ").strip()
    #     if user_input.lower() in ['quit', 'exit', 'q']:
    #         print("Goodbye!")
    #         break
        
    #     if not user_input:
    #         continue
        
    #     conversation.append({"role": "user", "content": user_input})
    #     response = chat_completion(model, tokenizer, conversation, max_length=200, temperature=0.8)
    #     conversation.append({"role": "assistant", "content": response})
        
    #     print(f"Assistant: {response}")


if __name__ == "__main__":
    main()
