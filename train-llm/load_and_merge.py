import argparse
import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel, PeftConfig


def load_model_with_lora(model_name, lora_path, quantized, use_cpu):
    device_map = 'cpu' if use_cpu else 'auto'

    if quantized and not use_cpu:
        # Enable quantization only if running on GPU
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype='float16',
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if not use_cpu else torch.float32,
            device_map=device_map,
            offload_folder='./offload',
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    lora_config = PeftConfig.from_pretrained(lora_path)
    lora_model = PeftModel.from_pretrained(
        model, lora_path, config=lora_config
    )
    lora_model = lora_model.merge_and_unload()

    return lora_model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description='Load and optionally quantize a model with LoRA'
    )
    parser.add_argument(
        'model_name', type=str, help='Path or name of the base model'
    )
    parser.add_argument(
        'lora_path', type=str, help='Path or name of the LoRA model'
    )
    parser.add_argument(
        '--quantized',
        action='store_true',
        help='Enable 4-bit quantization for the model (GPU only)',
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force model to run on CPU instead of GPU',
    )
    parser.add_argument(
        '--push_to_hub',
        type=str,
        help='Repository name to push to Hugging Face Hub',
    )
    parser.add_argument(
        '--save_to_disk',
        type=str,
        help='Directory path to save the merged model locally',
    )
    args = parser.parse_args()

    # Load model and tokenizer with optional quantization and device choice
    model, tokenizer = load_model_with_lora(
        args.model_name, args.lora_path, args.quantized, args.cpu
    )

    print('Model and tokenizer loaded successfully.')

    if args.save_to_disk:
        os.makedirs(args.save_to_disk, exist_ok=True)
        model.save_pretrained(args.save_to_disk)
        tokenizer.save_pretrained(args.save_to_disk)
        print(f'Model and tokenizer saved to {args.save_to_disk}')

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        print(
            f"Model and tokenizer pushed to Hugging Face Hub repository '{args.push_to_hub}'"
        )


if __name__ == '__main__':
    main()
