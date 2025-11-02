import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def load_config(cfg_path: str):
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def build_prompt(examples_text: str, user_input: str) -> str:
    return (
        examples_text.strip()
        + "\n\nNow format the following input in the same way."
        + "\nInput:\n"
        + user_input.strip()
        + "\nOutput:\n"
    )


def save_to_json(data: Dict[str, Any], output_path: str) -> None:
    """Save data to a JSON file with pretty printing and ensure directory exists."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_output_dir() -> Path:
    """Get the output directory, creating it if it doesn't exist."""
    # Use root-level outputs/experiment_name/ directory
    root_dir = Path(__file__).parent.parent.parent.parent  # Go up to project root
    output_dir = root_dir / "outputs" / "experiment_ascii_box"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    p = argparse.ArgumentParser(description="Run few-shot text generation with a language model.")
    p.add_argument("--model-id", default=None, help="Hugging Face model ID")
    p.add_argument("--examples-file", required=True, help="Path to file containing few-shot examples")
    p.add_argument("--inputs", nargs="*", default=["Hello world"], help="Input texts to process")
    p.add_argument("--output-dir", type=str, help="Override default output directory")
    p.add_argument("--config", default=os.path.join("configs", "default.yaml"), 
                  help="Path to config file")
    p.add_argument("--max-new-tokens", type=int, default=None, 
                  help="Maximum number of new tokens to generate")
    p.add_argument("--temperature", type=float, default=None, 
                  help="Sampling temperature (higher = more random)")
    args = p.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    model_id = args.model_id or cfg.get("model_id", "Qwen/Qwen2.5-0.5B-Instruct")
    max_new_tokens = args.max_new_tokens or int(cfg.get("max_new_tokens", 256))
    temperature = args.temperature if args.temperature is not None else float(cfg.get("temperature", 0.2))
    trust_remote_code = bool(cfg.get("trust_remote_code", True))

    # Prepare output structure
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_data = {
        "metadata": {
            "model_id": model_id,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "timestamp": timestamp,
            "config_file": os.path.basename(args.config)
        },
        "examples": []
    }

    # Load examples
    with open(args.examples_file, "r", encoding="utf-8") as f:
        examples_text = f.read()

    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        device_map="auto",
    )

    # Initialize generation pipeline
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # Process inputs
    for user_input in args.inputs:
        prompt = build_prompt(examples_text, user_input)
        out_text = gen(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )[0]["generated_text"][len(prompt):]

        output_data["examples"].append({
            "input": user_input,
            "prompt": prompt,
            "output": out_text.strip(),
            "timestamp": timestamp
        })
        
        # Print progress
        print(f"Processed input: {user_input}")

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a timestamped subdirectory for this run
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    # Save outputs and copy config
    output_path = run_dir / "results.json"
    
    save_to_json(output_data, output_path)
    print(f"\nResults saved to: {output_path.absolute()}")
    print(f"Total examples processed: {len(output_data['examples'])}")


if __name__ == "__main__":
    main()
