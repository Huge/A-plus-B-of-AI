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


def build_prompt(examples_text: str, user_input: str, end_sentinel: str) -> str:
    return (
        "Instruction: For each word in the input, output that word in a simple ASCII box. Output ONLY the boxes. No explanations. End your output with '" + end_sentinel + "'.\n\n"
        + examples_text.strip()
        + "\n\nInput:\n"
        + user_input.strip()
        + "\nOutput:\n"
    )


def save_to_json(data: Dict[str, Any], output_path: str) -> None:
    """Save data to a JSON file with pretty printing and ensure directory exists."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_boxes(text: str) -> str:
    lines = [ln.rstrip() for ln in text.splitlines()]
    out_lines: List[str] = []
    i = 0
    def is_border(s: str) -> bool:
        return len(s) >= 3 and s[0] == '+' and s[-1] == '+' and all(ch == '-' for ch in s[1:-1])
    def is_middle(s: str) -> bool:
        return len(s) >= 2 and s[0] == '|' and s[-1] == '|'
    while i + 2 < len(lines):
        a, b, c = lines[i], lines[i+1], lines[i+2]
        if is_border(a) and is_middle(b) and is_border(c) and len(a) == len(c) == len(b):
            out_lines.extend([a, b, c])
            i += 3
        else:
            i += 1
    return "\n".join(out_lines).strip()


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
    p.add_argument("--end-sentinel", type=str, default="<<END>>",
                  help="String that marks the end of desired output")
    args = p.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    model_id = args.model_id or cfg.get("model_id", "Qwen/Qwen2.5-0.5B")
    max_new_tokens = args.max_new_tokens or int(cfg.get("max_new_tokens", 128))
    temperature = args.temperature if args.temperature is not None else float(cfg.get("temperature", 0.0))
    trust_remote_code = bool(cfg.get("trust_remote_code", True))

    # Prepare output structure
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_data = {
        "metadata": {
            "model_id": model_id,
            "max_new_tokens": max_new_tokens,
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
    end_sentinel = args.end_sentinel
    for user_input in args.inputs:
        prompt = build_prompt(examples_text, user_input, end_sentinel)
        out_text = gen(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )[0]["generated_text"][len(prompt):]

        # Truncate at end sentinel if present
        cut_idx = out_text.find(end_sentinel)
        clean_out = (out_text[:cut_idx] if cut_idx != -1 else out_text).strip()
        boxed_only = extract_boxes(clean_out)

        output_data["examples"].append({
            "input": user_input,
            "prompt": prompt,
            "output": boxed_only or clean_out,
            "raw_output": out_text.strip(),
            "end_sentinel_found": cut_idx != -1,
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
