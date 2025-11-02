import argparse
import json
import os
from datetime import datetime

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default=None)
    p.add_argument("--examples-file", required=True)
    p.add_argument("--inputs", nargs="*", default=["Hello world"])
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--config", default=os.path.join("configs", "default.yaml"))
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)

    model_id = args.model_id or cfg.get("model_id", "Qwen/Qwen2.5-0.5B-Instruct")
    max_new_tokens = args.max_new_tokens or int(cfg.get("max_new_tokens", 256))
    temperature = args.temperature if args.temperature is not None else float(cfg.get("temperature", 0.2))
    trust_remote_code = bool(cfg.get("trust_remote_code", True))

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.examples_file, "r", encoding="utf-8") as f:
        examples_text = f.read()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        device_map="auto",
    )

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(args.output_dir, f"few_shot_{ts}.jsonl")
    with open(out_path, "w", encoding="utf-8") as out:
        for user_input in args.inputs:
            prompt = build_prompt(examples_text, user_input)
            out_text = gen(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )[0]["generated_text"][len(prompt):]
            rec = {
                "input": user_input,
                "output": out_text.strip(),
                "model_id": model_id,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "timestamp": ts,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(json.dumps(rec, ensure_ascii=False))

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
