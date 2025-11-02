import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--output-dir", default="outputs/lora")
    p.add_argument("--train-file", default="data/train.jsonl")
    p.add_argument("--val-file", default="data/val.jsonl")
    args = p.parse_args()

    # TODO: Implement LoRA training with peft + accelerate
    # Suggested steps:
    # 1) Load model/tokenizer
    # 2) Prepare supervised dataset (input -> boxed output)
    # 3) Configure LoRA (target modules, r, alpha, dropout)
    # 4) Train & eval; save adapter weights to output-dir
    print("LoRA training scaffold ready. Fill in implementation.")
    print(f"model-id={args.model_id}")
    print(f"output-dir={args.output_dir}")
    print(f"train-file={args.train_file}")
    print(f"val-file={args.val_file}")


if __name__ == "__main__":
    main()
