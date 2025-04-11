import json
import argparse

def convert_dataset(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    with open(output_path, "w", encoding="utf-8") as f_out:
        for item in dataset:
            instruction = item.get("instruction", "").strip()
            context = item.get("context", "").strip()
            target_code = item.get("target_code", "").strip()

            prompt = f"Instruction: {instruction}\nContext:\n{context}"
            completion = target_code

            record = {
                "prompt": prompt,
                "completion": completion
            }
            f_out.write(json.dumps(record) + "\n")

    print(f"Converted {len(dataset)} samples to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input dataset.json")
    parser.add_argument("--output", type=str, required=True, help="Path to output .jsonl file")
    args = parser.parse_args()

    convert_dataset(args.input, args.output)
