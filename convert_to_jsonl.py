import json
import argparse

# based on annotated_dataset/dataset/new_tasks/dataset.json to modify
def convert_dataset(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f_out:
        for entry in dataset:
            for turn in entry.get("turns", []):
                turn_data = turn.get("turn", {})
                instruction = turn_data.get("intent", {}).get("value", "").strip()
                code = turn_data.get("code", {}).get("value", "").strip()

                if not instruction or not code:
                    continue  # skip empty entries

                prompt = f"Instruction: {instruction}\nContext:"
                completion = code

                record = {
                    "prompt": prompt,
                    "completion": completion
                }
                f_out.write(json.dumps(record) + "\n")
                count += 1

    print(f"Converted {count} samples to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input dataset.json")
    parser.add_argument("--output", type=str, required=True, help="Path to output .jsonl file")
    args = parser.parse_args()

    convert_dataset(args.input, args.output)
