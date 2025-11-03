import argparse
import json
import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


def load_human_conversations(data_dir, num_samples):
    """Load first N conversations, collecting all 'from': 'human' entries (with or without images)."""
    data_json = os.path.join(data_dir, "data.json")
    img_dir = os.path.join(data_dir, "images_2")

    with open(data_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    conversations = data[:num_samples]
    items = []

    for conv_entry in conversations:
        convs = conv_entry.get("conversations", [])
        for conv in convs:
            if conv.get("from") == "human":
                for content in conv.get("content", []):
                    img_path = content.get("image")
                    try:
                        text = content.get("text", "").strip()
                        full_path = os.path.join(img_dir, img_path) if img_path else None
                        items.append((full_path, text))
                    except:
                        pass
    return items


def main():
    parser = argparse.ArgumentParser(description="Generate token IDs from Qwen3-VL-2B-Instruct (human-only inference)")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to Qwen3-VL-2B-Instruct folder")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to MMIE dataset folder")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of conversations to process")
    parser.add_argument("--output", type=str, default="tokens_output.json", help="Output JSON file for token IDs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {args.model_dir} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir,
        dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(args.model_dir)

    samples = load_human_conversations(args.data_dir, args.num_samples)
    print(f"Loaded {len(samples)} human messages from {args.num_samples} conversations")

    results = []
    for i, (image_path, text) in enumerate(samples, start=1):
        print(f"[{i}/{len(samples)}] Processing {'(text-only)' if image_path is None else os.path.basename(image_path or '')}")

        # Construct prompt and inputs
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            prompt_text = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{text}\n<|im_end|>\n<|im_start|>assistant\n"
            inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(device)
        else:
            prompt_text = f"<|im_start|>user\n{text}\n<|im_end|>\n<|im_start|>assistant\n"
            inputs = processor(text=prompt_text, return_tensors="pt").to(device)

        input_token_ids = inputs["input_ids"][0].tolist()
        num_img_pad_input = input_token_ids.count(151655)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                num_beams=1
            )

        # ---- Clean output tokens (remove special tokens) ----
        output_token_ids_full = generated[0].tolist()
        generated_only = generated[0][inputs["input_ids"].shape[-1]:]

        # remove special tokens from generated-only part
        special_tokens = set(processor.tokenizer.all_special_ids)
        output_token_ids_clean = [tid for tid in generated_only.tolist() if tid not in special_tokens]

        decoded_text = processor.batch_decode(
            torch.tensor(output_token_ids_clean).unsqueeze(0),
            skip_special_tokens=True
        )[0]

        results.append({
            "image": os.path.basename(image_path) if image_path else None,
            "input_text": text,
            "prompt_text": prompt_text,
            "num_img_pad_input": num_img_pad_input,
            "input_token_ids": input_token_ids,
            "output_token_ids": output_token_ids_clean,  # cleaned version
            "output_text": decoded_text,
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved input/output token IDs and decoded text (without prompt) to {args.output}")


if __name__ == "__main__":
    main()
