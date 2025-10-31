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
    parser = argparse.ArgumentParser(description="Inspect Qwen3-VL-2B-Instruct image preprocessing")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to Qwen3-VL-2B-Instruct folder")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to MMIE dataset folder")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of conversations to process")
    parser.add_argument("--output", type=str, default="tokens_output.json", help="Output JSON file for token IDs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {args.model_dir} ...")
    processor = AutoProcessor.from_pretrained(args.model_dir)

    # ---- Parse image preprocessing config ----
    image_processor = processor.image_processor
    patch_size = getattr(image_processor, "patch_size", 14)
    merge_size = getattr(image_processor, "merge_size", 1)

    # Directly parse min/max pixel scaling constraints
    min_pixels = image_processor.size.get("min_pixels", image_processor.size.get("shortest_edge", None))
    max_pixels = image_processor.size.get("max_pixels", image_processor.size.get("longest_edge", None))

    print(f"Image patch size: {patch_size}, merge size: {merge_size}")
    print(f"min_pixels = {min_pixels}, max_pixels = {max_pixels}")

    samples = load_human_conversations(args.data_dir, args.num_samples)
    print(f"Loaded {len(samples)} human messages from {args.num_samples} conversations")

    images_kwargs = {
        "patch_size": getattr(image_processor, "patch_size", 16),
        "merge_size": getattr(image_processor, "merge_size", 1),
        "min_pixels": getattr(image_processor.size, "get", lambda x, d=None: None)("min_pixels", None)
                    or image_processor.size.get("shortest_edge", 448),
        "max_pixels": getattr(image_processor.size, "get", lambda x, d=None: None)("max_pixels", None)
                    or image_processor.size.get("longest_edge", 1344),
    }

    print(images_kwargs)

    results = []
    for i, (image_path, text) in enumerate(samples, start=1):

        if image_path and os.path.exists(image_path):
            print(f"\n[{i}/{len(samples)}] Processing image: {os.path.basename(image_path)}")
            image = Image.open(image_path).convert("RGB")
            prompt_text = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{text}\n<|im_end|>\n<|im_start|>assistant\n"

            # Feed image + text through processor
            inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(device)

            # --- Display image dimensions ---
            print(f"Original image size (WxH): {image.size}")
            pixel_values = inputs['pixel_values']
            print(f"Processed image tensor shape: {pixel_values.shape}")  # [1, 3, H, W]

            # --- Compute exact number of image pad tokens ---
            num_patches_orig = image_processor.get_number_of_image_patches(*image.size, images_kwargs)
            print(f"Number of image pad tokens (original): {num_patches_orig}")

            input_token_ids = inputs["input_ids"][0].tolist()

            count_151655 = input_token_ids.count(151655)
            print(f"Number of 151655 tokens: {count_151655}")

            results.append({
                "image": os.path.basename(image_path),
                "text": text,
                "original_size": list(image.size),
                "processed_shape": list(pixel_values.shape),
                "num_patches_orig": num_patches_orig,
                "count_pad": count_151655
            })

        else:
            print(f"[{i}/{len(samples)}] Text-only message")
            results.append({"image": None, "text": text})

    # ---- Save results ----
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved token and resize information to {args.output}")


if __name__ == "__main__":
    main()
