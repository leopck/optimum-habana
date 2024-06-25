#!/usr/bin/env python
# coding=utf-8

import time
import torch
import logging
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset
from transformers import LiltForTokenClassification, LayoutLMv3FeatureExtractor, AutoTokenizer
from optimum.habana import GaudiConfig
import habana_frameworks.torch.core as htcore

logger = logging.getLogger(__name__)

def set_device(device_type):
    if device_type == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_type == "hpu":
        device = torch.device("hpu")
    else:
        device = torch.device("cpu")
    return device

def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]

label2color = {
    "B-HEADER": "blue",
    "B-QUESTION": "red",
    "B-ANSWER": "green",
    "I-HEADER": "blue",
    "I-QUESTION": "red",
    "I-ANSWER": "green",
}

def draw_boxes(image, boxes, predictions):
    width, height = image.size
    normalized_boxes = [unnormalize_box(box, width, height) for box in boxes]

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for prediction, box in zip(predictions, normalized_boxes):
        if prediction == "O":
            continue
        draw.rectangle(box, outline="black")
        draw.rectangle(box, outline=label2color[prediction])
        draw.text((box[0] + 10, box[1] - 10), text=prediction, fill=label2color[prediction], font=font)
    return image

def run_inference(images, model, feature_extractor, tokenizer, device, output_image=True, warm_up_steps=5):
    results = []
    total_inference_time = 0

    # Warm-up phase
    for _ in range(warm_up_steps):
        image = images[0]
        feature_extraction = feature_extractor(images=image, return_tensors="pt")
        words = feature_extraction["words"][0]
        boxes = feature_extraction["boxes"][0]
        encoding = tokenizer(text=words, boxes=boxes, return_tensors="pt", padding="max_length", truncation=True)
        encoding = {k: v.to(device) for k, v in encoding.items()}
        model.to(device)
        if device.type == "hpu":
            htcore.mark_step()
        outputs = model(**encoding)
        if device.type == "hpu":
            htcore.mark_step()

    # Main inference loop with performance measurement
    for image in images:
        start_time = time.time()

        feature_extraction_start = time.time()
        feature_extraction = feature_extractor(images=image, return_tensors="pt")
        feature_extraction_end = time.time()

        words = feature_extraction["words"][0]
        boxes = feature_extraction["boxes"][0]

        tokenization_start = time.time()
        encoding = tokenizer(text=words, boxes=boxes, return_tensors="pt", padding="max_length", truncation=True)
        tokenization_end = time.time()

        encoding = {k: v.to(device) for k, v in encoding.items()}
        model.to(device)

        if device.type == "hpu":
            htcore.mark_step()

        inference_start = time.time()
        outputs = model(**encoding)

        if device.type == "hpu":
            htcore.mark_step()

        inference_end = time.time()

        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        labels = [model.config.id2label[prediction] for prediction in predictions]

        unique_boxes = []
        unique_labels = []
        for box, label in zip(encoding["bbox"][0], labels):
            if box.tolist() not in unique_boxes:
                unique_boxes.append(box.tolist())
                unique_labels.append(label)

        if output_image:
            results.append(draw_boxes(image, unique_boxes, unique_labels))
        else:
            results.append(unique_labels)

        end_time = time.time()
        total_inference_time += end_time - start_time

        logger.info(f"Inference time for this image: {end_time - start_time:.2f} seconds")
        logger.info(f"  Feature extraction time: {feature_extraction_end - feature_extraction_start:.2f} seconds")
        logger.info(f"  Tokenization time: {tokenization_end - tokenization_start:.2f} seconds")
        logger.info(f"  Model inference time: {inference_end - inference_start:.2f} seconds")

    avg_inference_time = total_inference_time / len(images)
    logger.info(f"Average inference time per image: {avg_inference_time:.2f} seconds")

    return results

def main():
    device_type = "hpu"
    device = set_device(device_type)

    dataset_id = "nielsr/funsd-layoutlmv3"
    dataset = load_dataset(dataset_id)

    logger.info(f"Train dataset size: {len(dataset['train'])}")
    logger.info(f"Test dataset size: {len(dataset['test'])}")

    model = LiltForTokenClassification.from_pretrained("./results/")
    feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=True)
    tokenizer = AutoTokenizer.from_pretrained("./results")

    test_images = [dataset["test"][i]["image"].convert("RGB") for i in range(20)]
    result_images = run_inference(test_images, model, feature_extractor, tokenizer, device)

    for idx, result_image in enumerate(result_images):
        result_image.save(f"result_image_{idx}.png")


if __name__ == "__main__":
    main()
