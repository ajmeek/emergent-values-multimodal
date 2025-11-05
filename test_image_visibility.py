#!/usr/bin/env python3
"""
Test script to verify the model can actually see images.
Tests multiple approaches to passing images to Qwen2.5-VL with vLLM.

Run with: srun -p cais --gres=gpu:4 --mem=128G python test_image_visibility.py
"""

import os
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# Configuration
MODEL_PATH = "/data/huggingface/Qwen/Qwen2.5-VL-72B-Instruct"
TEST_IMAGE_PATH = "/data/superstimuli_group/all_superstimuli/2025_10_31 jitter0_seed20 (1).png"


def test_approach_1_image_in_content(llm, tokenizer, img):
    """
    Approach 1: Include image directly in content dict
    (Most explicit approach - recommended for vLLM + Qwen2.5-VL)
    """
    print("\n" + "="*80)
    print("APPROACH 1: Image in message content dict")
    print("="*80)
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Can you see an image in this message? If yes, please describe what you see in detail. If no, just say 'I cannot see any image.'"},
            {"type": "image", "image": img}
        ]
    }]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print(f"Prompt length: {len(prompt)} chars")
    print(f"Prompt preview (first 500 chars):\n{prompt[:500]}")
    print("\nGenerating response...")
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
    
    # Try with multi_modal_data
    request = {"prompt": prompt, "multi_modal_data": {"image": [img]}}
    output = llm.generate([request], sampling_params)[0]
    response = output.outputs[0].text
    
    print("\n" + "-"*80)
    print("MODEL RESPONSE:")
    print("-"*80)
    print(response)
    print("-"*80 + "\n")
    
    return response


def test_approach_2_image_placeholder_only(llm, tokenizer, img):
    """
    Approach 2: Image placeholder in messages, actual image in multi_modal_data
    (Current approach in your script - may not work correctly)
    """
    print("\n" + "="*80)
    print("APPROACH 2: Image placeholder in messages, image in multi_modal_data")
    print("="*80)
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Can you see an image in this message? If yes, please describe what you see in detail. If no, just say 'I cannot see any image.'"},
            {"type": "image"}  # No actual image object here
        ]
    }]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print(f"Prompt length: {len(prompt)} chars")
    print(f"Prompt preview (first 500 chars):\n{prompt[:500]}")
    print("\nGenerating response...")
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
    
    request = {"prompt": prompt, "multi_modal_data": {"image": [img]}}
    output = llm.generate([request], sampling_params)[0]
    response = output.outputs[0].text
    
    print("\n" + "-"*80)
    print("MODEL RESPONSE:")
    print("-"*80)
    print(response)
    print("-"*80 + "\n")
    
    return response


def test_approach_3_no_multi_modal_data(llm, tokenizer, img):
    """
    Approach 3: Image in content, NO multi_modal_data parameter
    (Alternative approach - might be correct for newer vLLM)
    """
    print("\n" + "="*80)
    print("APPROACH 3: Image in content, no multi_modal_data parameter")
    print("="*80)
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Can you see an image in this message? If yes, please describe what you see in detail. If no, just say 'I cannot see any image.'"},
            {"type": "image", "image": img}
        ]
    }]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print(f"Prompt length: {len(prompt)} chars")
    print(f"Prompt preview (first 500 chars):\n{prompt[:500]}")
    print("\nGenerating response...")
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
    
    # No multi_modal_data parameter
    request = {"prompt": prompt}
    output = llm.generate([request], sampling_params)[0]
    response = output.outputs[0].text
    
    print("\n" + "-"*80)
    print("MODEL RESPONSE:")
    print("-"*80)
    print(response)
    print("-"*80 + "\n")
    
    return response


def test_approach_4_simple_question(llm, tokenizer, img):
    """
    Approach 4: Image in content + multi_modal_data, with specific question
    (Tests with a more specific question about the image)
    """
    print("\n" + "="*80)
    print("APPROACH 4: Specific question about image content")
    print("="*80)
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "What colors are most prominent in this image? List the top 3 colors you see."}
        ]
    }]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print(f"Prompt length: {len(prompt)} chars")
    print(f"Prompt preview (first 500 chars):\n{prompt[:500]}")
    print("\nGenerating response...")
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
    
    request = {"prompt": prompt, "multi_modal_data": {"image": [img]}}
    output = llm.generate([request], sampling_params)[0]
    response = output.outputs[0].text
    
    print("\n" + "-"*80)
    print("MODEL RESPONSE:")
    print("-"*80)
    print(response)
    print("-"*80 + "\n")
    
    return response


def analyze_results(results):
    """Analyze which approaches successfully showed the image to the model."""
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    for approach_name, response in results.items():
        response_lower = response.lower()
        
        # Check for indicators that model saw the image
        saw_image = False
        indicators = []
        
        if "cannot see" in response_lower or "can't see" in response_lower or "no image" in response_lower:
            saw_image = False
            indicators.append("explicitly says cannot see image")
        elif any(word in response_lower for word in ["color", "colours", "image shows", "picture", "visual", "see a", "see an"]):
            saw_image = True
            indicators.append("describes visual content")
        elif len(response) > 50:  # Gave a substantial response
            saw_image = True
            indicators.append("provided detailed response (likely describing image)")
        
        status = "✅ SUCCESS" if saw_image else "❌ FAILED"
        
        print(f"\n{approach_name}: {status}")
        if indicators:
            print(f"  Reasoning: {', '.join(indicators)}")
        print(f"  Response preview: {response[:100]}...")
    
    print("\n" + "="*80)


def main():
    print("="*80)
    print("IMAGE VISIBILITY TEST FOR QWEN2.5-VL WITH VLLM")
    print("="*80)
    print(f"Model: {MODEL_PATH}")
    print(f"Test image: {TEST_IMAGE_PATH}")
    
    # Check if image exists
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"\nERROR: Test image not found at {TEST_IMAGE_PATH}")
        print("Please update TEST_IMAGE_PATH in the script to point to a valid image.")
        return
    
    # Load image
    print("\nLoading test image...")
    img = Image.open(TEST_IMAGE_PATH).convert("RGB")
    print(f"Image loaded: {img.size[0]}x{img.size[1]} pixels")
    
    # Initialize model
    print("\nInitializing model (this may take a few minutes)...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=4,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=8192,  # Shorter for testing
        gpu_memory_utilization=0.90,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=False
    )
    
    print("Model initialized!\n")
    
    # Run all tests
    results = {}
    
    try:
        results["Approach 1 (image in content)"] = test_approach_1_image_in_content(llm, tokenizer, img)
    except Exception as e:
        print(f"Approach 1 FAILED with error: {e}\n")
        results["Approach 1 (image in content)"] = f"ERROR: {str(e)}"
    
    try:
        results["Approach 2 (placeholder only)"] = test_approach_2_image_placeholder_only(llm, tokenizer, img)
    except Exception as e:
        print(f"Approach 2 FAILED with error: {e}\n")
        results["Approach 2 (placeholder only)"] = f"ERROR: {str(e)}"
    
    try:
        results["Approach 3 (no multi_modal_data)"] = test_approach_3_no_multi_modal_data(llm, tokenizer, img)
    except Exception as e:
        print(f"Approach 3 FAILED with error: {e}\n")
        results["Approach 3 (no multi_modal_data)"] = f"ERROR: {str(e)}"
    
    try:
        results["Approach 4 (specific question)"] = test_approach_4_simple_question(llm, tokenizer, img)
    except Exception as e:
        print(f"Approach 4 FAILED with error: {e}\n")
        results["Approach 4 (specific question)"] = f"ERROR: {str(e)}"
    
    # Analyze results
    analyze_results(results)
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nRecommendation: Use the approach that successfully showed the image to the model")
    print("in your main experiment script (run_task_ordering_4tasks.py)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

