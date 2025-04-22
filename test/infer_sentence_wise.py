import os
import json
import unicodedata
from nltk.metrics import edit_distance

def normalize_text(text):
    return unicodedata.normalize('NFKD', text)

def preprocess_gt(gt_lines):
    processed_lines = []
    for line in gt_lines:
        parts = line.split(maxsplit=1)
        if len(parts) > 1:
            processed_lines.append(parts[1].strip())  
        else:
            processed_lines.append("")  
    return processed_lines


# Function to calculate CRR and WRR
def calculate_metrics(data):
    correct = 0
    total = 0
    ned = 0  
    total_words = 0

    for item in data:
        # Normalize ground truth and OCR text
        gt = normalize_text(item['gt'])
        ocr = normalize_text(item['ocr'])

        if max(len(ocr), len(gt)) == 0:
            dist = 0
        else:
            dist = edit_distance(ocr, gt) / max(len(ocr), len(gt))

        ned += dist

        # WRR calculation (word-level comparison)
        gt_words = item['gt'].split()  
        ocr_words = item['ocr'].split()
        for ocr_word, gt_word in zip(ocr_words, gt_words):  
            if ocr_word == gt_word:
                correct += 1
        total_words += len(gt_words)

        total += 1

    wrr = (correct / total_words) * 100
    crr = (1 - ned / total) * 100

    return crr, wrr

def process_files(gt_txt_path, pred_txt_dir):
    with open(gt_txt_path, 'r', encoding='utf-8') as f:
        gt_lines = f.readlines()
    
    # Strip newlines and any extra whitespace
    gt_lines = preprocess_gt(gt_lines)

    comparison_data = []

    for idx, gt_text in enumerate(gt_lines):
        pred_file_path = os.path.join(pred_txt_dir, f'{idx}.txt')
        
        if os.path.exists(pred_file_path):
            with open(pred_file_path, 'r', encoding='utf-8') as f:
                try:
                    ocr_text = f.read().strip()
                except Exception as e:
                    print(f"Error reading file {pred_file_path}: {str(e)}")
                    continue
            
            comparison_data.append({
                'gt': gt_text,
                'ocr': ocr_text
            })

    if not comparison_data:
        raise ValueError("No valid comparison data found")

    return calculate_metrics(comparison_data)


gt_txt_path = "/DATA/Tawheed/new/test/UR-ST-160-images/gt.txt"  # Path to your ground truth txt file
pred_txt_dir = '/DATA/Tawheed/new/test/UR-ST-160-pred'  # Directory containing prediction files
crr, wrr = process_files(gt_txt_path, pred_txt_dir)

print(f"Correct Recognition Rate (CRR): {crr}%")
print(f"Word Recognition Rate (WRR): {wrr}%")