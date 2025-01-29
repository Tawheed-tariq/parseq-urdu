import os
import json
import unicodedata
from nltk.metrics import edit_distance

# Normalize text using Unicode normalization
def normalize_text(text):
    return unicodedata.normalize('NFKD', text)

# Function to calculate CRR and WRR
def calculate_metrics(data):
    correct_words = 0
    total_words = 0
    total_word_edit_distance = 0  # For NED calculation
    total = 0

    for item in data:
        # Normalize ground truth and OCR text
        gt = normalize_text(item['gt'])
        ocr = normalize_text(item['ocr'])

        ocr_dt = ocr.split()
        gt_dt = gt.split()
        gt_dt = gt_dt[1:]  # Remove the first string from gt_dt
        # WRR calculation (word-level comparison)
        for ocr_word, gt_word in zip(ocr_dt[::-1], gt_dt[::-1]):  
            if ocr_word == gt_word:
                correct_words += 1
            total_words += 1


        # NED calculation (word-level edit distance)
        for ocr_word, gt_word in zip(ocr_dt, gt_dt):
            max_len = max(len(ocr_word), len(gt_word))
            if max_len > 0:
                word_dist = edit_distance(ocr_word, gt_word) / max_len
            else:
                word_dist = 0
            total_word_edit_distance += word_dist

        total += 1

    # WRR calculation
    wrr = (correct_words / total_words) * 100 if total_words > 0 else 0
    # NED calculation (normalized across all words)
    ned = (total_word_edit_distance / total_words) * 100 if total_words > 0 else 0
    crr = ((1 - ned / total) * 100)
    return crr, wrr

def process_files(gt_txt_path, pred_txt_dir):
    # Load ground truth from txt file
    with open(gt_txt_path, 'r', encoding='utf-8') as f:
        gt_lines = f.readlines()
    
    # Strip newlines and any extra whitespace
    gt_lines = [line.strip() for line in gt_lines]

    comparison_data = []

    # Iterate through each prediction txt file and process
    for idx, gt_text in enumerate(gt_lines):
        pred_file_path = os.path.join(pred_txt_dir, f'{idx}.txt')
        
        if os.path.exists(pred_file_path):
            with open(pred_file_path, 'r', encoding='utf-8') as f:
                try:
                    ocr_text = f.read().strip()
                except Exception as e:
                    print(f"Error reading file {pred_file_path}: {str(e)}")
                    continue
            
            # Prepare data for metric calculation
            comparison_data.append({
                'gt': gt_text,
                'ocr': ocr_text
            })

    if not comparison_data:
        raise ValueError("No valid comparison data found")

    # Calculate and return CRR and WRR
    return calculate_metrics(comparison_data)

# Example usage
gt_txt_path = "/home/tawheed/parseq/data/crr-wrr/IIITH/gt.txt"  # Path to your ground truth txt file
pred_txt_dir = '/home/tawheed/parseq/data/crr-wrr/IIITH/pred_32'  # Directory containing prediction files
crr, wrr = process_files(gt_txt_path, pred_txt_dir)

print(f"Correct Recognition Rate (CRR): {crr}%")
print(f"Word Recognition Rate (WRR): {wrr}%")