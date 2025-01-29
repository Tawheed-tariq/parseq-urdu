import os
import json
import unicodedata
from nltk.metrics import edit_distance

# Normalize text using Unicode normalization
def normalize_text(text):
    return unicodedata.normalize('NFKD', text)

# Preprocess the ground truth lines to remove image names
def preprocess_gt(gt_lines):
    processed_lines = []
    for line in gt_lines:
        # Split by whitespace and remove the first part (assumed to be the image name)
        parts = line.split(maxsplit=1)
        if len(parts) > 1:
            processed_lines.append(parts[1].strip())  # Keep only the text portion
        else:
            processed_lines.append("")  # Handle cases where no text follows the image name
    return processed_lines

# Function to calculate CRR and WRR, and log missed characters
def calculate_metrics(data, missed_chars_log_path):
    correct = 0
    total = 0
    ned = 0  

    missed_chars = {}

    for item in data:
        # Normalize ground truth and OCR text
        gt = normalize_text(item['gt'])
        ocr = normalize_text(item['ocr'])

        if max(len(ocr), len(gt)) == 0:
            dist = 0
        else:
            dist = edit_distance(ocr, gt) / max(len(ocr), len(gt))

        ned += dist

        if ocr == gt:
            correct += 1
        else:
            # Identify missed characters
            for char in gt:
                if char not in ocr:
                    missed_chars[char] = missed_chars.get(char, 0) + 1

        total += 1

    # Write missed characters to a file
    with open(missed_chars_log_path, 'w', encoding='utf-8') as f:
        json.dump(missed_chars, f, ensure_ascii=False, indent=4)

    wrr = (correct / total) * 100
    crr = (1 - ned / total) * 100

    return crr, wrr

def process_files(gt_txt_path, pred_txt_dir, missed_chars_log_path):
    # Load ground truth from txt file
    with open(gt_txt_path, 'r', encoding='utf-8') as f:
        gt_lines = f.readlines()
    
    # Preprocess ground truth lines to remove image names
    gt_lines = preprocess_gt(gt_lines)

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
    return calculate_metrics(comparison_data, missed_chars_log_path)

# Example usage
gt_txt_path = "/home/tawheed/parseq/data/crr-wrr/UPTI/gt.txt"  # Path to your ground truth txt file
pred_txt_dir = '/home/tawheed/parseq/data/crr-wrr/UPTI/pred_32'  # Directory containing prediction files
missed_chars_log_path = '/home/tawheed/parseq/data/crr-wrr/custom/missed_chars_32.json'  # Path to save missed characters log

crr, wrr = process_files(gt_txt_path, pred_txt_dir, missed_chars_log_path)

print(f"Correct Recognition Rate (CRR): {crr}%")
print(f"Word Recognition Rate (WRR): {wrr}%")
print(f"Missed characters logged in: {missed_chars_log_path}")
