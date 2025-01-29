import json
import os
import unicodedata
from nltk.metrics import edit_distance  

# Normalize text using Unicode normalization
def normalize_text(text):
    return unicodedata.normalize('NFKD', text)

# Function to calculate CRR and WRR
def calculate_metrics(data):
    correct = 0
    total = 0
    ned = 0  

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

        total += 1

    wrr = (correct / total) * 100
    crr = (1 - ned / total) * 100

    return crr, wrr

def process_files(json_path, txt_dir):
    # Load JSON with ground truth and images
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    comparison_data = []

    # Iterate through each txt file and process
    for idx, item in enumerate(json_data):
        # Get the ground truth (GT) from the JSON file
        gt_text = item['gt']  # GT in Unicode
        print(gt_text)
        txt_file_path = os.path.join(txt_dir, f'{idx}.txt')
        
        if os.path.exists(txt_file_path):

            with open(txt_file_path, 'r', encoding='utf-8') as f:
                ocr_result = json.load(f)
            
            # Prepare data for metric calculation
            comparison_data.append({
                'gt': gt_text,  # Ground truth from JSON
                'ocr': ocr_result['text']  # OCR prediction from txt file
            })

    # Calculate and return CRR and WRR
    return calculate_metrics(comparison_data)

# Example usage
json_path = "/home/tawheed/parseq/BM-ST-UR-151.json" 
txt_dir = '/home/tawheed/parseq/data/dummy/pred'  # Directory containing txt files with OCR results
crr, wrr = process_files(json_path, txt_dir)

print(f"Correct Recognition Rate (CRR): {crr}%")
print(f"Word Recognition Rate (WRR): {wrr}%")
