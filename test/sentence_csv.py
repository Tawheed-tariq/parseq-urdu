import os
import pandas as pd
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

def process_files(gt_csv_path, pred_txt_dir):
    # Load ground truth from CSV file
    df = pd.read_csv(gt_csv_path)

    if 'ground_truth' not in df.columns:
        raise ValueError("CSV file must contain a 'ground_truth' column")

    comparison_data = []

    # Iterate through each row and process
    for idx, row in df.iterrows():
        gt_text = str(row['ground_truth']).strip()
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
gt_csv_path = "/DATA/Tawheed/new/test/UR-ST-151-images/gt.csv"  # Path to your ground truth CSV file
pred_txt_dir = '/DATA/Tawheed/new/test/UR-ST-151-pred'  # Directory containing prediction files
crr, wrr = process_files(gt_csv_path, pred_txt_dir)

print(f"Correct Recognition Rate (CRR): {crr}%")
print(f"Word Recognition Rate (WRR): {wrr}%")
