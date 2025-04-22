import os
import unicodedata
from nltk.metrics import edit_distance

# Normalize text using Unicode normalization
def normalize_text(text):
    return unicodedata.normalize('NFKD', text)

def preprocess_gt(gt_lines):
    image_names = []
    processed_lines = []
    
    for line in gt_lines:
        parts = line.split(maxsplit=1)
        if len(parts) > 1:
            image_names.append(parts[0])  # Store image name
            processed_lines.append(parts[1].strip())  # Store text portion
        else:
            image_names.append(parts[0] if parts else "")
            processed_lines.append("")
    
    return image_names, processed_lines

# Function to calculate CRR and WRR
def calculate_metrics(data, image_names, output_file):
    correct = 0
    total = 0
    ned = 0  
    errors = []

    for idx, item in enumerate(data):
        gt = normalize_text(item['gt'])
        ocr = normalize_text(item['ocr'])
        # image_name = image_names[idx]
        image_name = normalize_text(item['name'])

        if max(len(ocr), len(gt)) == 0:
            dist = 0
        else:
            dist = edit_distance(ocr, gt) / max(len(ocr), len(gt))

        ned += dist

        if ocr == gt:
            correct += 1
        else:
            errors.append(f"{image_name}\t\t\t{ocr}\t\t\t************************\t{gt}")  

        total += 1

    wrr = (correct / total) * 100
    crr = (1 - ned / total) * 100

    # Write errors to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"image_name\t\t\tground_truth\t\t\t************************\tpredicted_text\n")  
        for error in errors:
            f.write(error + "\n")

    return crr, wrr


def process_files(gt_txt_path, pred_txt_dir, output_file="ocr_mistakes.txt"):
    with open(gt_txt_path, 'r', encoding='utf-8') as f:
        gt_lines = f.readlines()
    
    image_names, gt_lines = preprocess_gt(gt_lines)

    comparison_data = []

    for idx, gt_text in enumerate(gt_lines):
        if idx >= 2750:
            break
        pred_file_path = os.path.join(pred_txt_dir, f'{idx}.txt')
        if os.path.exists(pred_file_path):
            with open(pred_file_path, 'r', encoding='utf-8') as f:
                try:
                    ocr_text = f.read().strip()
                except Exception as e:
                    print(f"Error reading file {pred_file_path}: {str(e)}")
                    continue
            
            comparison_data.append({
                'name':pred_file_path,
                'gt': gt_text,
                'ocr': ocr_text
            })

    if not comparison_data:
        raise ValueError("No valid comparison data found")

    return calculate_metrics(comparison_data, image_names, output_file)


# Example usage
gt_txt_path = "/DATA/Tawheed/data/crr-wrr/IIITH/gt.txt"
pred_txt_dir = "/DATA/Tawheed/data/crr-wrr/IIITH/pred"
output_file = "/home/tawheed/ProjectIITD/parseq/data_test/OCR_mistakes_IIITH.txt"

crr, wrr = process_files(gt_txt_path, pred_txt_dir, output_file)

print(f"Correct Recognition Rate (CRR): {crr}%")
print(f"Word Recognition Rate (WRR): {wrr}%")
print(f"Errors saved in: {output_file}")
