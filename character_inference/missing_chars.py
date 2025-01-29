import yaml

def load_charset(charset_file):
    """Load the charset from the provided YAML file."""
    with open(charset_file, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    charset = data['model']['charset_train']
    return charset

def load_ground_truth(ground_truth_file):
    """Load the ground truth labels from the provided file."""
    ground_truth = []
    with open(ground_truth_file, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            image, text = line.strip().split(' ', 1)
            ground_truth.append(text)
    return ground_truth

def find_missing_characters(charset, ground_truth):
    """Find characters that are in ground truth but not in the charset."""
    charset_set = set(charset)
    all_text = ''.join(ground_truth)
    missing_chars = set(all_text) - charset_set
    return missing_chars

def main(charset_path, ground_truth_path):
    charset = load_charset(charset_path)
    ground_truth = load_ground_truth(ground_truth_path)
    
    missing_chars = find_missing_characters(charset, ground_truth)
    
    if missing_chars:
        print("Missing characters:")
        for char in sorted(missing_chars):
            print(f"'{char}'")
    else:
        print("No missing characters!")

if __name__ == "__main__":
    charset_path = 'configs/charset/52_urdu.yaml'
    ground_truth_path = '/home/tawheed/parseq/data/crr-wrr/IIITH/gt.txt'
    
    main(charset_path, ground_truth_path)
