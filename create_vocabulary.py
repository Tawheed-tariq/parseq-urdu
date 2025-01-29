# Function to process the GT file and extract words
def process_gt_file(gt_file_path, output_file_path):
    words = []

    # Read the GT file line by line
    with open(gt_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            # Split at the first space to separate the filename from the text
            parts = line.split(' ', 1)
            if len(parts) > 1:
                text = parts[1].strip()  # Get the text part
                words.extend(text.split())  # Split the text into words and add to the list

    # Write the words to the output file, one per line
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for word in words:
            f.write(word + '\n')

# Input and output file paths
gt_file_path = "/home/tawheed/parseq/data/crr-wrr/UTRSet-Synth/gt.txt"  # Replace with the path to your input file
output_file_path = "/home/tawheed/parseq/data/crr-wrr/UTRSet-Synth/vocab.txt"  # Replace with the desired output file path

# Process the GT file
process_gt_file(gt_file_path, output_file_path)

print(f"Words have been extracted and written to {output_file_path}")
