import json
from collections import Counter
import matplotlib.pyplot as plt

# Function to read GT files and calculate character distribution
def calculate_character_distribution(gt_file_path):
    char_counter = Counter()

    # Read the GT files line by line
    with open(gt_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            # Split at the first space to separate filename from the text
            parts = line.split(' ', 1)
            if len(parts) > 1:
                text = parts[1].strip()  # Get the ground truth text after the filename
                char_counter.update(text)
    
    return char_counter

# Function to print and visualize character distribution
def visualize_character_distribution(char_distribution):
    # Sort characters by frequency
    sorted_characters = sorted(char_distribution.items(), key=lambda x: x[1], reverse=True)
    characters, frequencies = zip(*sorted_characters)

    # Plot the distribution
    plt.figure(figsize=(15, 5))
    plt.bar(characters, frequencies)
    plt.xlabel("Characters")
    plt.ylabel("Frequency")
    plt.title("Character Distribution")
    plt.xticks(rotation=90)

    # Show the plot
    plt.show()

    return sorted_characters

# Function to save character distribution as a JSON file
def save_to_json(char_distribution, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(char_distribution, json_file, ensure_ascii=False, indent=4)
    print(f"Character distribution saved to {output_file_path}")

# Example usage
gt_file_path = "/home/tawheed/parseq/data/crr-wrr/UTRSet-Real/gt.txt"  # Replace this with the path to your GT file
output_json_path = "/home/tawheed/parseq/data/crr-wrr/UTRSet-Real/char_distribution_UTRSet-Real.json"  # Replace this with your desired output path

# Calculate character distribution
char_distribution = calculate_character_distribution(gt_file_path)

# Print the results
print("Character Distribution:")
for char, freq in char_distribution.items():
    print(f"'{char}': {freq}")

# Visualize the distribution
visualize_character_distribution(char_distribution)

# Save the distribution as JSON
save_to_json(char_distribution, output_json_path)
