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

# Function to save character distribution plot to a file
def save_character_distribution_plot(char_distribution, save_path, dataset_name):
    # Sort characters by frequency
    sorted_characters = sorted(char_distribution.items(), key=lambda x: x[1], reverse=True)
    characters, frequencies = zip(*sorted_characters)

    # Plot the distribution
    plt.figure(figsize=(15, 5))
    plt.bar(characters, frequencies)
    plt.xlabel("Characters")
    plt.ylabel("Frequency")
    plt.title(f"Character Distribution {dataset_name}")
    plt.xticks(rotation=90)

    # Save the plot to the specified location
    plt.savefig(save_path)
    plt.close()  # Close the plot to avoid display

    print(f"Plot saved at: {save_path}")

# Example usage
gt_file_path = "/home/tawheed/parseq/data/crr-wrr/UPTI/gt.txt"  # Replace this with the path to your GT file
save_path = "/home/tawheed/parseq/data/char_dist_upti.png"
dataset_name = gt_file_path.split('/')[-2]

# Calculate character distribution
char_distribution = calculate_character_distribution(gt_file_path)

# Print the results
print("Character Distribution:")
for char, freq in char_distribution.items():
    print(f"'{char}': {freq}")

# Save the plot to the user-defined location
save_character_distribution_plot(char_distribution, save_path, dataset_name)
