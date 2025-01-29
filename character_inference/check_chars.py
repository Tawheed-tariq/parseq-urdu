# Define the input file path for the ground truth
file_path = "/home/tawheed/parseq/data/crr-wrr/UPTI/gt.txt"  # Replace with your file's path

try:
    # Read the file
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Remove the image names and extract only the text
    texts = [line.split(" ", 1)[1].strip() for line in lines if " " in line]

    # Find the length of each line
    lengths = [len(text) for text in texts]

    # Calculate max, min, and average character counts
    max_length = max(lengths)
    min_length = min(lengths)
    avg_length = sum(lengths) / len(lengths)

    # Find the corresponding lines with max and min lengths
    max_line = texts[lengths.index(max_length)]
    min_line = texts[lengths.index(min_length)]

    # Print the results
    print(f"Maximum characters: {max_length}\nLine: {max_line}")
    print(f"Minimum characters: {min_length}\nLine: {min_line}")
    print(f"Average characters per line: {avg_length:.2f}")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")