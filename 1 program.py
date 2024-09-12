import pandas as pd

# Find-S Algorithm to find the most specific hypothesis
def find_s(training_data):
    # Convert DataFrame to a list of lists
    data_list = training_data.values.tolist()
    hypothesis = None

    # Initialize the most specific hypothesis with the first positive example
    for instance in data_list:
        if instance[-1] == 'Yes':  
            hypothesis = instance[:-1]  # Remove the last element (the label)
            break

    # If no positive examples were found, return None
    if hypothesis is None:
        return None

    # Generalize the hypothesis based on other positive examples
    for instance in data_list:
        if instance[-1] == 'Yes':
            for i in range(len(hypothesis)):
                if instance[i] != hypothesis[i]:
                    hypothesis[i] = '?'  # Replace differing attribute values with '?'

    return hypothesis

def main():
    # File path to the CSV file
    file_path = r"C:/Users/dinak/Downloads/Outlook.csv"

    # Try to read the CSV file into a DataFrame
    try:
        training_data = pd.read_csv(file_path, encoding='ISO-8859-1')
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"The file is empty: {file_path}")
        return
    except pd.errors.ParserError:
        print(f"Error parsing the file: {file_path}")
        return

    # Call the Find-S algorithm
    most_specific_hypothesis = find_s(training_data)

    # Display the most specific hypothesis found
    if most_specific_hypothesis:
        print("Most Specific Hypothesis:")
        for attr, value in zip(training_data.columns[:-1], most_specific_hypothesis):
            print(f"{attr}: {value}")
    else:
        print("No hypothesis found (no positive examples in the dataset).")

# Run the main function if the script is executed
if __name__ == "__main__":
    main()

