import pandas as pd

def find_s(training_data):
    data_list = training_data.values.tolist()
    hypothesis = None
    for instance in data_list:
        if instance[-1] == 'Yes':  
            hypothesis = instance[:-1]
            break
    
    if hypothesis is None:
        return None
    for instance in data_list:
        if instance[-1] == 'Yes':
            for i in range(len(hypothesis)):
                if instance[i] != hypothesis[i]:
                    hypothesis[i] = '?'

    return hypothesis

def main():
    file_path = r'C:\Users\Welcome\OneDrive\Documents\training.csv'
    training_data = pd.read_csv(file_path)
    
    most_specific_hypothesis = find_s(training_data)
    
    if most_specific_hypothesis:
        print("Most Specific Hypothesis:")
        for attr, value in zip(training_data.columns[:-1], most_specific_hypothesis):
            print(f"{attr}: {value}")
    else:
        print("No hypothesis found (no positive examples in the dataset).")

if __name__ == "__main__":
    main()
