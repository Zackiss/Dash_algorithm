import csv
import json
import shutil


# Function to save parsed training dataset to a JSON file
def save_parsed_data_to_json(parsed_data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f)


# Function to load parsed dataset from a JSON file
def load_parsed_data_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        parsed_data = json.load(f)
    return parsed_data


# Function to process dataset from a CSV file and return parsed dataset
def process_csv_data(file_path):
    parsed_data = [[] for _ in range(7)]
    with open(file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if row['user_id'] == '':
                continue
            user_id = int(float(row['user_id']))
            item_id = int(row['problem_id'])
            score = int(row['correct'])
            time_window = int(row['time_window'])
            parsed_record = {'user_id': user_id, 'item_id': item_id, 'score': score}
            parsed_data[time_window - 1].append(parsed_record)
    return list(filter(None, parsed_data))


# Process train dataset
train_data_file = 'data_parser/train_data.csv'
parsed_train_data = process_csv_data(train_data_file)

# Save parsed training dataset to a JSON file
train_data_json_file = 'parsed_train_data.json'
save_parsed_data_to_json(parsed_train_data, train_data_json_file)

# Process test dataset
test_data_file = 'data_parser/test_data.csv'
parsed_test_data = process_csv_data(test_data_file)[0]

# Save test dataset to a JSON file
test_data_json_file = 'parsed_test_data.json'
save_parsed_data_to_json(parsed_test_data, test_data_json_file)

shutil.copy("data_parser/q_matrix.csv", "q_matrix.csv")
