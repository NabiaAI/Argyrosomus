
import os
import random
from collections import Counter, defaultdict

def load_data(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data = [line.strip() for line in lines]
    return data

def stratified_split_10_fold(data, label_list):
    label_to_data = defaultdict(list)
    for line in data:
        labels = line.strip().split('\t')[1].split()  
        label_str = "_".join(labels)
        label_to_data[label_str].append(line)
    
    # Shuffle data within each label group
    for label, items in label_to_data.items():
        random.shuffle(items)
    
    # Initialize folds
    folds = [[] for _ in range(10)]

    # Distribute data into folds
    for label, items in label_to_data.items():
        fold_size = len(items) // 10
        for i in range(10):
            folds[i].extend(items[i * fold_size:(i + 1) * fold_size])

    return folds

def count_labels(data, label_list):
    label_counter = Counter({label: 0 for label in label_list})
    for line in data:
        labels = line.strip().split('\t')[1].split() 
        for i, value in enumerate(labels):
            if int(value) == 1:
                label_counter[label_list[i]] += 1
    return label_counter

def write_data(data, filepath):
    with open(filepath, 'w') as f:
        for line in data:
            f.write(line + '\n')

def main(summary_filepath, label_list, output_dir):
    data = load_data(summary_filepath)
    folds = stratified_split_10_fold(data, label_list)

    for fold in range(10):
        test_data = folds[fold]
        train_data = [item for i in range(10) if i != fold for item in folds[i]]
        

        train_label_counts = count_labels(train_data, label_list)
        test_label_counts = count_labels(test_data, label_list)


        train_filepath = os.path.join(output_dir, f'train{fold}.txt')
        test_filepath = os.path.join(output_dir, f'test{fold}.txt')
        write_data(train_data, train_filepath)
        write_data(test_data, test_filepath)


        print(f"Generated train{fold}.txt and test{fold}.txt")
        print("Training set label distribution:")
        for label, count in train_label_counts.items():
            print(f"{label}: {count}")
        print("Testing set label distribution:")
        for label, count in test_label_counts.items():
            print(f"{label}: {count}")
        print("-" * 30)

if __name__ == "__main__":
    summary_filepath = './new_fish-split/output_osh/output_label/summary.txt'
    label_list = ['lt','m', 'w'] 
    output_dir = 'output_dir'

    os.makedirs(output_dir, exist_ok=True)
    main(summary_filepath, label_list, output_dir)