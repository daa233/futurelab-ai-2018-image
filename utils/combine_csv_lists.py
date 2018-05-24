import csv


def load_csv_file(filepath):
    list_file = []
    with open(filepath, 'r') as csv_file:
        all_lines = csv.reader(csv_file)
        for line in all_lines:
            list_file.append(line)
    list_file.remove(list_file[0])
    return list_file


def write_list_to_csv(filepath, data_list):
    with open(filepath, 'w') as csv_file:
        csv_file.write('FILE_ID,CATEGORY_ID\n')
        for item in data_list:
            csv_file.write(str(item[0]) + ',' + str(item[1]) + '\n')


csv_file_a = '/media/ouc/4T_B/DuAngAng/datasets/futurelab/train_plus/5-fold-1-train.csv'
csv_file_b = '/media/ouc/4T_B/DuAngAng/datasets/futurelab/test/a/checked-test-data.csv'
data_list_a = load_csv_file(csv_file_a)
data_list_b = load_csv_file(csv_file_b)

data_list_combined = data_list_a
for item in data_list_b:
    data_list_combined.append(item)

write_list_to_csv('/media/ouc/4T_B/DuAngAng/datasets/futurelab/train_plus/train_plus.csv',
                  data_list_combined)