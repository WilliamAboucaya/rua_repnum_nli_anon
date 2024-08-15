from sklearn.metrics import cohen_kappa_score
from pprint import pprint
import csv
import sys

if __name__ == '__main__':
    first_file = sys.argv[1]
    second_file = sys.argv[2]
    first = []
    second = []
    annotations = []
    with open(first_file, 'r', encoding='utf-8') as data:
        for line in csv.DictReader(data, delimiter=";"):
            #print(line["label"])
            if len(line["label"]):
                first.append(line["label"])
                annotations.append(line)

    with open(second_file, 'r', encoding='utf-8') as data:
        for line in csv.DictReader(data, delimiter=';'):
            #print(line["label"])
            second.append(line["label"])

    print(cohen_kappa_score(first, second[0:len(first)]))
    print(len(first))
    for i in range(0, len(first)):
        print(str(first[i]) + str(first[i]))
        if first[i] != second[i]:
            annotations[i]["line"] = i
            pprint(annotations[i])
            print("\n")
