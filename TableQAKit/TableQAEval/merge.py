import json
import random

with open('TableQAEval_Encyclopedia.json') as file1, open('TableQAEval_SpreadSheet2.json') as file2, open('TableQAEval_StructuredQA.json') as file3:
    data1, data2, data3 = json.load(file1), json.load(file2), json.load(file3)
data1 = data1[:300]
data2 = data2[:300]
data3 = data3[:400]
print(len(data1), len(data2), len(data3))
total_data = []
for d1 in data1:
    d1['source'] = 'multihop'
    total_data.append(d1)
for d2 in data2:
    d2['source'] = 'numerical'
    total_data.append(d2)
for d3 in data3:
    d3['source'] = 'structured'
    total_data.append(d3)


random.shuffle(total_data)

with open('TableQAEval.json', 'w') as f:
    json.dump(total_data, f, indent=2)