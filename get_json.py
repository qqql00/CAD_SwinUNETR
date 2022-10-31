import json
import os

root = 'C:\\Users\\20211070\\Desktop\\test_json\\dataset'

training = []
dict = {'validation':[]}

# get the json file for the dataset
for i in os.listdir(root):
    if i=='image':
        img_r = os.path.join(root,i)
        for name in os.listdir(img_r):
            img_label = {'image': os.path.join(i, name), 'label': os.path.join('label', name)}
            dict['validation'].append(img_label)
# dataset_json_t = json.dumps(dict)
print(dict)
dataset_json_t = json.dumps(dict, indent=4)
# store this json file in 'dataset_json_t'
with open('dataset_json_t', 'w') as json_file:
    json_file.write(dataset_json_t)

################################

'''''
for i in os.listdir(root):
    if i=='image':
        img_r = os.path.join(root,i)
        for name in os.listdir(img_r):
            img_label = {'image': os.path.join('label', name), 'label': os.path.join(i, name)}
            dict['validation'].append(img_label)
# dataset_json_t = json.dumps(dict)
print(dict)
dataset_json_mask = json.dumps(dict, indent=4)
with open('dataset_json_mask', 'w') as json_file:
    json_file.write(dataset_json_mask)
'''''

