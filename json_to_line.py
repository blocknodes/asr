import sys
# read file to string
with open(sys.argv[1], 'r') as f:
    data = f.read()
# replace newlines with spaces
data = data.replace('\'', '\"')
# print result
print(data)

# load into json object
import json
json_data = json.loads(data)
# iterate over items and write to file
with open(sys.argv[2], 'w') as f:
    for item in json_data:
        f.write(f'{item["key"]}    {item["text"]}\n')

