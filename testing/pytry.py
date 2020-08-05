import json
my_dict = {}
for i in ['1','2','3','4','5']:
  my_dict[i] = {}
  my_dict[i]['name'] = 'john'
  my_dict[i]['age'] = 23
  my_dict[i]['rollno'] = 5792
with open('test.json', 'w') as f:
  json.dump(my_dict,f)

