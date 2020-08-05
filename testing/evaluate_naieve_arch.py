with open('conv1.weights', 'r') as f:
 weights = f.readlines()
prev_wei = 0
cycles = 0
for j in range():
  for i in range(256):
    curr_wei = int(weights[i])
    if curr_wei == prev_wei:
      cycles += 1
  cycles += 1
