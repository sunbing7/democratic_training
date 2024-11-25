import random
import numpy as np

test_index = random.sample(range(0, (50000 - 1)), 2000)
print(np.array(test_index))
np.save('index_test.npy', np.array(test_index))


