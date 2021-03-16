import pickle
import numpy as np

with open("results_storage.list", "rb") as f:
    result_storage=pickle.load(f)

with open("results_outflow.list", "rb") as f:
    result_outflow=pickle.load(f)

result_outflow = np.asarray(result_outflow)
result_storage = np.asarray(result_storage)

print(result_storage.shape)
print(result_outflow.shape)
