import numpy as np

data = np.load("lbph_model.npz", allow_pickle=True)

print(data.files)

descriptors = data["descriptors"]
student_ids = data["student_ids"]

print(descriptors.shape)
print(student_ids)