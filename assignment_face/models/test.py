import numpy as np

data = np.load("lbph_model.npz", allow_pickle=True)

print(data.files)

descriptors = data["descriptors"]
student_ids = data["student_ids"]
image_paths = data["image_paths"]
grid_size = data["grid_size"]
lbp_variant = data["lbp_variant"]

print(descriptors.shape)
print(student_ids)