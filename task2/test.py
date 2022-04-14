import numpy as np

a = np.matrix('1 5; 1 6; 2,7; 2,8')
h, w = a.shape
num_patients = 2
num_hours = 2
# A = np.zeros((1, num_hours*w))
# for i in range(num_patients):
#     sub_a = a[num_hours*i:num_hours*(i+1)]
#     b = np.reshape(sub_a, (1, num_hours*w), order='F')
#     A = np.append(A, b, axis=0)

print(a)


# arr = np.array([[1, 2, 3], [4, 5, 6]])
# row = np.array([7, 8, 9])
# arr = np.append(arr, [row], axis=0)
# print(arr)
