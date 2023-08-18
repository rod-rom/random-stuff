import numpy as np

arr = np.array([[1, 1, 1, 0, 0], 
                [3, 3, 3, 0, 0],
                [4, 4, 4, 0, 0],
                [5, 5, 5, 0, 0],
                [0, 2, 0, 4, 4],
                [0, 0, 0, 5, 5],
                [0, 1, 0, 2, 2]])

q = np.array([5,0,0,0,0])
s, v, d = np.linalg.svd(arr)

# print(arr.shape)
# print(f's.shape: {s.shape} \ns:{s}')
# print(f'v.shape: {v.shape} \nv:{v}')
# print(f'd.shape: {d.shape} \nd:{d}')

# dot product of s, v, d
# print(np.dot(np.dot(s, np.diag(v)), d))

# dot product of q, v
print(f'd shape : {d.shape}')
print(f'dot product: {np.dot(q, d)}')

# u,v,x are orthnormal
print(f'{s @ s.T}')
# print(np.dot(s.T, s))