import numpy as np
import cv2

# 1.Create array of 10f zeros
vector = np.zeros(10)
print(vector)

# 2.Create array of 10f zeros whose 5th element is 1
vector = np.zeros(10)
vector[4] = 1
print(vector)

# 3.Create vector of int from 10 to 49
vector = np.arange(10, 50)
print(vector)

# 4.Create matrix 3x3f from 1 to 9
vector = np.arange(1, 10)
vector = vector.reshape((3, 3))
print(vector)

# 5.Create matrix 3x3 from 1 to 9 and flip horizontally
vector_fh = np.flip(vector, 1)
print(vector_fh)

# 6.Create matrix 3x3 from 1 to 9 and flip horizontally
vector_fv = np.flip(vector, 0)
print(vector_fv)

# 7.Create 3x3 identity mat
mat = np.identity(3)
print(mat)

# 8.Create 3x3 mat rand values
r_mat = np.random.randint(0, 100, 9)
r_mat = r_mat.reshape((3, 3))
print(r_mat)

# 9.Create rand vector of 10 num and compute the mean
r_vec = np.random.randint(0, 20, 10)
print(r_vec)
average = r_vec.mean()
print(average)

# 10.Create 10x10 zero array sourranded by 1
mat = np.ones((10, 10))
mat[1:9, 1:9] = 0.0
print(mat)

# 11.Create 5x5 matrix of rows from 1 to 5
mat = np.zeros((5, 5))
mat[:] = np.arange(1, 6)
print(mat)

# 12.Create array of 9 rand int and reshape to 3x3 mat float
mat = np.random.seed(101)
mat = np.random.randint(0, 100, 9)
mat = mat.reshape((3, 3))
mat = mat.astype(float)
print(mat)

# 13.Create 5x5 rand mat and substarct average
r_mat = np.random.randint(0, 100, 25)
r_mat = r_mat.reshape((5, 5))
print(r_mat)
average = r_mat.mean()
print(average)

# 14.Create 5x5 rand mat and substract average of eah row to each row
r_mat = np.random.randint(0, 100, 25)
r_mat = r_mat.reshape((5, 5))
print(r_mat)

# 15.Create 5x5 rand mat and return value closer to 0.5
r_mat = np.random.seed(101)
r_mat = np.random.random_sample((5, 5))

index = (np.abs(r_mat - 0.5)).argmin()
print(r_mat)
print(index)

# 16.Make 3x3 rand from 0 to 10 and count how many of them are > 5
r_mat = np.random.randint(0, 11, (3, 3))

print(r_mat)
print(len(r_mat[r_mat > 5]))

# 17.Create gradient image 64x64 from black to white
img = np.zeros((64, 64))
row = np.arange(0, 64)
row = row/64
print(row)
img[:] = row

while(True):
    cv2.imshow('Hola', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
