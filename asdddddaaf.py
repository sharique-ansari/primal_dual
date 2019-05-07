import numpy as np
f = open('cons.txt')
A = []
for l in f:
    A.append([int(i) for i in l.split(' ')])
A = np.array(A)
b = A[:-1, -1]
c = A[-1, :-1].T
A = A[:-1,:-1]

# x = pdmethod(A, b, c.T)

print("Question 1:")
# print(x)
# print(c.T@x)



print()
print()
print()

f = open('cons2.txt')
A = []
for l in f:
    A.append([np.float(i) for i in l.split(' ')])
A = np.array(A)

f = open('cons3.txt')
b = []
for l in f:
    b.append(np.float(l))
b = np.array(b)

f = open('cons4.txt')
c = []
for l in f:
    c.append(np.float(l))
c = np.array(c)

# x = pdmethod(A, b.T, c.T)

print("Question 2:")
# print(c.T@x)