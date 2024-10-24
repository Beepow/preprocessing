X = int(input())
N = int(input())
x = 0
for _ in range(N):
    a, b = map(int, input().split(' '))
    x += a*b
print('Yes') if x== X else print('No')