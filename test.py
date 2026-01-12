import sys
sys.setrecursionlimit(1000000)
input = sys.stdin.readline
visited1 = []
visited2 = []
visited3 = []

Tree = dict()
N = int(input())
for _ in range(N):
    s, l, r = map(str, input().split())
    Tree[s] = [l, r]

def preorder(now, Tree, visited):
    if now not in visited:
        visited.append(now)
    for i in range(2):
        if Tree[now][i] != '.':
            preorder(Tree[now][i], Tree, visited)

preorder('A', Tree, visited1)
print(''.join(visited1))

def inorder(now, Tree, visited):
    if Tree[now][0] not in visited and Tree[now][0] != '.':
        inorder(Tree[now][0], Tree, visited)
    visited.append(now)
    if Tree[now][1] != '.':
        inorder(Tree[now][1], Tree, visited)


inorder('A', Tree, visited2)
print(''.join(visited2))

def postorder(now, Tree, visited):
    for i in range(2):
        if Tree[now][i] != '.':
            postorder(Tree[now][i], Tree, visited)
        if i ==1 :
            visited.append(now)

postorder('A', Tree, visited3)
print(''.join(visited3))