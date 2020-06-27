from functools import lru_cache
from numpy import argmax

n, k = [int(s) for s in input().split(" ")]
board = []
for i in range(n):
    board.append([int(s) for s in input().split(" ")])

@lru_cache(maxsize = None)
def maxworth(rows, tokens, zero, one, two):
    if rows < 0 or tokens < 0:
        return float('-inf')
    if rows == 0 and tokens != 0:
        return float('-inf')
    if tokens == 0:
        return 0
    c1 = maxworth(rows - 1, tokens, 0, 0, 0)
    if zero == one == 0:
        c2 = maxworth(rows - 1, tokens - 1, 0, 0, 0) + board[rows - 1][0] + board[rows - 1][1]
    else:
        c2 = float('-inf')
    if one == two == 0:
        c3 = maxworth(rows - 1, tokens - 1, 0, 0, 0) + board[rows - 1][1] + board[rows - 1][2]
    else:
        c3 = float('-inf')
    if zero == one == two == 0 and rows >= 2:
        c4 = maxworth(rows - 1, tokens - 2, 0, 0, 1) + board[rows - 1][0] + board[rows - 1][1] + board[rows - 1][2] + board[rows - 2][2]
    else:
        c4 = float('-inf')
    if zero == one == two == 0 and rows >= 2:
        c5 = maxworth(rows - 1, tokens - 2, 1, 0, 0) + board[rows - 1][0] + board[rows - 1][1] + board[rows - 1][2] + board[rows - 2][0]
    else:
        c5 = float('-inf')
    if zero == 0 and rows >= 2:
        c6 = maxworth(rows - 1, tokens - 1, 1, 0, 0) + board[rows - 1][0] + board[rows - 2][0]
    else:
        c6 = float('-inf')
    if one == 0 and rows >= 2:
        c7 = maxworth(rows - 1, tokens - 1, 0, 1, 0) + board[rows - 1][1] + board[rows - 2][1]
    else:
        c7 = float('-inf')
    if two == 0 and rows >= 2:
        c8 = maxworth(rows - 1, tokens - 1, 0, 0, 1) + board[rows - 1][2] + board[rows - 2][2]
    else:
        c8 = float('-inf')
    if zero == one == 0 and rows >= 2:
        c9 = maxworth(rows - 1, tokens - 2, 1, 1, 0) + board[rows - 1][0] + board[rows - 2][0] + board[rows - 1][1] + board[rows - 2][1]
    else:
        c9 = float('-inf')
    if one == two == 0 and rows >= 2:
        c10 = maxworth(rows - 1, tokens - 2, 0, 1, 1) + board[rows - 1][1] + board[rows - 2][1] + board[rows - 1][2] + board[rows - 2][2]
    else:
        c10 = float('-inf')
    if zero == two == 0 and rows >= 2:
        c11 = maxworth(rows - 1, tokens - 2, 1, 0, 1) + board[rows - 1][2] + board[rows - 2][2] + board[rows - 1][0] + board[rows - 2][0]
    else:
        c11 = float('-inf')
    if zero == one == two == 0 and rows >= 2:
        c12 = maxworth(rows - 2, tokens - 3, 0, 0, 0) + board[rows - 1][1] + board[rows - 2][1] + board[rows - 1][2] + board[rows - 2][2] + board[rows - 1][0] + board[rows - 2][0]
    else:
        c12 = float('-inf')

    cs = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12]
    print(cs)
    cmax = argmax(cs)
    print(rows, tokens, zero, one, two, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, cmax, cs[cmax])
    return cs[cmax]


print("Case {}: {}".format(i, maxworth(n, k, 0, 0, 0)))