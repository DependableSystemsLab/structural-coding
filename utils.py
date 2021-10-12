def lcs(X, Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None] * (n + 1) for i in range(m + 1)]

    """Following steps build L[m + 1][n + 1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = []
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + [(i -1, j - 1)]
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1], key=lambda l: len(l))

    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]


def biggest_power_of_two(n):
    result = 1
    while n % 2 == 0:
        n = n // 2
        result *= 2
    return result


def biggest_divisor_smaller_than(n, k):
    for i in range(k, -1, -1):
        if n % i == 0:
            return i
    return 1
