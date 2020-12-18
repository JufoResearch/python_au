# Array

## Image Smoother

+ [Image Smoother](#image-smoother)

https://leetcode.com/problems/image-smoother/

``` python
def imageSmoother(self, M: List[List[int]]) -> List[List[int]]:
    mass = [[0 for _ in range(len(M[0]))] for i in range(len(M))]
    for i in range(len(M)):
        for j in range(len(M[0])):
            count = 0
            for k in (i - 1, i,i + 1):
                for l in (j - 1, j, j + 1):
                    if 0 <= l < len(M[0]) and 0 <= k < len(M):
                        mass[i][j] += M[k][l]
                        count += 1
            mass[i][j] //= count
    return mass