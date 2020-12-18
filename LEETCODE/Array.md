# Array

+ [Squares of a Sorted Array](#squares-of-a-sorted-array)
+ [Move Zeroes](#move-zeroes)
+ [Transpose Matrix](#transpose-matrix)
+ [Flipping an Image](#flipping-an-image)
+ [Image Smoother](#image-smoother)
+ [Reshape the Matrix](#reshape-the-matrix)
+ [Max Consecutive Ones](#max-consecutive-ones)

## Squares of a Sorted Array

+ [Squares of a Sorted Array](#squares-of-a-sorted-array)

https://leetcode.com/problems/squares-of-a-sorted-array/

``` python
def get_first_nonnegative(self, A: List[int]):
    for i, val in enumerate(A):
        if val >= 0:
            return i
    return -1
def sortedSquares(self, A: List[int]) -> List[int]:
    ind = self.get_first_nonnegative(A)
    if ind == -1:
        return [x ** 2 for x in A[::-1]]
    elif ind == 0:
        return [x ** 2 for x in A]
    else:
        left, right = ind - 1, ind
        length = len(A)
        res = []
        while right < length and left >= 0:
            if A[left] ** 2 < A[right] ** 2:
                res.append(A[left] ** 2)
                left -= 1
            else:
                res.append(A[right] ** 2)
                right += 1
        while left >= 0:
            res.append(A[left] ** 2)
            left -= 1
        while right < length:
            res.append(A[right] ** 2)
            right += 1
        return res
```

## Move Zeroes

+ [Move Zeroes](#move-zeroes)

https://leetcode.com/problems/move-zeroes/

``` python
def moveZeroes(self, nums: List[int]) -> None:
    count = 0
    while 0 in nums:
        nums.remove(0)
        count += 1
    for i in range(count):
        nums.append(0)
    return nums
```

## Transpose Matrix

+ [Transpose Matrix](#transpose-matrix)

https://leetcode.com/problems/transpose-matrix/

``` python
def transpose(self, A: List[List[int]]) -> List[List[int]]:
    new_arr = [[0]*len(A) for i in range(len(A[0]))]
    for i in range(len(A[0])):
        for j in range(len(A)):
            new_arr[i][j] = A[j][i]
    return new_arr
```

## Flipping an Image

+ [Flipping an Image](#flipping-an-image)

https://leetcode.com/problems/flipping-an-image/

``` python
def ReverseLine(self,line):
    l = len(line)
    for num_ind in range(l//2):
        line[num_ind], line[l - num_ind - 1] = line[l - num_ind - 1], line[num_ind]
    return line
def flipAndInvertImage(self, A: List[List[int]]) -> List[List[int]]:
    line_len = 0
    collumn_len = 0
    for line in A:
        line_len = len(line)
        collumn_len += 1
    for line_ind in range(collumn_len):
        for num_ind in range(line_len):
            if A[line_ind][num_ind] == 0:
                A[line_ind][num_ind] = 1
                continue
            else:
                A[line_ind][num_ind] = 0
    for line in A:
        line = self.ReverseLine(line)
    return A
```

## Image Smoother

+ [Image Smoother](#image-smoother)

https://leetcode.com/problems/image-smoother/

``` python
def imageSmoother(self, M: List[List[int]]) -> List[List[int]]:
    collumn_len = 0
    line_len = 0
    num_of_elements = 0
    summ_of_elements = 0
    for line in M:
        line_len = len(line)
        collumn_len += 1
    NewMatrix = M
    for line_ind in range(collumn_len):
        for num_ind in range(line_len):
            num_of_elements = 1
            summ_of_elements = M[line_ind][num_ind]
            if line_ind + 1 != collumn_len:
                num_of_elements += 1
                summ_of_elements += M[line_ind+1][num_ind]
            if line_ind + 1 != collumn_len and num_ind + 1 != line_len:
                num_of_elements += 1
                summ_of_elements += M[line_ind+1][num_ind+1]
            if num_ind + 1 != line_len:
                num_of_elements += 1
                summ_of_elements += M[line_ind][num_ind+1]
            if num_ind + 1 != line_len and line_ind - 1 >= 0:
                num_of_elements += 1
                summ_of_elements += M[line_ind - 1][num_ind+1]
            if line_ind - 1 >= 0:
                num_of_elements += 1
                summ_of_elements += M[line_ind - 1][num_ind]
            if num_ind - 1 >= 0 and line_ind - 1 >= 0:
                num_of_elements += 1
                summ_of_elements += M[line_ind - 1][num_ind - 1]
            if num_ind -1 >= 0:
                num_of_elements += 1
                summ_of_elements += M[line_ind][num_ind - 1]
            if num_ind - 1 >= 0 and line_ind + 1 != collumn_len:
                num_of_elements += 1
                summ_of_elements += M[line_ind + 1][num_ind - 1]
            NewMatrix[line_ind][num_ind] = summ_of_elements // num_of_elements
    return NewMatrix
```

## Reshape the Matrix

+ [Reshape the Matrix](#reshape-the-matrix)

https://leetcode.com/problems/reshape-the-matrix/

``` python
def matrixReshape(self, nums: List[List[int]], r: int, c: int) -> List[List[int]]:
    new_arr = []
    res = [[0]*c for i in range(r)]
    for i in range(len(nums)):
        for j in range(len(nums[i])):
            new_arr.append(nums[i][j])
    count = 0
    for i in range(r):
        for j in range(c):
            if count < len(new_arr):
                res[i][j] = int(new_arr[count])
            else:
                return nums
                break
            count += 1
    return(res)
```

## Max Consecutive Ones

+ [Max Consecutive Ones](#max-consecutive-ones)

https://leetcode.com/problems/max-consecutive-ones/

``` python
def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
    maxx, count = 0, 0
    for i in range(len(nums)):
        if nums[i] == 1:
            count += 1
        else:
            if count > maxx:
                maxx = count
            count = 0
    if count > maxx:
        maxx = count
    return maxx
```