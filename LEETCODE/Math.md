# Math

+ [Sqrt(x)](#sqrt(x))
+ [Largest Perimeter Triangle](#largest-perimeter-triangle)
+ [Fibonacci Number](#fibonacci-number)
+ [Base 7](#base-7)
+ [Fizz Buzz](#fizz-buzz)
+ [Palindrome Number](#palindrome-number)
+ [Reverse Integer](#reverse-integer)

## Sqrt(x)

+ [Sqrt(x)](#sqrtx)

https://leetcode.com/problems/sqrtx/

``` python
def mySqrt(self, x: int) -> int:
    return int(math.sqrt(x))
```

## Largest Perimeter Triangle

+ [Largest Perimeter Triangle](#largest-perimeter-triangle)

https://leetcode.com/problems/largest-perimeter-triangle/

``` python
def largestPerimeter(self, A: List[int]) -> int:
    maxx=0
    for i in range(0,len(A) - 2):
        for j in range(i+1,len(A) - 1):
            for t in range(j + 1,len(A)):
                if A[i] + A[j] > A[t] and A[i] + A[t] > A[j] and A[t] + A[j] > A[i]:
                    if A[i] + A[j] + A[t] > maxx:
                        maxx = A[i] + A[j] + A[t]
    return maxx
```

## Fibonacci Number

+ [Fibonacci Number](#fibonacci-number)

https://leetcode.com/problems/fibonacci-number/

``` python
def fib(self, N: int) -> int:
    if N == 0:
        return 0
    elif N == 1:
        return 1
    elif N > 1:
        return(self.fib(N-1) + self.fib(N-2))
```

## Base 7

+ [Base 7](#base-7)

https://leetcode.com/problems/base-7/

``` python
def convertToBase7(self, num: int) -> str:
    new_int = ''
    k = 1
    if num < 0:
        num = num * -1
        k = -1
    if num == 0:
        return '0'
    else:
        while num != 0:
            new_int = new_int + str(num % 7)
            num = num // 7
        new_int = new_int[::-1]
        return(str(int(new_int)*k))
```

## Fizz Buzz

+ [Fizz Buzz](#fizz-buzz)

https://leetcode.com/problems/fizz-buzz/

``` python
def fizzBuzz(self, n: int) -> List[str]:
    res = []
    for i in range(1,n + 1):
        if i % 3 == 0 and i % 5 == 0:
            res.append("FizzBuzz")
        elif i % 3 == 0:
            res.append("Fizz")
        elif i % 5 == 0:
            res.append("Buzz")
        else:
            res.append(str(i))
    return res
```

## Palindrome Number

+ [Palindrome Number](#palindrome-number)

https://leetcode.com/problems/palindrome-number/

``` python
def isPalindrome(self, x: int) -> bool:
    if x < 0:
        return False
    else:
        for i in range(0,len(str(x))//2):
            if (str(x))[i] != (str(x))[len(str(x))-i-1]:
                return False
        return True
```

## Reverse Integer

+ [Reverse Integer](#reverse-integer)

https://leetcode.com/problems/reverse-integer/

``` python
def reverse(self, x: int) -> int:
        a = 0
        if x == 0:
            return x
        elif x < 0:
            x = x * (-1)
            while x != 0:
                a = a * 10 + x % 10
                x = x // 10
            if abs(a) < pow(2,31) - 1:
                return a * (-1)
            else:
                return 0
        else:
            while x != 0:
                a = a * 10 + x % 10
                x = x // 10
            if abs(a) < pow(2,31) - 1:
                return a
            else:
                return 0
```