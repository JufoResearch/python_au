# String

+ [Valid Anagram](#valid-anagram)
+ [Reverse String](#reverse-string)
+ [Reverse Vowels of a String](#reverse-vowels-of-a-string)
+ [Reverse Words in a String III](#reverse-words-in-a-string-iii)
+ [To Lower Case](#to-lower-case)

## Valid Anagram

+ [Valid Anagram](#valid-anagram)

https://leetcode.com/problems/valid-anagram/

``` python
def isAnagram(self, s: str, t: str) -> bool:
    first = []
    second = []
    if len(s) != len(t):
        return False
    else:
        for i in range(len(s)):
            first.append(s[i])
            second.append(t[i])
        first.sort()
        second.sort()
        if first == second:
            return True
        return False
```

## Reverse String

+ [Reverse String](#reverse-string)

https://leetcode.com/problems/reverse-string/

``` python
def reverseString(self, s: List[str]) -> None:
    for i in range(len(s) // 2):
        t = s[len(s)- 1 - i]
        s[len(s) - 1 - i] = s[i]
        s[i] = t
    return s
```

## Reverse Vowels of a String

+ [Reverse Vowels of a String](#reverse-vowels-of-a-string)

https://leetcode.com/problems/reverse-vowels-of-a-string/

``` python
def reverseVowels(self, s: str) -> str:
    glas = []
    new_str = ''
    for i in range(len(s)):
        if s[i] == 'a' or s[i] == 'e' or s[i] == 'o' or s[i] == 'u' or s[i] == 'i' or s[i] == 'A' or s[i] == 'E' or s[i] == 'O' or s[i] == 'U' or s[i] == 'I':
            glas.append(s[i])
    j = len(glas) - 1
    for i in range(len(s)):
        if s[i] == 'a' or s[i] == 'e' or s[i] == 'o' or s[i] == 'u' or s[i] == 'i' or s[i] == 'A' or s[i] == 'E' or s[i] == 'O' or s[i] == 'U' or s[i] == 'I':
            new_str = new_str + glas[j]
            j -= 1
        else:
            new_str = new_str + s[i]
    return new_str
```

## Reverse Words in a String III

+ [Reverse Words in a String III](#reverse-words-in-a-string-iii)

https://leetcode.com/problems/reverse-words-in-a-string-iii/

``` python
def reverseWords(self, s: str) -> str:
    d = s.split()
    new_str = ''
    if s == '':
        return ''
    else:
        for i in range(len(d)-1):
            d[i] = d[i][::-1]
            new_str = new_str + str(d[i]) + ' '
        d[len(d)-1] = d[len(d)-1][::-1]
        new_str = new_str + str(d[len(d)-1])
        return new_str
```

## To Lower Case

+ [To Lower Case](#to-lower-case)

https://leetcode.com/problems/to-lower-case/

``` python
def toLowerCase(self, str):
    return(str.lower())
```