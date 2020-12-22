from string import ascii_uppercase
from string import digits
import sys

class Node:
    def __init__(self, value=None, next=None):
        self.val = value
        self.next = next


class HexNumber:
    def __init__(self, num):
        self.head = Node()
        self.arr = []
        for i in num[::-1]:
            if i.isdigit() or (i >= 'A' and i <= 'F'):
                self.addAtTail(i)
            else:
                TypeError


    def addAtTail(self, val):
        newNode = Node(val)
        if not self.head:
            self.head = newNode
            self.arr.append(newNode)
        else:
            cur = self.head
            while cur.next != None:
                cur = cur.next
            cur.next = newNode
            self.arr.append(newNode)


    def __str__(self):
        return ''.join(reversed(list(i.val for i in self.arr)))


class Solution:
    def __init__(self, str1, str2):
        self.str1 = str1
        self.str2 = str2
        self.ostatok = 0
        self.d = 0
        self.res = HexNumber('')

    def from_hex_to_decimal(self, num):
        return ord(num) - ord('A') + 10 if num >= 'A' and num <= 'F' else ord(num) - ord('0')


    def from_decimal_to_hex(self, num):
        return chr(ord('A') + num - 10) if num > 9 else chr(ord('0') + num)


    def add(self):
        len1 = len(self.str1.arr)
        len2 = len(self.str2.arr)

        if len1 == len2:
            mx = self.str1.arr
            mn = self.str2.arr
        else:
            if len(self.str1.arr) > len(self.str2.arr):
                mx = self.str1.arr
                mn = self.str2.arr
            else:
                mx = self.str2.arr
                mn = self.str1.arr
        for i in range(len(mn)):
            summa = self.from_hex_to_decimal(mx[i].val) + self.from_hex_to_decimal(mn[i].val)
            if summa >= 16:
                d = summa // 16
            o16 = summa % 16
            if self.ostatok != 0:
                self.res.addAtTail(self.from_decimal_to_hex(o16 + self.ostatok))
            else:
                self.res.addAtTail(self.from_decimal_to_hex(o16))
            if summa >= 16:
                self.res.addAtTail(self.from_decimal_to_hex(d))
            self.ostatok = summa // 16
            self.d = summa // 16
        if len1 != len2:
            for i in mx[len(mn)::]:
                self.res.addAtTail(i.val)
        if self.from_hex_to_decimal(self.res.arr[-1].val) >= 15 and self.ostatok != 0:
            self.res.addAtTail(self.from_decimal_to_hex(self.ostatok))
def proverka(params):
    k = 0
    for i in params:
            if i.isdigit()==False and (i < 'A' or i > 'F'):
                k += 1
    return k

if __name__ == '__main__':
    params = sys.argv
    k,d = 0, 0
    params = ['','F','4']
    if proverka(params[1]) == 0 and proverka(params[2]) ==0:
        sol = Solution(HexNumber(params[1]), HexNumber(params[2]))
        sol.add()
        print(sol.res)
    else:
        print('неправильный ввод')