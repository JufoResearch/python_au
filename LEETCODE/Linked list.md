# Linked list

+ [Reverse Linked List](#reverse-linked-list)
+ [Middle of the Linked List](#middle-of-the-linked-list)
+ [Palindrome Linked List](#palindrome-linked-list)
+ [Sort List](#sort-list)
+ [Intersection of Two Linked Lists](#intersection-of-two-linked-lists)
+ [Linked List Cycle](#linked-list-cycle)
+ [Linked List Cycle II](#linked-list-cycle-ii)
+ [Merge Two Sorted Lists](#merge-two-sorted-lists)
+ [Remove Nth Node From End of List](#remove-nth-node-from-end-of-list)
+ [Reorder List](#reorder-list)


## Reverse Linked List

+ [Reverse Linked List](#reverse-linked-list)

https://leetcode.com/problems/reverse-linked-list/

``` python
ass ListNode:
  def __init__(self, val=0, next=None):
      self.val = val
      self.next = next
s Solution:
def reverseList(self, head: ListNode) -> ListNode:
    prev = None
    next = None
    cur = head
    while cur != None:
        next = cur.next
        cur.next = prev
        prev = cur
        cur = next
    return  prev
```

## Middle of the Linked List

+ [Middle of the Linked List](#middle-of-the-linked-list)

https://leetcode.com/problems/middle-of-the-linked-list/

``` python
ass ListNode:
  def __init__(self, val=0, next=None):
      self.val = val
      self.next = next
s Solution:
def middleNode(self, head: ListNode) -> ListNode:
    array = [head]
    while array[-1].next!=None:
        array.append(array[-1].next)
    return array[len(array)//2]
```

## Palindrome Linked List

+ [Palindrome Linked List](#palindrome-linked-list)

https://leetcode.com/problems/palindrome-linked-list/

``` python
ass ListNode:
  def __init__(self, val=0, next=None):
      self.val = val
      self.next = next
s Solution:
def reverse(self, head: ListNode):
    cur = head
    prev = None
    next = None
    while cur != None:
        next = cur.next
        cur.next = prev
        prev = cur
        cur = next
    return prev
def lenght(self, head:ListNode):
    count = 0
    while head:
        count += 1
        head = head.next
    return count
def isPalindrome(self, head:ListNode) -> int:
    first = head
    second = self.reverse(head)
    for i in range(self.lenght(head)):
        if first.val != second.val:
            return False
        else:
            first = first.next
            second = second.next
    return True
```

## Merge Two Sorted Lists

+ [Merge Two Sorted Lists](#merge-two-sorted-lists)

https://leetcode.com/problems/merge-two-sorted-lists/

``` python
ass ListNode:
  def __init__(self, val=0, next=None):
      self.val = val
      self.next = next
s Solution:
def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
    cur = output = ListNode()
    while l1 != None and l2 != None:
        if l1.val < l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    if l1 == None:
        cur.next = l2
    if l2 == None:
        cur.next = l1
    return output.next
```

## Remove Nth Node From End of List

+ [Remove Nth Node From End of List](#remove-nth-node-from-end-of-list)

https://leetcode.com/problems/remove-nth-node-from-end-of-list/

``` python
ass ListNode:
  def __init__(self, val=0, next=None):
      self.val = val
      self.next = next
s Solution:
def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
    List = []
    ReversedNewHead = None
    NewHead = None
    while head:
        List.append(head.val)
        head = head.next
    del List[-n]
    for element in List:
        ReversedNewHead = ListNode(element, ReversedNewHead)
    while ReversedNewHead:
        NewHead = ListNode(ReversedNewHead.val, NewHead)
        ReversedNewHead = ReversedNewHead.next
    return(NewHead)
```

## Linked List Cycle II

+ [Linked List Cycle II](#linked-list-cycle-ii)

https://leetcode.com/problems/linked-list-cycle-ii/

``` python
ass ListNode:
  def __init__(self, x):
      self.val = x
      self.next = None
s Solution:
def detectCycle(self, head: ListNode) -> ListNode:
    List = []
    sectime = 0
    n1 = 0
    n2 = 0
    while head:
        if head.val in List:
            sectime = head.val
            while head:
                if head.val == sectime:
                    while head:
                        if head.val == sectime:
                            if n1 == n2:
                                return(head)
                        n2 += 1
                        head = head.next
                n1 += 1
                head = head.next
        List.append(head.val)
        head = head.next
    return(None)
```

## Reorder List

+ [Reorder List](#reorder-list)

https://leetcode.com/problems/reorder-list/

``` python
ass ListNode:
  def __init__(self, val=0, next=None):
      self.val = val
      self.next = next
s Solution:
def __init__(self, val=0, next=None):
    self.val = val
    self.next = next
s Solution:
def reverse_list(self, head):
    prev = None
    curr = head
    while curr is not None:
        next = curr.next
        curr.next = prev
        prev = curr
        curr = next
    return prev
def get_middle_of_list(self, head):
    if head is None:
        return head
    slow = head
    fast = head
    prev_slow = None
    while fast is not None and fast.next is not None:
        prev_slow = slow
        slow = slow.next
        fast = fast.next.next
    if fast is None:
        prev_slow.next = None
        return slow
    else:
        res = slow.next
        slow.next = None
        return res
def merge_lists(self, left, right):
    cur_left = left
    cur_right = right
    while cur_left is not None and cur_right is not None:
        next_left = cur_left.next
        next_right = cur_right.next
        cur_left.next = cur_right
        cur_right.next = next_left
        cur_left = next_left
        cur_right = next_right
    return left
def reorderList(self, head):
    prev_middle = self.get_middle_of_list(head)
    first = head
    second = prev_middle
    second = self.reverse_list(second)
    head = self.merge_lists(first, second)
```

## Intersection of Two Linked Lists

+ [Intersection of Two Linked Lists](#intersection-of-two-linked-lists)

https://leetcode.com/problems/intersection-of-two-linked-lists/

``` python
ass ListNode:
  def __init__(self, x):
      self.val = x
      self.next = None
s ListNode:
def __init__(self, val=0, next=None):
    self.val = val
    self.next = next
s Solution:
def sortList(self, head: ListNode) -> ListNode:
    if head == None or head.next == None:
        return head
    p, slow, fast = None, head, head
    while fast and fast.next:
        p = slow
        slow = slow.next
        fast = fast.next.next
    p.next = None
    return self.merge(self.sortList(head), self.sortList(slow))
def merge(self, l1: ListNode, l2: ListNode) -> ListNode:
    cur = output = ListNode()
    while l1 != None and l2 != None:
        if l1.val < l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    if l1 == None:
        cur.next = l2
    if l2 == None:
        cur.next = l1
    return output.next
```

## Sort List

+ [Sort List](#sort-list)

https://leetcode.com/problems/sort-list/

``` python
ass ListNode:
  def __init__(self, val=0, next=None):
      self.val = val
      self.next = next
s ListNode:
def __init__(self, val=0, next=None):
    self.val = val
    self.next = next
s Solution:
def sortList(self, head: ListNode) -> ListNode:
    if head == None or head.next == None:
        return head
    p, slow, fast = None, head, head
    while fast and fast.next:
        p = slow
        slow = slow.next
        fast = fast.next.next
    p.next = None
    return self.merge(self.sortList(head), self.sortList(slow))
def merge(self, l1: ListNode, l2: ListNode) -> ListNode:
    cur = output = ListNode()
    while l1 != None and l2 != None:
        if l1.val < l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    if l1 == None:
        cur.next = l2
    if l2 == None:
        cur.next = l1
    return output.next
```

## Intersection of Two Linked Lists

+ [Intersection of Two Linked Lists](#intersection-of-two-linked-lists)

https://leetcode.com/problems/intersection-of-two-linked-lists/

``` python
ass ListNode:
  def __init__(self, x):
      self.val = x
      self.next = None
s ListNode:
def __init__(self, val=0, next=None):
    self.val = val
    self.next = next
s Solution:
def sortList(self, head: ListNode) -> ListNode:
    if head == None or head.next == None:
        return head
    p, slow, fast = None, head, head
    while fast and fast.next:
        p = slow
        slow = slow.next
        fast = fast.next.next
    p.next = None
    return self.merge(self.sortList(head), self.sortList(slow))
def merge(self, l1: ListNode, l2: ListNode) -> ListNode:
    cur = output = ListNode()
    while l1 != None and l2 != None:
        if l1.val < l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    if l1 == None:
        cur.next = l2
    if l2 == None:
        cur.next = l1
    return output.next
```