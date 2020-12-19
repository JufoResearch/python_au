class Node:
    def __init__(self,val):
        self.val = val
        self.next = None


class MyLinkedList:
    def __iter__(self):
        return self


    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.head = None
        self.count = 0
        self.limit = 3
        self.counter = 0


    def __next__(self):
        if self.counter < self.limit:
            self.counter += 1
            return self.get(self.counter - 1)
        else:
            raise StopIteration


    def printlist(self):
        for elem in self:
            print(elem)
        self.counter = 0


    def get(self, index: int) -> int:
        """
        Get the value of the index-th node in the linked list. If the index is invalid, return -1.
        """
        if index < 0 or index >= self.count:
            return -1
        else:
            cur = self.head
            while index > 0:
                cur = cur.next
                index -= 1
            return cur.val

    def addAtHead(self, val: int) -> None:
        """
        Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
        """
        if not self.head:
            self.head = Node(val)
        else:
            newNode = Node(val)
            newNode.next = self.head
            self.head = newNode
        self.count += 1

    def addAtTail(self, val: int) -> None:
        """
        Append a node of value val to the last element of the linked list.
        """
        newNode = Node(val)
        if not self.head:
            self.head = newNode
        else:
            cur = self.head
            while cur.next != None:
                cur = cur.next
            cur.next = newNode
        self.count += 1


    def addAtIndex(self, index: int, val: int) -> None:
        """
        Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
        """
        if index == 0:
            return self.addAtHead(val)
        elif index == self.count:
            return self.addAtTail(val)
        elif index < 0 or index >= self.count:
            return
        else:
            newNode = Node(val)
            cur = self.head
            for i in range(index - 1):
                cur = cur.next
            newNode.next = cur.next
            cur.next = newNode
            self.count += 1


    def deleteAtIndex(self, index: int) -> None:
        """
        Delete the index-th node in the linked list, if the index is valid.
        """
        if index < 0 or index >= self.count:
            return
        elif self.count == 1 and index == 0:
            self.head = None
            self.count = 0
            return
        elif self.count - 1 == index:
            cur = self.head
            while cur.next.next != None:
                cur = cur.next
            cur.next = None
        elif index == 0:
            self.head = self.head.next
        else:
            cur = self.head
            prev = None
            while index != 0:
                prev = cur
                cur = cur.next
                index -= 1
            prev.next = cur.next
            cur = None
        self.count -= 1

class Student:
    def __init__(self, name, group, subgroup):
        self.name = name
        self.group = group
        self.subgroup = subgroup

    def __str__(self):
        return f"{self.name} {self.group} {self.subgroup}"

if __name__ == "__main__":
    student1 = Student('A', 102, 1)
    student2 = Student('B', 102, 2)
    student3 = Student('C', 102, 3)

    lst = MyLinkedList()
    lst.addAtHead(student1)
    lst.addAtTail(student2)
    lst.addAtTail(student3)
    lst.printlist()

    print(next(lst))
    print(next(lst))
    print(next(lst))


# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)