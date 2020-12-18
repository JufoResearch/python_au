# Tree

+ [Maximum Depth of Binary Tree](#maximum-depth-of-binary-tree)
+ [Binary Tree Inorder Traversal](#binary-tree-inorder-traversal)
+ [Invert Binary Tree](#invert-binary-tree)
+ [Binary Search Tree Iterator](#binary-search-tree-iterator)
+ [Binary Tree Level Order Traversal](#binary-tree-level-order-traversal)
+ [Kth Smallest Element in a BST](#kth-smallest-element-in-a-bst)
+ [Validate Binary Search Tree](#validate-binary-search-tree)
+ [Symmetric Tree](#symmetric-tree)

## Maximum Depth of Binary Tree

+ [Maximum Depth of Binary Tree](#maximum-depth-of-binary-tree)

https://leetcode.com/problems/maximum-depth-of-binary-tree/

``` python
ass TreeNode:
  def __init__(self, val=0, left=None, right=None):
      self.val = val
      self.left = left
      self.right = right
s Solution:
def maxDepth(self, root: TreeNode) -> int:
```

## Binary Tree Inorder Traversal

+ [Binary Tree Inorder Traversal](#binary-tree-inorder-traversal)

https://leetcode.com/problems/binary-tree-inorder-traversal/

``` python
ass TreeNode:
  def __init__(self, val=0, left=None, right=None):
      self.val = val
      self.left = left
      self.right = right
s Solution:
def inorderTraversal(self, root: TreeNode) -> List[int]:
```

## Invert Binary Tree

+ [Invert Binary Tree](#invert-binary-tree)

https://leetcode.com/problems/invert-binary-tree/

``` python
ass TreeNode:
  def __init__(self, val=0, left=None, right=None):
      self.val = val
      self.left = left
      self.right = right
s Solution:
def invertTree(self, root: TreeNode) -> TreeNode:
```

## Binary Search Tree Iterator

+ [Binary Search Tree Iterator](#binary-search-tree-iterator)

https://leetcode.com/problems/binary-search-tree-iterator/

``` python
ass TreeNode:
  def __init__(self, val=0, left=None, right=None):
      self.val = val
      self.left = left
      self.right = right
s BSTIterator:

def __init__(self, root: TreeNode):


def next(self) -> int:


def hasNext(self) -> bool:



ur BSTIterator object will be instantiated and called as such:
j = BSTIterator(root)
ram_1 = obj.next()
ram_2 = obj.hasNext()
```

## Binary Tree Level Order Traversal

+ [Binary Tree Level Order Traversal](#binary-tree-level-order-traversal)

https://leetcode.com/problems/binary-tree-level-order-traversal/

``` python
ass TreeNode:
  def __init__(self, val=0, left=None, right=None):
      self.val = val
      self.left = left
      self.right = right
s Solution:
def levelOrder(self, root: TreeNode) -> List[List[int]]:
```

## Kth Smallest Element in a BST

+ [Kth Smallest Element in a BST](#kth-smallest-element-in-a-bst)

https://leetcode.com/problems/kth-smallest-element-in-a-bst/

``` python
ass TreeNode:
  def __init__(self, val=0, left=None, right=None):
      self.val = val
      self.left = left
      self.right = right
s Solution:
def kthSmallest(self, root: TreeNode, k: int) -> int:
```

## Validate Binary Search Tree

+ [Validate Binary Search Tree](#validate-binary-search-tree)

https://leetcode.com/problems/validate-binary-search-tree/

``` python
ass TreeNode:
  def __init__(self, val=0, left=None, right=None):
      self.val = val
      self.left = left
      self.right = right
s Solution:
def isValidBST(self, root: TreeNode) -> bool:
```

## Symmetric Tree

+ [Symmetric Tree](#symmetric-tree)

https://leetcode.com/problems/symmetric-tree/

``` python
ass TreeNode:
  def __init__(self, val=0, left=None, right=None):
      self.val = val
      self.left = left
      self.right = right
s Solution:
def isSymmetric(self, root: TreeNode) -> bool:
```