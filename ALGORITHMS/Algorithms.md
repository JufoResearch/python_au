# Algorithms

+ [Maximum Depth of N-ary Tree](#maximum-depth-of-n-ary-tree)
+ [Shortest Path In Binary Matrix](#shortest-path-in-binary-matrix)
+ [Is Graph Bipartite](#is-graph-bipartite)
+ [Cheapest Flights Within k Stops](#cheapest-flights-within-k-stops)
+ [Course Schedule II](#course-schedule-ii)
+ [Course Schedule](#course-schedule)
+ [Number of Islands](#number-of-islands)
+ [Implement Stack using Queues](#implement-stack-using-queues)
+ [House Robber II](#house-robber-ii)
+ [Design Twitter](#design-twitter)
+ [Min Stack](#min-stack)
+ [Implement Queue using Stacks](#implement-queue-using-stacks)
+ [House Robber](#house-robber)
+ [Merge K Sorted Lists](#merge-k-sorted-lists)
+ [K Closest Points to Origin](#k-closest-points-to-origin)
 

## Maximum Depth of N-ary Tree

https://leetcode.com/problems/maximum-depth-of-n-ary-tree/

```python
def maxDepth(self, root: 'Node') -> int:
    queue = []
    if root: queue.append((root,1))
    depth = 0
    for (node, level) in queue:
        depth = level
        queue += [(child, level+1) for child in node.children]
    return depth
```


## Shortest Path In Binary Matrix 

https://leetcode.com/problems/shortest-path-in-binary-matrix/

```python
from collections import deque
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        if grid[0][0] != 0:
            return -1
        n = len(grid)
        if n == 1:
            if grid[0][0] == 0:
                return 1
        dq = deque()
        dq.append([0, 0])
        length = 1
        while dq:
            length += 1
            for _ in range(len(dq)):
                x, y = dq.popleft()
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        if i == 0 and j == 0:
                            continue
                        if 0 <= x + i < n and 0 <= y + j < n and grid[x + i][y + j] == 0:
                            if x + i == n - 1 and y + j == n - 1:
                                return length
                            dq.append([x + i, y + j])
                            grid[x + i][y + j] = 1
        return -1
```


## Is Graph Bipartite 

https://leetcode.com/problems/is-graph-bipartite/

```python
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        n,color = len(graph),{}
        def dfs(u):
            for v in graph[u]:
                if v not in color:
                    color[v] = 1 - color[u]
                    if not dfs(v): return False
                elif color[v] == color[u]:
                    return False
            return True
        for i in range(n):
            if i not in color and graph[i]:
                color[i] = 1
                if not dfs(i): return False
        return True
```


## Cheapest Flights Within k Stops 

https://leetcode.com/problems/cheapest-flights-within-k-stops/

```python
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, K: int) -> int:
        flight_list = defaultdict(list)
        for s,d,c in flights:
            flight_list[s].append([c,d])
        search_list = [(0,src,0)]
        while(search_list):
            distance, curr, stops = heapq.heappop(search_list) 
            if curr == dst:
                return distance
            elif stops > K:
                continue
            for cost, des in flight_list[curr]: 
                heapq.heappush(search_list, (distance+cost,des, stops+1))
        return -1
```


## Course Schedule II

https://leetcode.com/problems/course-schedule-ii/

```python
from collections import deque
def dfs(self, curr, visited, adj_list, answer):
    visited[curr] = 1 
    for neighbor in adj_list[curr]:
        if neighbor not in visited:
            if self.dfs(neighbor, visited, adj_list, answer):
                return True
        elif visited[neighbor] == 1:
            return True
    visited[curr] = 2
    answer.appendleft(curr)
    return False
    
def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    adj_list = [[] for i in range(numCourses)]
    for course, pre in prerequisites:
        adj_list[pre].append(course)
    visited = {}
    answer = deque()
    for curr in range(numCourses):
        if curr not in visited:
            if self.dfs(curr, visited, adj_list, answer):
                return []
    return answer
```


## Course Schedule

https://leetcode.com/problems/course-schedule/

```python
from collections import deque
def dfs(self, curr, visited, adj_list, answer):
    visited[curr] = 1 
    for neighbor in adj_list[curr]:
        if neighbor not in visited:
            if self.dfs(neighbor, visited, adj_list, answer):
                return True
        elif visited[neighbor] == 1:
            return True
    visited[curr] = 2
    answer.appendleft(curr)
    return False
    
def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    adj_list = [[] for i in range(numCourses)]
    for course, pre in prerequisites:
        adj_list[pre].append(course)
    visited = {}
    answer = deque()
    for curr in range(numCourses):
        if curr not in visited:
            if self.dfs(curr, visited, adj_list, answer):
                return []
    if answer == []:
        return 0
    else:
        return 1
```


## Number of Islands

https://leetcode.com/problems/number-of-islands/

```python
def dfs(self, grid, row, col, n):
    if  0 <= row < len(grid) and 0 <= col < n and grid[row][col] == "1":
        grid[row][col] = "0"
        for x, y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            self.dfs(grid, row + x , col + y, n)
        
def numIslands(self, grid: List[List[str]]) -> int:
    counter = 0
    m = len(grid)
    n = len(grid[0])
    for row in range(m):
        for col in range(n):
            if grid[row][col] == '1':
                self.dfs(grid, row, col, n)
                counter += 1
    return counter 
```


## Implement Stack using Queues

https://leetcode.com/problems/implement-stack-using-queues/

```python
def __init__(self):
    """
    Initialize your data structure here.
    """
    self.q1 = deque()
    self.q2 = deque()
    self._top = None
def push(self, x: int) -> None:
    """
    Push element x onto stack.
    """
    self.q1.append(x)
    self._top = x
def pop(self) -> int:
    """
    Removes the element on top of the stack and returns that element.
    """
    while len(self.q1) > 1:
        self._top = self.q1.popleft()
        self.q2.append(self._top)
    temp = self.q1.popleft()
    self.q1, self.q2 = self.q2, self.q1
    return temp
def top(self) -> int:
    """
    Get the top element.
    """
    return self._top
def empty(self) -> bool:
    """
    Returns whether the stack is empty.
    """
    return len(self.q1) == 0
```


## House Robber II

https://leetcode.com/problems/house-robber-ii/

```python
def rob_1(self, nums: List[int]) -> int:
    length = len(nums)
    if length == 0:
        return 0
    elif length == 1:
        return nums[0]
    elif length == 2:
        return max(nums)
    else:
        summ = [0]*length # assign dp array
        summ[0], summ[1] = nums[0], max(nums[0], nums[1])
        for i in range(2, length):
            summ[i] = max(summ[i-1], summ[i-2]+nums[i])
        return max(summ)
def rob(self, nums: List[int]) -> int:
    if not nums:
        return 0
    elif len(nums) == 1:
        return nums[0]
    else:
        return max(self.rob_1(nums[1:]), self.rob_1(nums[:-1]))
```


## Design Twitter

https://leetcode.com/problems/design-twitter/

```python
def __init__(self):
    """
    Initialize your data structure here.
    """
    self.users = {}
    self.followers = {}
    self.post = 0
    
def postTweet(self, userId, tweetId):
    """
    Compose a new tweet.
    :type userId: int
    :type tweetId: int
    :rtype: None
    """
    self.post += 1
    if userId in self.users:
        self.users[userId].append((tweetId, self.post))
    else: self.users[userId] = [(tweetId,self.post)]
    
def get_tweet(self,userId):
    tweets = []
    if userId in self.users:
        tweets += self.users[userId]
    if userId in self.followers:
        followees = self.followers[userId]
        for u_id in followees:
            if u_id in self.users:
                tweets += self.users[u_id]
    tweets = sorted(tweets, key = lambda posts: posts[1],reverse=True)
    return tweets[0:10]
def getNewsFeed(self, userId):
    """
    Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
    :type userId: int
    :rtype: List[int]
    """
    recent_Tweets = self.get_tweet(userId)
    return [post[0] for post in recent_Tweets] 
def follow(self, followerId, followeeId):
    """
    Follower follows a followee. If the operation is invalid, it should be a no-op.
    :type followerId: int
    :type followeeId: int
    :rtype: None
    """
    if followerId != followeeId:
        if followerId in self.followers :
            self.followers[followerId].add(followeeId)
        else: self.followers[followerId] = {followeeId}
def unfollow(self, followerId, followeeId):
    """
    Follower unfollows a followee. If the operation is invalid, it should be a no-op.
    :type followerId: int
    :type followeeId: int
    :rtype: None
    """
    if followerId in self.followers:
        if followeeId in self.followers[followerId]:
            self.followers[followerId].remove(followeeId)
```


## Min Stack

https://leetcode.com/problems/min-stack/

```python
class MinStack:
    def __init__(self):
        self.stack = []
    def push(self, val: int) -> None:
        if self.stack:
            if val <= self.stack[-1][1]:
                self.stack.append((val,val))
            else:
                self.stack.append((val,(self.stack[-1][1])))
        else:
            self.stack.append((val,val))
        
    def pop(self) -> None:
        self.stack.pop()
    def top(self) -> int:
        return self.stack[-1][0]
    def getMin(self) -> int:
        return self.stack[-1][1]
```


## Implement Queue using Stacks

https://leetcode.com/problems/implement-queue-using-stacks/

```python
def __init__(self):
    self.queue = []
def push(self, x: int) -> None:
    self.queue.append(x)
def pop(self) -> int:
    new = self.queue[0]
    self.queue.pop(0)
    return new
def peek(self) -> int:
    return self.queue[0]
def empty(self) -> bool:
    if self.queue == []:
        return True
    else:
        return False
```


## House Robber

https://leetcode.com/problems/house-robber/ 

```python
def rob(self, nums: List[int]) -> int:
    rob1, rob2 = 0, 0
    for num in nums:
        newrob = max(rob1+num, rob2)
        rob1 = rob2
        rob2 = newrob
    return rob2
```


## Merge K Sorted Lists

https://leetcode.com/problems/merge-k-sorted-lists/

```python
def mergeKLists(self, lists: List[ListNode]) -> ListNode:
    if not lists:
        return None
    if len(lists) == 1:
        return lists[0]
    mid = len(lists) // 2
    l = self.mergeKLists(lists[:mid]), 
    r = self.mergeKLists(lists[mid:])
    return self.merge(l, r)
    
def merge(self, l, r):
    prev = head = ListNode()
    while l and r:
        if l.val < r.val:
            head.next = l
            l = l.next
        else:
            head.next = r
            r = r.next
        head = head.next
    head.next = l or r
    return prev.next
```


## K Closest Points to Origin

https://leetcode.com/problems/k-closest-points-to-origin/

```python
def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
    points.sort(key=lambda x: pow((x[0]**2 + x[1]**2), 0.5))
    return points[:k]
```