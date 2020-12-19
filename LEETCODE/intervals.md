# Intervals

+ [Non overlapping intervals](#non-overlapping-intervals)
+ [Merge intervals](#merge-intervals)
+ [Insert interval](#insert-interval)

## Non-overlapping Intervals

+ [Non-overlapping Intervals](#non-overlapping-interval)

https://leetcode.com/problems/non-overlapping-intervals

``` python
def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
    intervals = sorted(intervals)
    prev = float("-inf")
    count = 0
    for i in intervals:
        if i[0] >= prev:
            prev = i[1]
        else:
            count +=1
            prev = min(prev,i[1])
    return count
```

## Merge Intervals

+ [Merge Intervals](#merge-intervals)

https://leetcode.com/problems/merge-intervals/

``` python
def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    intervals = sorted(intervals, key = lambda x: x[0])
    merge = []
    for i in intervals:
        if not merge or merge[-1][1]<i[0]:
            merge.append(i)
        else:
            merge[-1][1] = max(merge[-1][1],i[1])
    return merge
```

## Insert Interval

+ [Insert Interval](#insert-interval)

https://leetcode.com/problems/insert-interval/

``` python
def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    i = 0
    intervals = sorted(intervals)
    while i < len(intervals) and intervals[i][0] < newInterval[0]:
        i+=1
    intervals.insert(i,newInterval)
    merge = []
    for i in intervals:
        if not merge or merge[-1][1] < i[0]:
            merge.append(i)
        else:
            merge[-1][1] = max(merge[-1][1],i[1])
    return merge
```