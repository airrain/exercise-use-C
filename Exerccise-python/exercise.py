"""元素为hashtable"""
def dedupe(items):
    seen=set()
    for item in items:
        if item not in seen:
            yield item
            seen.add(item)
"""元素不为hashtable"""
def depute(items,key=None):
    seen = set()
    for item in items:
        val = item if key is None else key(val)
        if val not in seen:
                yield val 
                seen.add(val)

from collections import Counter
words = 'abcda'
word_counts = Counter.(words)
top_three = word_counts.most_common(3)
print(top_three)

from operator import itemgetter
rows = [1,2,3]
rows_by_fname = sorted(rows,key = itemgetter('fname'))
rows_by_uid = sorted(rows,key = itemgetter('uid'))
print(rows_by_fname)
print(rows_by_uid)

class Node:
        def __init__(self,val):
                self.val = val
                self.children = []
'''快乐数字'''
def getSumOfSquares(num):
        numStr = str(num)
        sum = 0
        digitls = [int(x) for x in numStr]
        for i in digitls:
                sum += i ** 2
        return sum
def main():
        n = input()
        sumofSqrs = eval(n)
        count = 0
        while sumofSqrs != 1:
                sumofSqrs = getSumOfSquares(sumofSqrs)
                count += 1
                if count > 2000:
                        print("False")
                        break
        else:
                print("True")
main()
        
def f(x):
        return x * x

import hashlib 
md5 = hashlib.md5()
md5.update()
