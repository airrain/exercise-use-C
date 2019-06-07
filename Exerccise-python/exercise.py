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
word_counts = Counter.(words)
top_three = word_counts.most_common(3)
print(top_three)

from operator import itemgetter
rows_by_fname = sorted(rows,key = itemgetter('fname'))
rows_by_uid = sorted(rows,key = itemgetter('uid'))
print(rows_by_fname)
print(rows_by_uid)
