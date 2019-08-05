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


import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0,2 * np.pi,num = 100)
y = np.sin(x)
plt.plot(x,y)

import numpy as np
import matplotlib.pyplot as plt
n_person = 2000
n_times = 500
t = np.arange(n_times)
steps = 2 * np.random.randint(0,1,(n_person,n_times)) - 1
amount = np.cumsum(steps,axis = 1)
std_amount = amount ** 2
mean_sd_amount = std_amount.mean(axis = 0)
plt.xlabel(r"$t$")
plt.ylabel(r"$\sqrt{\langle (\delta x)^2 \rangle}$")
plt.plot(t, np.sqrt(mean_sd_amount), 'g.', t, np.sqrt(t), 'r-')

import pandas as pd
s = pd.Series(2,4,6,7)

from sklearn import datasets
digits = datasets.load_digits()
images_and_labels = list(zip(digits.images,digits.target))
plt.figure(figsize = (8,6),dpi = 200)
for index,(image,label) in enumerate(images_and_labels[:8]):
    plt.subplot(2,4,index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Digit:%i' % label,fontsize = 20)
from sklearn.model_selection import train_test_split 
Xtrain, Xtest, Ytrain, Ytest = train_test_split(digits.data, digits.target, test_size=0.20, random_state=2)
from sklearn import svm
clf = svm.SVC(gamma = 0.001,C = 100.,probability = True)
clf.fit(Xtrain,Ytrain)
clf.score(Xtest,Ytest)
