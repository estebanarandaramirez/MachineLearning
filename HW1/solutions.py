class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        if not nums:
            return None  
        unique = set(nums)
        for num in unique:
            if nums.count(num) == 1:
                return num

    def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
        for i in range(len(matrix)):
            r, c = i+1, 1
            while r < len(matrix) and c < len(matrix[0]):
                if matrix[r][c] != matrix[i][0]:
                    return False
                r += 1
                c += 1
        for i in range(1, len(matrix[0])):
            r, c = 1, i+1
            while r < len(matrix) and c < len(matrix[0]):
                if matrix[r][c] != matrix[0][i]:
                    return False
                r += 1
                c += 1
        return True

    def minDominoRotations(self, A: List[int], B: List[int]) -> int:
        A1=Counter(A)
        B1=Counter(B)
        maxval_A,maxval_B=max(A1.values()),max(B1.values())
        if maxval_A<maxval_B:
            A,B=B,A
            A1,B1=B1,A1
        maxkey=max(A1, key=A1.get)
        count=0
        for ind,el in enumerate(A):
            if el!=maxkey:
                A[ind],B[ind]=B[ind],A[ind]
                count+=1
        if len(set(A))==1:
            return count
        return -1

    def time_taken(keyboard, text):
        """
        Strategy: map letter to key index. for each char in text, lookup char in map and get abs val diff
        to curr position, add to res. update curr position.
        """
        res, curr = 0, 0
        map = {char: i for i,char in enumerate(keyboard)}
        for char in text:
            diff = abs(map[char] - curr)
            res += diff
            curr = map[char]
        return res

    def maxLevelSum(self, root: TreeNode) -> int:
        sum = [0]*100
        def traverse(root,level):
            sum[level] += root.val
            if root.left:
                traverse(root.left,level+1)
            if root.right:
                traverse(root.right,level+1)    
        traverse(root,0)
        return sum.index(max(sum))+1

    def minNumChairs(starts, ends):
        all = [(s, 1) for s in starts] + [(e, -1) for e in ends]
        all = sorted(all)
        num = 0
        largest = 0
        for pos, t in all:
            num += t
            if largest < num:
                largest = num
        return largest

    def kClosest(self, points, K):
        points.sort(key = lambda P: P[0]**2 + P[1]**2)
        return points[:K]

    def licenseKeyFormatting(self, S, K):
        """
        :type S: str
        :type K: int
        :rtype: str
        """
        S = S.replace("-", "").upper()[::-1]
        return '-'.join(S[i:i+K] for i in range(0, len(S), K))[::-1]

    def numUniqueEmails(self, emails):
        seen = set()
        for email in emails:
            local, domain = email.split('@')
            if '+' in local:
                local = local[:local.index('+')]
            seen.add(local.replace('.','') + '@' + domain)
        return len(seen)

    def totalFruit(self, tree):
        ans = i = 0
        count = collections.Counter()
        for j, x in enumerate(tree):
            count[x] += 1
            while len(count) >= 3:
                count[tree[i]] -= 1
                if count[tree[i]] == 0:
                    del count[tree[i]]
                i += 1
            ans = max(ans, j - i + 1)
        return ans

    def nearest_stores(houses, stores):
        houses = [(houses[i], i) for i in range(len(houses))]
        houses.sort(key = lambda x: x[0])
        stores.sort()
        j = 0
        output = [0 for i in range(len(houses))]
        for i in range(len(houses)):
            min_dist = float('inf')
            while j < len(stores) and abs(stores[j] - houses[i][0]) < min_dist:
                min_dist = abs(stores[j] - houses[i][0])
                j+=1
            j-=1
            output[houses[i][1]] = stores[j]
        return output

'''
    GOOGLE PREP

    sortedDict = sorted(dict.items(), key=lambda x: x[1], reverse=True) -> reverse=True (descending)
    dict = collections.defaultdict(int)
    max_value = max(dict.values)
    -----------------------------------------------------
    print(("Hola putos como estan.".replace('.','')).split(' '))
    [char for char in word]
    -----------------------------------------------------
    str.isalpha()
    str.isnumeric()
    str.isalnum()
    str.lower() // converto to lowercase
    str.upper() // convert to uppercase
    ''.join(list)
    str.replace(',', '')
    str.split()
    STRINGS ARE INMUTABLE
    -----------------------------------------------------
    print("Price: %d, Tax: %5.2f" % (price, tax))
    -----------------------------------------------------
    REGEXP: For IP address \b\d{1,3}\.\d{1,3}\.\d{1,3}\b       (import re)
    regex = re.compile('[^a-zA-Z]')
    regex.sub('','string8 con numeros1')
    re.search(pattern, string)

    Grep - command line tool that helps with regexp
    -----------------------------------------------------
    ENUMERATION
    from enum import Enum:

    class Values(Enum):
    unknown = -1
    empty = 0
    full = 1
    -----------------------------------------------------
    STRUCT
    from typing import NamedTuple

    class Node(NamedTuple): 	
    node: int
    numParents: int
    parents: str
    value: Values
    probabilities: list
    -----------------------------------------------------
    CLASS
    class Dog:
    kind = 'corgi'

    def __init__(self, name):
        self.name = name

    doggo = Dog('Pui')
    -----------------------------------------------------
    MAIN
    def main():
    ...

    if __name__ == '__main__':
    main()
    -----------------------------------------------------
    FUNCTION DEFINITION
    def sum(num1, num2):
    ...
    -----------------------------------------------------
    yield -> like return, but saves enough memory to return to execution of function where it left off
    -----------------------------------------------------
    RECURSION
    def permutation(lst):  
        if not lst: 
            return [] 
    
        if len(lst) == 1: 
            return [lst] 
    
        l = [] # empty list that will store current permutation 
    
        # Iterate the input(lst) and calculate the permutation 
        for i in range(len(lst)): 
        m = lst[i] 
    
        # Extract lst[i] or m from the list.  remLst is 
        # remaining list 
        remLst = lst[:]    ( ORIGINAL: remLst = lst[:i] + lst[i+1:] ) 
        remLst.pop(i)
    
        # Generating all permutations where m is first 
        # element 
        for p in permutation(remLst): 
            l.append([m] + p) 
        return l 
    
    data = list('123') 
    for p in permutation(data): 
        print(p)
    -----------------------------------------------------
    ALGORITHM BEHIND HASHTABLES
    Utilizes a hash function on the key to store the value on a specific index.
    For example, use the modulo of the key with the size of array to get the index values.
                    Index
    Key 1:   1 % 20 = 1
    Key 2:   2 % 20 = 2
    Key 48: 48 % 20 = 8
    Then use linear probing (search adjacent index spaces) if a certain index is already filled in.
    -----------------------------------------------------
    IMPORTS
    import random
    random.seed(datetime.now())
    random.random()
    random.uniform(0,1)

    import math
    math.inf
    math.floor
    math.log10

    >>> from collections import Counter
    >>> words = "if there was there was but if \
    ... there was not there was not".split()
    >>> counts = Counter(words)
    >>> counts
    Counter({'if': 2, 'there': 4, 'was': 4, 'not': 2, 'but': 1})

    >>> import itertools
    >>> friends = ['Monique', 'Ashish', 'Devon', 'Bernie']
    >>> list(itertools.permutations(friends, r=2))
    [('Monique', 'Ashish'), ('Monique', 'Devon'), ('Monique', 'Bernie'),
    ('Ashish', 'Monique'), ('Ashish', 'Devon'), ('Ashish', 'Bernie'),
    ('Devon', 'Monique'), ('Devon', 'Ashish'), ('Devon', 'Bernie'),
    ('Bernie', 'Monique'), ('Bernie', 'Ashish'), ('Bernie', 'Devon')]
    >>> list(itertools.combinations(friends, r=2))
    [('Monique', 'Ashish'), ('Monique', 'Devon'), ('Monique', 'Bernie'),
    ('Ashish', 'Devon'), ('Ashish', 'Bernie'), ('Devon', 'Bernie')]

    import numpy as np
    import pandas as pd
    import csv

    import string
    string.ascii_letters
    string.ascii_uppercase
    -----------------------------------------------------
    DATA STRUCTURES:
    LIST - [1,2,3,4] built-in in Python
        list.append(element)
        list.pop() or list.pop(index) or del list[index] or list.remove(value)
        list.insert(index, element)
        list.sort(key=..., reverse=...) or sorted(list, key=..., reverse=...)
        if not list: -> check for an empty list
        value in list -> check for an element in a list
        for i in range(len(list)):
        for ele in list:
        for i, ele in enumerate(len(list)):
        [square(x) for x in list]
        [x for x in list if is_odd(x)]
        sum([i * i for i in range(1, 1001)])
        newlist = list1 + list2
        list[:index] -> return sublist with all elements with exclusive upper bound
        list[index:] -> return sublist with all elements with inclusive lower bound
        list[0:len(list)] or list[:] -> entire list
        
    ARRAY - array([2,4,6]) like lists but can perform arithmetic functions on all elements at once. Example a/2 = array[1,2,3]

    STACKS (LIFO - last in, first out)[DFS] - Use a normal python list. append() adds an item to the top and pop() removes it

    QUEUE (FIFO - first in, first out)[BFS] - Use a normal python list. pop(0) to remove first item in the list
                        Or use deque (from collections import deque) (deque.appendleft(), deque.popleft())

    SET - Cannot repeat elements, no specific order
        words = set()
        words.add(word)
        words.remove(word)
        words.dicard(word)
        words.clear()
        'hola' in words (returns true or false)
        'hola' not in words
        len(s)

    DICTIONARY (HASHTABLE) - dict = {} or dict = {1:'esteban, 2:'karla'} or dict([(1,'esteban'), (2,'karla')])
                            dict[1] = 'esteban'
                            dict[1] = 'karla' -> change value associated with key
                            dict[1] or dict.get[1] to get value stored in key
                            dict[1] = dict[1] + 1
                            dict.pop[1] or del dict[1]
                            if not dict: -> check for empty dict
                            for key in dict.keys()
                            for value in dict.values()
                            for key, value in dict.items()
                            sorted(dict) -> sorts the dict based on the keys
                            list(dict) -> list of keys in dict
                            len(dict)
                            1 in dict:  or  1 not in dict.keys(): (return true or false)
                            if 'esteban' in dict.values():
                            if not dict.get(1): -> if no value inside, no key
    -----------------------------------------------------
    TREE 
    class Node:
    def __init__(self, value):
        self.left = None
        self.data = value
        self.right = None
            
    def __eq__(self, other):
        return (self.val == other)

    class Tree:
    ...

    Traversals:
    (a) Inorder (Left, Root, Right)
    (b) Preorder (Root, Left, Right)
    (c) Postorder (Left, Right, Root)
    -----------------------------------------------------
    GRAPH can be implemented as a dictionary
    -----------------------------------------------------
    ALGORITHMS
    QUICKSORT - Pick a random pivot, pick a low and high bound. Move the bounds and reodrder the elements so that smaller ones
                go to the left of the pivot and bigger ones to the right. Call recursively.
    BINARY SORT - Given a sorted list, start in middle see if it is bigger or smaller and repeat
    MERGESORT - Divides input list into two halves, and calls itself until it cant divide anymore then it merges the two sorted halves.
    DFS(LIFO, stack) - not complete nor optimal. IDDFS. O(Vertices + Edges)0
    BFS(FIFO, queue) - O(Vertices + Edges)
    BINARY TREE - each node has at most two children. Balanced if the subtrees do not differ by more than length 1. 

    NP-complete - Nondeterministic polynomial time. Brute force search. "Yes" if solution is non-empty, "no" if solution is empty.
    Travelling salesman - "Given a list of cities and the distances between each pair of cities, 
                        what is the shortest possible route that visits each city and returns to the origin city?"
    -----------------------------------------------------
    Symmetric Tree

    class Solution:
        def isSymmetric(self, root: TreeNode) -> bool:
            if root == None:
                return True
            return self.isMirror(root.left, root.right)
            
        def isMirror(self, t1: TreeNode, t2: TreeNode):
            if t1 == None and t2 == None: return True
            if t1 == None or t2 == None: return False
            return t1.val == t2.val and self.isMirror(t1.left, t2.right) and self.isMirror(t1.right, t2.left)
    -----------------------------------------------------
    Validity of BST

    class Solution:
        def isValidBST(self, root):
            """
            :type root: TreeNode
            :rtype: bool
            """
            def helper(node, lower = float('-inf'), upper = float('inf')):
                if not node:
                    return True
                
                val = node.val
                if val <= lower or val >= upper:
                    return False

                if not helper(node.right, val, upper):
                    return False
                if not helper(node.left, lower, val):
                    return False
                return True

            return helper(root)
'''