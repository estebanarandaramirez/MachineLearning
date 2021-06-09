class Node:
  def __init__(self, val, left=None, right=None):
    self.val = val
    self.left = left
    self.right = right

  def BFS(self):
    queue = [self]
    while queue:
      head = queue.pop(0)
      print(head.val)
      if head.left:
        queue.append(head.left)
      if head.right:
        queue.append(head.right)
    return

  def inorder(self):
    if self.left:
      self.left.inorder()
    print(self.val)
    if self.right:
      self.right.inorder()
    return

  def preorder(self):
    print(self.val)
    if self.left:
      self.left.preorder()
    if self.right:
      self.right.preorder()
    return

  def postorder(self):
    if self.left:
      self.left.postorder()
    if self.right:
      self.right.postorder()
    print(self.val)
    return

node6 = Node(6)
node5 = Node(5)
node4 = Node(4)
node3 = Node(3, node5, node6)
node2 = Node(2, node4)
node1 = Node(1, node2, node3)
print("BFS:")
node1.BFS()
print("Inorder:")
node1.inorder()
print("Preorder:")
node1.preorder()
print("Postorder:")
node1.postorder()
print("\n")
import collections
dict = {1:"Esteban", 3:"Erik", 7:"Zebra", 2:"Ara"}
for ele in dict.items():
  print(ele)
print("\n")
sortedDict1 = sorted(dict.items(), key=lambda x: x[0], reverse=False)
for ele in sortedDict1:
  print(ele)
print("\n")
sortedDict2 = sorted(dict.items(), key=lambda x: x[1], reverse=False)
for ele in sortedDict2:
  print(ele)
print("\n")
sortedDict3 = sorted(dict.keys(), key=lambda x: x, reverse=False)
for ele in sortedDict3:
  print(ele)
print("\n")
sortedDict4 = sorted(dict.values(), key=lambda x: x, reverse=False)
for ele in sortedDict4:
  print(ele)
print("\n")
int_dict = collections.defaultdict(int)
for i in range(10):
  int_dict[i] += 1
  print(i, int_dict[i])
int_dict.pop(1)
print(len(int_dict))
print(int_dict.get(1, None))
print("\n")
import re
regex = re.compile('[^a-zA-Z ]')
string = 'string8 con numeros1'
new = regex.sub('',string)
print(new)
match = re.findall(regex, string)
print(match)

import random
from datetime import date
random.seed(date.today())
print(random.random())
print(random.uniform(0,1))
print(random.randint(0,10))
print("\n")
l1 = ["Zebra", "Esteban", "Erik", "Karla", "Manon"]
l1.sort(reverse=True)
print(l1)