#test TODO1
from util import Stack

s = Stack()
s.push(1)
s.push(2)
s.push(3)
s.push(4)
s.push(5)
s.push(6)
print s.pop()
#test TODO2
from util import  Queue
s = Queue()
s.push(1)
s.push(2)
s.push(3)
s.push(4)
s.push(5)
s.push(6)
print s.pop()
#test TODO3
from util import  PriorityQueue
pq = PriorityQueue()
pq.push("a",1)
pq.push("b",2)
pq.push("c",3)
pq.push("c",-5)
print pq.heap
print pq.dict
