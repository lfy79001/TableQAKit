class Stack(object):
    def __init__(self):
        self.list = []
    def isEmpty(self):
        return self.list == []
    def push(self, item):
        self.list.append(item)
    def pop(self):
        return self.list.pop()
    def top(self):
        return self.list[len(self.list)-1]
    def size(self):
        return len(self.list)

def pretomid(x):
    s = Stack()
    list = x.split()
    for par in list:
        if par in "+-*/":
            s.push(par)
        else: 
            if s.top() in '+-*/':
                s.push(par)
            else:
                while (not s.isEmpty()) and (not s.top() in '+-*/'):
                    shu = s.pop()
                    fu = s.pop()
                    par = '( '+shu+fu+par+' )'
                s.push(str(par))

    answer = s.pop()
    answer = answer.replace('(','').replace(')','').replace('+',' + ').replace('*',' * ')
    return answer#
