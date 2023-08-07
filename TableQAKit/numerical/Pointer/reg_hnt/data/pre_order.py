class Stack:
    def __init__(self):
        self.items = []
    def is_empty(self):
        return self.items == []
 
    def push(self, item):
        self.items.append(item)
 
    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items) - 1]
    def size(self):
        return len(self.items)
 
 
def infix_to_prefix(infix_expr):
    prec = dict()
    prec[")"] = 4
    prec["*"] = 3
    prec["/"] = 3
    prec["+"] = 2
    prec["-"] = 2
    prec["("] = 1
    prefix_expr = []
    s = Stack()
    for item in reversed(infix_expr.split()):
        if item not in prec.keys():
            prefix_expr.append(item)
        elif item == ')':
            s.push(item)
        elif item == '(':
            while s.peek() != ')':
                prefix_expr.append(s.pop())
            s.pop()
        else:
            while (not s.is_empty())\
                    and s.peek() != ')'\
                    and prec[s.peek()] > prec[item]:
                prefix_expr.append(s.pop())
                s.push(item)
            s.push(item)
    while not s.is_empty():
        prefix_expr.append(s.pop())
    prefix_expr.reverse()
    return ' '.join(prefix_expr)
 
 
def prefix_eval(prefix_expr):
    s = Stack()
    for item in reversed(prefix_expr.split()):
        if item not in '+-*/':
            s.push(item)
        else:
            op1 = float(s.pop())
            op2 = float(s.pop())
            result = do_match(item, op1, op2)
            s.push(result)
    return result
 
def do_match(op, op1, op2):
    if op == '+':
        return op1 + op2
    elif op == '-':
        return op1 - op2
    elif op == '*':
        return op1 * op2
    elif op == '/':
        return op1 / op2
    else:
        raise Exception('Error operation!')
    
    
