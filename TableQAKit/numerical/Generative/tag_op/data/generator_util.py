#     "CELL(":0
#     "CELL_VALUE(":1,
#     "SPAN(":2,
#     "VALUE(":3,
#     "COUNT(":4,
#     "ARGMAX(":5,
#     "ARGMIN(":6,
#     "KEY_VALUE(":7,
# #     "ROW_KEY_VALUE(":7,
# #     "COL_KEY_VALUE(":8,
#     "SUM(":8,
#     "DIFF(":9,
#     "DIV(":10,
#     "AVG(":11,
#     "CHANGE_R(":12,
#     "MULTI-SPAN(":13,
#     "TIMES(":14,
#     ")":15,
# #     "[":17,
# #     "]":18,
# } 

def get_program_mask(prog_ids, op_stack, GRAMMER_CLASS,AUX_NUM, vocab_size, target_shift):
    
    grammar_size = len(GRAMMER_CLASS.keys())
    aux_num_size = len(AUX_NUM.keys())
    max_enumerate = 10
    
    if  len(prog_ids) == 0:
        program_mask = [0]*vocab_size
        program_mask[target_shift:target_shift+grammar_size-1]= [1]*(grammar_size-1)
        id = GRAMMER_CLASS['KEY_VALUE(']
        program_mask[target_shift+id]=0
        
        return program_mask
    
    program_mask = [1] * vocab_size
    
    if len(op_stack) == 0:
        op_list = ['SUM(', "DIFF(", "DIV(", "AVG(", "CHANGE_R(", "TIMES(", "COUNT(", 'CELL(', "CELL_VALUE(", "SPAN(", "VALUE(", "MULTI-SPAN(", "ARGMAX(", "ARGMIN"]
        if prog_ids[0] in [target_shift+GRAMMER_CLASS[op] for op in op_list]:
            program_mask = [0, 1] + [0] * (vocab_size - 2) ## eos
            return program_mask
   
    if op_stack[-1][1] > max_enumerate and op_stack[-1][0] in [target_shift+ GRAMMER_CLASS[op] for op in ["COUNT(","ARGMAX(","ARGMIN(", "MULTI-SPAN(", 'AVG(']]:
        program_mask = [0]*vocab_size
        program_mask[target_shift + GRAMMER_CLASS[')']] = 1
        return program_mask

    
    if op_stack[-1][0] == target_shift+GRAMMER_CLASS["COUNT("]:
        op_list = [')', 'CELL(', 'CELL_VALUE(', 'SPAN(', "VALUE("]
        program_mask = [0] * vocab_size
        for op in op_list:
            program_mask[target_shift + GRAMMER_CLASS[op]] =1
        return program_mask
    
    if op_stack[-1][0] in [ GRAMMER_CLASS[op]+ target_shift for  op in ['ARGMAX(','ARGMIN(']]: ##?
        program_mask = [0] * vocab_size
        if op_stack[-1][1] > 0:
            program_mask[target_shift + GRAMMER_CLASS[')']] = 1
        program_mask[target_shift + GRAMMER_CLASS['KEY_VALUE(']] = 1
        return program_mask
                                        
    if op_stack[-1][1] == 2:
        op_list = ['KEY_VALUE(', 'DIFF(', 'SUM(', 'TIMES(', 'DIV(', 'CELL(', 'CELL_VALUE(', 'SPAN(', 'VALUE(', 'CHANGE_R(']
        if op_stack[-1][0] in [target_shift + GRAMMER_CLASS[op] for op in op_list]:
            program_mask = [0] * vocab_size
            program_mask[target_shift + GRAMMER_CLASS[')']] = 1
            return program_mask
    
    if  op_stack[-1][0] == target_shift + GRAMMER_CLASS['CELL(']:
        program_mask = [0] * (grammar_size + aux_num_size + target_shift)  ## refine
        program_mask = program_mask + [1] * (vocab_size-len(program_mask))
        return program_mask
    
    if  op_stack[-1][0] == target_shift + GRAMMER_CLASS['SPAN(']:
        program_mask = [0] * (grammar_size + aux_num_size + target_shift)  ## refine
        program_mask = program_mask + [1] * (vocab_size-len(program_mask))
        return program_mask
    if op_stack[-1][0] == target_shift + GRAMMER_CLASS['KEY_VALUE(']:
        if op_stack[-1][1] == 0:
            program_mask = [0] * vocab_size
            program_mask[ target_shift + GRAMMER_CLASS['CELL(']] = 1
            program_mask[ target_shift + GRAMMER_CLASS['SPAN(']] = 1
            return program_mask
        else:
            program_mask = [0] * vocab_size
            program_mask[ target_shift + GRAMMER_CLASS['CELL_VALUE(']]=1
            program_mask[ target_shift + GRAMMER_CLASS['VALUE(']] = 1
            return program_mask
                                        
    if op_stack[-1][0] ==  target_shift + GRAMMER_CLASS['VALUE(']:

        program_mask = [0] * (grammar_size + aux_num_size + target_shift)  ## refine
        program_mask = program_mask + [1] * (vocab_size-len(program_mask))
        return program_mask
        
    if op_stack[-1][0] == target_shift + GRAMMER_CLASS['CELL_VALUE(']:
        import pdb
#         pdb.set_trace()
        program_mask = [0] * (grammar_size + aux_num_size + target_shift)  ## refine
        program_mask = program_mask + [1] * (vocab_size-len(program_mask))
        return program_mask
    
    if op_stack[-1][0] == target_shift + GRAMMER_CLASS['MULTI-SPAN(']:
        program_mask = [0] * vocab_size
        if op_stack[-1][1] > 0:
            program_mask[target_shift + GRAMMER_CLASS[')']] = 1
        op_list = ['CELL(', "SPAN(", "CELL_VALUE(", "VALUE("]
        for op in op_list:
            program_mask[target_shift + GRAMMER_CLASS[op]] = 1
        return program_mask
    
    if op_stack[-1][0] == target_shift + GRAMMER_CLASS['AVG(']:
        if op_stack[-1][0] == 0:
            program_mask = [0] * vocab_size
            program_mask[ target_shift + GRAMMER_CLASS['VALUE(']] = 1
            program_mask[ target_shift + GRAMMER_CLASS['CELL_VALUE(']] = 1
        else:
            program_mask = [0] * vocab_size
            program_mask[ target_shift + GRAMMER_CLASS['VALUE(']] = 1
            program_mask[ target_shift + GRAMMER_CLASS['CELL_VALUE(']] = 1
            program_mask[ target_shift + GRAMMER_CLASS[')']] = 1
        return program_mask
    if  op_stack[-1][0] == target_shift + GRAMMER_CLASS['CHANGE_R(']:
        program_mask = [0] * vocab_size
        program_mask[ target_shift + GRAMMER_CLASS['VALUE(']] = 1
        program_mask[ target_shift + GRAMMER_CLASS['CELL_VALUE(']] = 1
        return program_mask
    
    op_list= ['SUM(', "DIFF(", "DIV(", "TIMES("]
    assert op_stack[-1][0] in [target_shift + GRAMMER_CLASS[op] for op in op_list]
    
    if op_stack[-1][0] in [target_shift + GRAMMER_CLASS[op] for op in op_list]:
        program_mask = [0] * vocab_size
        program_mask[target_shift + GRAMMER_CLASS['SUM(']] = 1
        program_mask[target_shift + GRAMMER_CLASS['DIFF(']] = 1
        program_mask[target_shift + GRAMMER_CLASS['TIMES(']] = 1
        program_mask[target_shift + GRAMMER_CLASS['DIV(']] = 1
        program_mask[target_shift + GRAMMER_CLASS['AVG(']] = 1
        program_mask[target_shift + GRAMMER_CLASS['CHANGE_R(']] = 1
        
        program_mask[target_shift + GRAMMER_CLASS['CELL_VALUE(']] = 1
        program_mask[target_shift + GRAMMER_CLASS['VALUE(']] = 1
        program_mask[target_shift + grammar_size: target_shift + grammar_size+ aux_num_size] = [1]*aux_num_size
        return program_mask
    if op_stack[-1][1] >= 2:
        program_mask[target_shift + GRAMMER_CLASS[')']] = 1
    return program_mask
                                        

def update_op_stack(progs,  GRAMMER_CLASS, AUX_NUM, vocab_size, target_shift):
    grammar_size = len(GRAMMER_CLASS.keys())
    aux_num_size = len(AUX_NUM.keys())
    
    op_stack = []
    for prog in progs:
        if prog == 1:
            return []
        if prog in [GRAMMER_CLASS[item]+target_shift for item in ['CELL(', 'CELL_VALUE(', 'SPAN(', 'VALUE(',
                                      'COUNT(', 'ARGMAX(', 'ARGMIN(', 'KEY_VALUE(', 'SUM(',
                                      'DIFF(', 'DIV(', 'AVG(', 'CHANGE_R(', 'MULTI-SPAN(', "TIMES(" ]]:
            op_stack.append([prog, 0])

        elif prog == GRAMMER_CLASS[')'] + target_shift:
            if op_stack[-1][0] == GRAMMER_CLASS['CELL('] + target_shift:
                op_stack.pop()
            elif op_stack[-1][0] == GRAMMER_CLASS['SPAN('] + target_shift:
                op_stack.pop()
            elif op_stack[-1][0] == GRAMMER_CLASS['CELL_VALUE('] + target_shift:
                op_stack.pop()
            elif op_stack[-1][0] == GRAMMER_CLASS['VALUE('] + target_shift:
                op_stack.pop()
            elif op_stack[-1][0] == GRAMMER_CLASS['COUNT('] + target_shift:
                op_stack.pop()
            elif op_stack[-1][0] == GRAMMER_CLASS['ARGMAX('] + target_shift:
                op_stack.pop()
            elif op_stack[-1][0] == GRAMMER_CLASS['ARGMIN('] + target_shift:
                op_stack.pop()
            elif op_stack[-1][0] == GRAMMER_CLASS['KEY_VALUE('] + target_shift:
                op_stack.pop()
            elif op_stack[-1][0] == GRAMMER_CLASS['SUM('] + target_shift:
                op_stack.pop()
            elif op_stack[-1][0] == GRAMMER_CLASS['DIFF('] + target_shift:
                op_stack.pop()
            elif op_stack[-1][0] == GRAMMER_CLASS['DIV('] + target_shift:
                op_stack.pop()
            elif op_stack[-1][0] == GRAMMER_CLASS['AVG('] + target_shift:
                op_stack.pop()
            elif op_stack[-1][0] == GRAMMER_CLASS['CHANGE_R('] + target_shift:
                op_stack.pop()
            elif op_stack[-1][0] == GRAMMER_CLASS['MULTI-SPAN('] + target_shift:
                op_stack.pop()
            elif op_stack[-1][0] == GRAMMER_CLASS['TIMES('] + target_shift:
                op_stack.pop()
            else:
                op_stack.pop()
            if len(op_stack) == 0:
                return op_stack
            else:
                op_stack[-1][1] += 1
        else:
            if op_stack[-1][0] in [ GRAMMER_CLASS[op] + target_shift for op in ['SPAN(', 'CELL(', 'VALUE(', 'CELL_VALUE(']]:
                op_stack[-1][1] +=1
            else:
                op_stack[-1][1] +=1
    return op_stack
