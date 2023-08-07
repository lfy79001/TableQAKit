#     "CELL(":0
#     "CELL_VALUE(":1,
#     "SPAN(":2,
#     "VALUE(":3,
#     "COUNT(":4,
#     "ARGMAX(":5,
#     "ARGMIN(":6,
#     "KEY_VALUE(":7,
#     "SUM(":8,
#     "DIFF(":9,
#     "DIV(":10,
#     "AVG(":11,
#     "CHANGE_R(":12,
#     "MULTI-SPAN(":13,
#     "TIMES(":14,
#     ")":15,
# } 

import numpy as np

def get_program_mask(prog_ids, op_stack, GRAMMER_CLASS, AUX_NUM, vocab_size, question_id, table_mask, paragraph_mask, table_cell_number_value, paragraph_number_value, table_cell_index, paragraph_index, row_include_cells, col_include_cells, target_shift):
    
    
    assert len(table_cell_index) ==512
    assert len(paragraph_index) == 512
    
    grammar_size = len(GRAMMER_CLASS.keys())
    aux_num_size = len(AUX_NUM.keys())
    max_enumerate = 10
    op_size = grammar_size + aux_num_size + target_shift
    atom_op_list = ['CELL(', "CELL_VALUE(", "SPAN(", "VALUE("] ## atom operations
            
    
    if  len(prog_ids) == 0:
        program_mask = [0]*vocab_size
        program_mask[target_shift:target_shift+grammar_size-1]= [1]*(grammar_size-1)
        id = GRAMMER_CLASS['KEY_VALUE(']
        program_mask[target_shift+id]=0
        
        return program_mask
    
    if len(op_stack) == 0:
        op_list = ['SUM(', "DIFF(", "DIV(", "AVG(", "CHANGE_R(", "TIMES(", "COUNT(", 'CELL(', "CELL_VALUE(", "SPAN(", "VALUE(", "MULTI-SPAN(", "ARGMAX(", "ARGMIN("]
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
        import pdb
        
        program_mask = [0]*(op_size)
        program_mask += [1]* (vocab_size - len(program_mask))
        op_list = ['COUNT(', 'MULTI-SPAN('] ## avoid repetition
        if prog_ids[0] in [target_shift + GRAMMER_CLASS[op] for op in op_list]:
            import pdb
            prog_idx = 0
            while prog_idx < len(prog_ids) - 3:
                while prog_idx < len(prog_ids) - 3 and prog_ids[prog_idx] not in [target_shift + GRAMMER_CLASS[op] for op in atom_op_list]:
                    prog_idx += 1
                if prog_idx >= len(prog_ids) - 3:
                    break
                span_st = prog_ids[prog_idx + 1]
                span_ed = prog_ids[prog_idx + 2]
                for i in range(span_st, span_ed + 1):
                    program_mask[i] = 0
                prog_idx += 4
        
        table_start = -1
        try:
            table_start = table_mask.index(1)
            table_end = table_start + int(sum(table_mask))
        except:
            table_start = -1
            
        if op_stack[-1][1] == 0:
            current_program_mask = [0] * vocab_size
            if table_start!=-1:  ## all the tokens in table are set to 1
                current_program_mask[op_size+table_start:op_size+table_end] = [1] * (table_end - table_start)
            return [a*b for a, b in zip(program_mask, current_program_mask)]
        
        else:
            prev_prog_id = prog_ids[-1]
            current_program_mask = [0] * (prev_prog_id)

            real_prog_id = prev_prog_id - op_size
            table_id = table_cell_index[real_prog_id]
            reversed_table_cell_index = list(reversed(table_cell_index))
            table_end = reversed_table_cell_index.index(table_id)
            table_end = len(reversed_table_cell_index)- table_end ## not include end index
#             pdb.set_trace()
            current_program_mask += [1] * (table_end+op_size - prev_prog_id)
            current_program_mask += [0]*(vocab_size - len(current_program_mask))
            return [a*b for a, b in zip(program_mask, current_program_mask)]
        
    if  op_stack[-1][0] == target_shift + GRAMMER_CLASS['SPAN(']:
        passage_start = -1
        
        
        program_mask = [0]*(op_size)
        program_mask += [1]* (vocab_size - len(program_mask))
        
        op_list = ['COUNT(', 'MULTI-SPAN('] ## avoid repetition
        if prog_ids[0] in [target_shift + GRAMMER_CLASS[op] for op in op_list]:
            import pdb
            prog_idx = 0
            try:
                while prog_idx < len(prog_ids) - 3:
                    while prog_idx < len(prog_ids) - 3 and prog_ids[prog_idx] not in [target_shift + GRAMMER_CLASS[op] for op in atom_op_list]:
                        prog_idx += 1
                    if prog_idx >= len(prog_ids) - 3:
                        break
                    span_st = prog_ids[prog_idx + 1]
                    span_ed = prog_ids[prog_idx + 2]
                    for i in range(span_st, span_ed + 1):
                        program_mask[i] = 0
                    prog_idx += 4
            except:
                import pdb
                pdb.set_trace()
        try:
            passage_start = paragraph_mask.index(1)
            passage_end = passage_start + int(sum(paragraph_mask))
        except:
            passage_start = -1
            
        if op_stack[-1][1] == 0:
            current_program_mask = [0] * vocab_size
            if passage_start!=-1:
                current_program_mask[op_size+passage_start:op_size+passage_end] = [1] * (passage_end - passage_start)
            return [a*b for a, b in zip(program_mask, current_program_mask)]
        else:
            prev_prog_id = prog_ids[-1]
            current_program_mask = [0] * (prev_prog_id)
            current_program_mask += [1]*(passage_end+op_size -prev_prog_id) 
            current_program_mask += [0]*(vocab_size - len(current_program_mask))
            
            return [a*b for a, b in zip(program_mask, current_program_mask)]
        ## st, ed, st, ed region
        
    if op_stack[-1][0] == target_shift + GRAMMER_CLASS['KEY_VALUE(']:
        if op_stack[-1][1] == 0:
            program_mask = [0] * vocab_size
            program_mask[ target_shift + GRAMMER_CLASS['CELL(']] = 1
#             program_mask[ target_shift + GRAMMER_CLASS['SPAN(']] = 1
            return program_mask
        else:
            program_mask = [0] * vocab_size
            program_mask[ target_shift + GRAMMER_CLASS['CELL_VALUE(']]=1
#             program_mask[ target_shift + GRAMMER_CLASS['VALUE(']] = 1
            return program_mask
                                        
    if op_stack[-1][0] ==  target_shift + GRAMMER_CLASS['VALUE(']:
#         ## st, ed, region, include value
#         program_mask = [0] * (grammar_size + aux_num_size + target_shift)  ## refine
#         program_mask = program_mask + [1] * (vocab_size-len(program_mask))

        program_mask = [0]*(op_size)
        program_mask += [1]* (vocab_size - len(program_mask))
        
        op_list = ['COUNT(', 'MULTI-SPAN('] ## avoid repetition
        if prog_ids[0] in [target_shift + GRAMMER_CLASS[op] for op in op_list]:
            import pdb
            prog_idx = 0
            try:
                while prog_idx < len(prog_ids) - 3:
                    while prog_idx < len(prog_ids) - 3 and prog_ids[prog_idx] not in [target_shift + GRAMMER_CLASS[op] for op in atom_op_list]:
                        prog_idx += 1
                    if prog_idx >= len(prog_ids) - 3:
                        break
                    span_st = prog_ids[prog_idx + 1]
                    span_ed = prog_ids[prog_idx + 2]
                    for i in range(span_st, span_ed + 1):
                        program_mask[i] = 0
                    prog_idx += 4
            except:
                import pdb
                pdb.set_trace()


        
        import pdb
        if op_stack[-1][1] == 0:
            current_program_mask = [0] * vocab_size
            try:
                for id, number_value in enumerate(paragraph_number_value):
                    if not np.isnan(number_value) and id+1 in paragraph_index:
                        start_token_index = paragraph_index.index(id+1)
                        current_program_mask[start_token_index+op_size] = 1
            except:
                import pdb
                pdb.set_trace()
            return [a*b for a, b in zip(program_mask, current_program_mask)]
        else:
#             pdb.set_trace()
            prev_prog_id = prog_ids[-1]
            current_program_mask = [0] * (prev_prog_id)
            real_prog_id = prev_prog_id - op_size
            value_id = paragraph_index[real_prog_id]
            reversed_paragraph_index = list(reversed(paragraph_index))
            value_end = reversed_paragraph_index.index(value_id)
            value_end = len(reversed_paragraph_index)- value_end ## not include end index
#             pdb.set_trace()
            current_program_mask += [1] * (value_end+op_size - prev_prog_id)
            current_program_mask += [0]*(vocab_size - len(current_program_mask))
            
            assert len(current_program_mask) == vocab_size
            return  [a*b for a, b in zip(program_mask, current_program_mask)]
        
        
    if op_stack[-1][0] == target_shift + GRAMMER_CLASS['CELL_VALUE(']:
        ## st, ed, region, include value
        import pdb
#         program_mask = [0] * (grammar_size + aux_num_size + target_shift)  ## refine
#         program_mask = program_mask + [1] * (vocab_size-len(program_mask))


        program_mask = [0]*(op_size)
        program_mask += [1]* (vocab_size - len(program_mask))
        
        op_list = ['COUNT(', 'MULTI-SPAN('] ## avoid repetition
        if prog_ids[0] in [target_shift + GRAMMER_CLASS[op] for op in op_list]:
            import pdb
            prog_idx = 0
            try:
                while prog_idx < len(prog_ids) - 3:
                    while prog_idx < len(prog_ids) - 3 and prog_ids[prog_idx] not in [target_shift + GRAMMER_CLASS[op] for op in atom_op_list]:
                        prog_idx += 1
                    if prog_idx >= len(prog_ids) - 3:
                        break
                    span_st = prog_ids[prog_idx + 1]
                    span_ed = prog_ids[prog_idx + 2]
                    for i in range(span_st, span_ed + 1):
                        program_mask[i] = 0
                    prog_idx += 4
            except:
                import pdb
                pdb.set_trace()


        ## cell_value 和 cell 必须在同一行 或者同一列
#         op_list = ['ARGMAX(', 'ARGMIN(']
#         if prog_ids[0] in [target_shift + GRAMMER_CLASS[op] for op in op_list]:
#             import pdb
            
#             current_program_mask = [0] * vocab_size
            
#             prog_idx = len(prog_ids)-1
#             while prog_idx>=0:
#                 if prog_ids[prog_idx] == target_shift + GRAMMER_CLASS['CELL(']:
#                     cell_token_start = prog_ids[prog_idx+1]
#                     cell_token_end = prog_ids[prog_idx+2]
#                     break
#                 prog_idx-=1
                
#             cell_id = table_cell_index[cell_token_start-op_size]
            
#             row_cells = None
#             col_cells = None
#             for cells in row_include_cells:
#                 if cell_id in cells:
#                     row_cells = cells
#                     break
                
#             for cells in col_include_cells:
#                 if cell_id in cells:
#                     col_cells = cells
#                     break
            
            
#             if row_cells is not None:
#                 for cur_cell in row_cells:
#                     if cur_cell != cell_id and cur_cell!=0:
#                         token_start = table_cell_index.index(cur_cell)
#                         reversed_table_index = list(reversed(table_cell_index))
#                         token_end = reversed_table_index.index(cur_cell)
#                         token_end = len(reversed_table_index) - token_end
#                         current_program_mask[op_size +token_start:op_size+token_end] = [1]*(token_end-token_start)
                       
                
#             if col_cells is not None:
#                 for cur_cell in col_cells:
#                     if cur_cell != cell_id and cur_cell!=0:
#                         token_start = table_cell_index.index(cur_cell)
#                         reversed_table_index = list(reversed(table_cell_index))
#                         token_end = reversed_table_index.index(cur_cell)
#                         token_end = len(reversed_table_index) - token_end
#                         current_program_mask[op_size +token_start:op_size+token_end] = [1]*(token_end-token_start)
#             if row_cells is not None or col_cells is not None:
#                 program_mask = [a*b for a, b in zip(program_mask, current_program_mask)]  
        
        if op_stack[-1][1] == 0:
            current_program_mask = [0] * vocab_size
            try:
                for id, cell_number_value in enumerate(table_cell_number_value):
                    if not np.isnan(cell_number_value) and id+1 in table_cell_index:
                        start_token_index = table_cell_index.index(id+1)
                        current_program_mask[start_token_index+op_size] = 1
            except:
                import pdb
                pdb.set_trace()
            return  [a*b for a, b in zip(program_mask, current_program_mask)]
        else:
#             pdb.set_trace()
            prev_prog_id = prog_ids[-1]
            current_program_mask = [0] * (prev_prog_id)

            real_prog_id = prev_prog_id - op_size
            table_id = table_cell_index[real_prog_id]
            reversed_table_cell_index = list(reversed(table_cell_index))
            table_end = reversed_table_cell_index.index(table_id)
            table_end = len(reversed_table_cell_index)- table_end ## not include end index
#             pdb.set_trace()
            current_program_mask += [1] * (table_end+op_size - prev_prog_id)
            current_program_mask += [0]*(vocab_size - len(current_program_mask))
            
            return  [a*b for a, b in zip(program_mask, current_program_mask)]
                
    if op_stack[-1][0] == target_shift + GRAMMER_CLASS['MULTI-SPAN(']:
        ##  not duplicated
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
    
#     if op_stack[-1][1] >= 2:
#         import pdb
#         pdb.set_trace()
#         program_mask[target_shift + GRAMMER_CLASS[')']] = 1
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
