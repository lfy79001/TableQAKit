def change_output2(output, pre_order_tokens, question_id):
    try:
        if len(output) % 2 == 0:
            if output[0] == output[1]:
                output.pop(0)
                pre_order_tokens.pop(0)
            elif output[1] == output[2]:
                output.pop(1)
                pre_order_tokens.pop(1)
            elif output[2] == output[3]:
                output.pop(2)
                pre_order_tokens.pop(2)
            elif output[4] == output[5]:
                output.pop(4)
                pre_order_tokens.pop(4)
            elif output[7] == output[8]:
                output.pop(7)
                pre_order_tokens.pop(7)
    except:
        import pdb; pdb.set_trace()    
    return output, pre_order_tokens