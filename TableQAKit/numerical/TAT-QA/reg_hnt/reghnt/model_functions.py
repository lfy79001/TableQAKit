import numpy as np
import copy 
def get_continuous_tag_slots(paragraph_token_tag_prediction):
    tag_slots = []
    span_start = False
    for i in range(0, len(paragraph_token_tag_prediction)):
        if paragraph_token_tag_prediction[i] != 0 and not span_start:
            span_start = True
            start_index = i
        if paragraph_token_tag_prediction[i] == 0 and span_start:
            span_start = False
            tag_slots.append((start_index, i))
    if span_start:
        tag_slots.append((start_index, len(paragraph_token_tag_prediction)))
    return tag_slots


def get_single_span_tokens_from_table_and_paragraph(table_cell_tag_prediction, table_cell_tag_prediction_score, paragraph_token_tag_prediction, paragraph_token_tag_prediction_score, global_tokens):
    paragraph_token_tag_prediction = paragraph_token_tag_prediction[0]
    paragraph_token_tag_prediction_score = paragraph_token_tag_prediction_score[0]
    tag_slots = get_continuous_tag_slots(paragraph_token_tag_prediction)
    best_result = float("-inf")
    best_combine = []
    for tag_slot in tag_slots:
        current_result = np.mean(paragraph_token_tag_prediction_score[tag_slot[0]:tag_slot[1]])
        if current_result > best_result:
            best_result = current_result
            best_combine = tag_slot
    table_cell_tag_prediction = table_cell_tag_prediction[0]
    table_cell_tag_prediction_score = table_cell_tag_prediction_score[0]
    tagged_cell_index = [i for i in range(len(table_cell_tag_prediction)) if table_cell_tag_prediction[i] != 0]
    
    if not tagged_cell_index and not best_combine:
        deep_cp = copy.deepcopy(table_cell_tag_prediction_score)
        aa = np.where(deep_cp==0)[0]
        deep_cp[aa] = float("-inf")
        index = np.argmax(deep_cp)
        return [str(global_tokens[index])]
    if not tagged_cell_index:
        return [" ".join(global_tokens[best_combine[0]: best_combine[1]])]
    if not best_combine:
        tagged_cell_tag_prediction_score = \
            [table_cell_tag_prediction_score[i] for i in tagged_cell_index]
        best_result_index = tagged_cell_index[int(np.argmax(tagged_cell_tag_prediction_score))]
        return [str(global_tokens[best_result_index])]
    
    tagged_cell_tag_prediction_score = \
        [table_cell_tag_prediction_score[i] for i in tagged_cell_index]
    tagged_cell_max_score = np.max(tagged_cell_tag_prediction_score)
    if tagged_cell_max_score > current_result:
        best_result_index = tagged_cell_index[int(np.argmax(tagged_cell_tag_prediction_score))]
        return [str(global_tokens[best_result_index])]
    else:
        return [" ".join(global_tokens[best_combine[0]: best_combine[1]])]

def get_single_span_tokens_from_question_table_paragraph(question_tag_prediction, question_tag_prediction_score, table_cell_tag_prediction,table_cell_tag_prediction_score,paragraph_token_tag_prediction, paragraph_token_tag_prediction_score, global_tokens):
    paragraph_token_tag_prediction = paragraph_token_tag_prediction[0]
    paragraph_token_tag_prediction_score = paragraph_token_tag_prediction_score[0]
    paragraph_tag_slots = get_continuous_tag_slots(paragraph_token_tag_prediction)
    paragraph_best_result = float("-inf")
    paragraph_best_combine = []
    for tag_slot in paragraph_tag_slots:
        current_result = np.mean(paragraph_token_tag_prediction_score[tag_slot[0]:tag_slot[1]])
        if current_result > paragraph_best_result:
            paragraph_best_result = current_result
            paragraph_best_combine = tag_slot
    
    question_tag_prediction = question_tag_prediction[0]
    question_tag_prediction_score = question_tag_prediction_score[0]
    question_tag_slots = get_continuous_tag_slots(question_tag_prediction)
    question_best_result = float("-inf")
    question_best_combine = []
    for tag_slot in question_tag_slots:
        current_result = np.mean(question_tag_prediction_score[tag_slot[0]:tag_slot[1]])
        if current_result > question_best_result:
            question_best_result = current_result
            question_best_combine = tag_slot

    table_cell_tag_prediction = table_cell_tag_prediction[0]
    table_cell_tag_prediction_score = table_cell_tag_prediction_score[0]
    tagged_cell_index = [i for i in range(len(table_cell_tag_prediction)) if table_cell_tag_prediction[i] != 0]

    if not tagged_cell_index and not question_best_combine and not paragraph_best_combine:
        deep_cp = copy.deepcopy(table_cell_tag_prediction_score)
        aa = np.where(deep_cp==0)[0]
        deep_cp[aa] = float("-inf")
        index = np.argmax(deep_cp)
        return [str(global_tokens[index])]
    if not tagged_cell_index and not paragraph_best_combine:
        return [" ".join(global_tokens[question_best_combine[0]: question_best_combine[1]])]
    if not tagged_cell_index and not question_best_combine:
        return [" ".join(global_tokens[paragraph_best_combine[0]: paragraph_best_combine[1]])]
    if not paragraph_best_combine and not question_best_combine:
        tagged_cell_tag_prediction_score = \
            [table_cell_tag_prediction_score[i] for i in tagged_cell_index]
        best_result_index = tagged_cell_index[int(np.argmax(tagged_cell_tag_prediction_score))]
        return [str(global_tokens[best_result_index])]
    
    q_score, t_score, p_score = float("-inf"), float("-inf"), float("-inf")
    if tagged_cell_index:
        tagged_cell_tag_prediction_score = \
            [table_cell_tag_prediction_score[i] for i in tagged_cell_index]
        t_score = np.max(tagged_cell_tag_prediction_score)
    if question_best_combine:
        q_score = question_best_result
    if paragraph_best_combine:
        p_score = paragraph_best_result
    index = np.argmax([q_score, t_score, p_score])
    if index == 0:
        return [" ".join(global_tokens[question_best_combine[0]: question_best_combine[1]])]
    elif index == 1:
        best_result_index = tagged_cell_index[int(np.argmax(tagged_cell_tag_prediction_score))]
        return [str(global_tokens[best_result_index])]
    elif index == 2:
        return [" ".join(global_tokens[paragraph_best_combine[0]: paragraph_best_combine[1]])]




def get_single_span_tokens_from_paragraph(paragraph_token_tag_prediction,
                                          paragraph_token_tag_prediction_score,
                                          global_tokens):
    paragraph_token_tag_prediction = paragraph_token_tag_prediction[0]
    paragraph_token_tag_prediction_score = paragraph_token_tag_prediction_score[0]
    tag_slots = get_continuous_tag_slots(paragraph_token_tag_prediction)
    best_result = float("-inf")
    best_combine = []
    for tag_slot in tag_slots:
        current_result = np.mean(paragraph_token_tag_prediction_score[tag_slot[0]:tag_slot[1]])
        if current_result > best_result:
            best_result = current_result
            best_combine = tag_slot
    if not best_combine:
        return []
    else:
        return [" ".join(global_tokens[best_combine[0]: best_combine[1]])]


def get_single_span_tokens_from_table(table_cell_tag_prediction,
                                      table_cell_tag_prediction_score,
                                      global_tokens):
    table_cell_tag_prediction = table_cell_tag_prediction[0]
    table_cell_tag_prediction_score = table_cell_tag_prediction_score[0]
    tagged_cell_index = [i for i in range(len(table_cell_tag_prediction)) if table_cell_tag_prediction[i] != 0]
    if not tagged_cell_index:
        return []
    tagged_cell_tag_prediction_score = \
        [table_cell_tag_prediction_score[i] for i in tagged_cell_index]
    best_result_index = tagged_cell_index[int(np.argmax(tagged_cell_tag_prediction_score))]
    return [str(global_tokens[best_result_index])]

def get_span_tokens_from_paragraph(paragraph_token_tag_prediction, global_tokens):
    span_tokens = []
    span_start = False
    paragraph_token_tag_prediction = paragraph_token_tag_prediction[0]

    for i in range(0, min(len(global_tokens), len(paragraph_token_tag_prediction))):
        if paragraph_token_tag_prediction[i] == 0:
            span_start = False
        if paragraph_token_tag_prediction[i] != 0:
            if not span_start:
                span_tokens.append([global_tokens[i]])
                span_start = True
            else:
                span_tokens[-1] += [global_tokens[i]]
    span_tokens = [" ".join(tokens) for tokens in span_tokens]
    return span_tokens

def get_span_tokens_from_question(question_tag_prediction, global_tokens):
    span_tokens = []
    span_start = False
    question_tag_prediction = question_tag_prediction[0]
    for i in range(0, min(len(global_tokens), len(question_tag_prediction))):
        if question_tag_prediction[i] == 0:
            span_start = False
        if question_tag_prediction[i] != 0:
            if not span_start:
                span_tokens.append([global_tokens[i]])
                span_start = True
            else:
                span_tokens[-1] += [global_tokens[i]]
    span_tokens = [" ".join(tokens) for tokens in span_tokens]
    return span_tokens


def get_span_tokens_from_table(table_cell_tag_prediction, global_tokens):
    table_cell_tag_prediction = table_cell_tag_prediction[0]
    span_tokens = []
    for i in range(0, len(table_cell_tag_prediction)):
        if table_cell_tag_prediction[i] != 0:
            span_tokens.append(str(global_tokens[i]))
    return span_tokens