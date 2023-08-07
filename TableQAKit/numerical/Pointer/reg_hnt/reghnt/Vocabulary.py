CONST_LIST = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
CONST_LIST2 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '12','100','1000']
OPERATOR_LIST = ['+', '-', '*', '/']
OPERATOR_CLASSES = {"SPAN": 0, "MULTI_SPAN": 1, "COUNT": 2, "ARITHMETIC": 3}
OPERATOR_CLASSES_R = {0:"SPAN", 1:"MULTI_SPAN", 2:"COUNT" ,3:"ARITHMETIC"}
SCALE = ["", "thousand", "million", "billion", "percent"]

RC_OPERATOR_LIST = ['*', '+']
RC_CONST_LIST = []

Relation_vocabulary = {
    "word2id": {
    "Row-Contain-Cell":0, "Cell-Belong-Row":1,
    "Column-Contain-Cell":2, "Cell-Belong-Column":3,
    "Time-Contain-Cell":4, "Cell-Belong-Time":5,
    "Cell-SameRow-Cell":6, 
    "Q_Word-NextWord-Q_Word":7, "Q_Word-PriorWord-Q_Word":8,####
    "Q_Word-PartialMatch-Row":9, "Row-PartialMatch-Q_Word":10,
    "Q_Word-ExactMatch-Row":11, "Row-ExactMatch-Q_Word":12,
    "Q_Word-Partof-Sentence":13, "Sentence-Contain-Q_Word":14,
    "P_Word-Partof-Sentence":15, "Sentence-Contain-P_Word":16,
    "Sentence-Contain-Row":17, "Row-Partof-Sentence":18,
    "Word-Same-Word": 19, 
    "P_Word-PartialMatch-Row":20,"Row-PartialMatch-P_Word":21, 
    "P_Word-ExactMatch-Row":22,"Row-ExactMatch-P_Word": 23,
    "self-same-self":24
},
    "id2word": {
    0:"Row-Contain-Cell", 1:"Cell-Belong-Row",
    2:"Column-Contain-Cell", 3:"Cell-Belong-Column",
    4:"Time-Contain-Cell", 5:"Cell-Belong-Time",
    6:"Cell-SameRow-Cell", 
    7:"Q_Word-NextWord-Q_Word", 8:"Q_Word-PriorWord-Q_Word",####
    9:"Q_Word-PartialMatch-Row", 10:"Row-PartialMatch-Q_Word",
    11:"Q_Word-ExactMatch-Row", 12:"Row-ExactMatch-Q_Word",
    13:"Q_Word-Partof-Sentence", 14:"Sentence-Contain-Q_Word",
    15:"P_Word-Partof-Sentence", 16:"Sentence-Contain-P_Word",
    17:"Sentence-Contain-Row", 18:"Row-Partof-Sentence",
    19:"Word-Same-Word",
    20:"P_Word-PartialMatch-Row", 21:"Row-PartialMatch-P_Word",
    22:"P_Word-ExactMatch-Row", 23:"Row-ExactMatch-P_Word",
    24:"self-same-self"
    }
}