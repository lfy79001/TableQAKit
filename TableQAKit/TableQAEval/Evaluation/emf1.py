import re, string, collections, json

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1




def compute_emf1(predictions, references):
    
    total = len(references)
    em_list = []
    f1_list = []
    include_list = []
    for prediction, reference in zip(predictions, references):
        em = compute_exact(str(prediction), str(reference))
        f1 = compute_f1(str(prediction), str(reference))
        em_list.append(em)
        f1_list.append(f1)
        if str(reference) in str(prediction):
            include_list.append(1)
        else:
            include_list.append(0)
    print(f"include score: {sum(include_list) / len(include_list)}")
    return sum(em_list) / total, sum(f1_list) / total


def save_results(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(str(item) + '\n')
        
        
def read_results(file_path):
    with open(file_path, 'r') as f:
        data_list = [line.strip() for line in f.readlines()]
    return data_list