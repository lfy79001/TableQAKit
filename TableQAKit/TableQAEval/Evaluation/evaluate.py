import json
import argparse
from emf1 import compute_emf1, compute_exact, compute_f1



def main(args):
    with open(args.gold_file, 'r') as file:
        gold_data = json.load(file)
    with open(args.predict_file) as pred_file:
        pred_data = json.load(pred_file)

    gold_dict = {}
    
    for item in gold_data:
        gold_dict[item['question_id']] = {'answer': item['answer'], 'source': item['source']}
        
    pred_keys = set(pred_data.keys())
    gold_keys = set(gold_dict.keys())

    join_keys = list(pred_keys & gold_keys)
    
    numerical1, multihop1, structured1, total1 = [], [], [], []
    numerical2, multihop2, structured2, total2 = [], [], [], []
    for id in join_keys:
        gold = gold_dict[id]['answer']
        source = gold_dict[id]['source']
        pred = pred_data[id]['answer']

        em = compute_exact(str(pred), str(gold))
        f1 = compute_f1(str(pred), str(gold))
        if source == 'numerical':
            numerical1.append(em)
            numerical2.append(f1)
        elif source == 'multihop':
            multihop1.append(em)
            multihop2.append(f1)
        elif source == 'structured':
            structured1.append(em)
            structured2.append(f1)
        total1.append(em)
        total2.append(f1)
    
    if len(numerical1) > 0: print(f"Numerical: em {sum(numerical1) / len(numerical1) * 100}, f1: {sum(numerical2) / len(numerical2) * 100} {len(numerical1)}")
    if len(multihop1) > 0: print(f"Multihop: em {sum(multihop1) / len(multihop1) * 100}, f1: {sum(multihop2) / len(multihop2) * 100} {len(multihop1)}")
    if len(structured1) > 0: print(f"Structured: em {sum(structured1) / len(structured1) * 100}, f1: {sum(structured2) / len(structured2) * 100} {len(structured1)}")
    print(f"Total: em {sum(total1) / len(total1) * 100}, f1: {sum(total2) / len(total2) * 100} {len(total2)}")
    print(f"Final Total: em {sum(total1) / 1000 * 100}, f1: {sum(total2) / 1000 * 100} 1000")
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('--predict_file', type=str, default='../Results/Turbo16k.json')
    parser.add_argument('--gold_file', type=str, default='../TableQAEval.json')
    
    args = parser.parse_args()
    main(args)