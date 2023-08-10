import json
import re

def extract_single_expression(expression):
    parts = expression.split("(")
    operator = parts[0].strip()
    arguments = parts[1].split(")")[0].split(",")
    # 去除参数两侧的空格
    arguments = [arg.strip() for arg in arguments]

    return operator, arguments[0], arguments[1]
def program_to_expression(program):
    expressions = program.split("),")
    last_res = ""
    result_history = []
    for expression in expressions:
        op, arg1, arg2 = extract_single_expression(expression)
        if arg1[0] == "#":
            arg1_num=int(arg1[1:])
            arg1 = result_history[arg1_num]
        if arg2[0] == "#":
            arg2_num = int(arg2[1:])
            arg2 = result_history[arg2_num]

        if arg1.startswith("const_"):
            arg1 = arg1[6:]
        if arg2.startswith("const_"):
            arg2 = arg2[6:]

        this_res=""
        if op == "add":
            this_res = arg1 + "+" + arg2
        elif op == "subtract":
            this_res = arg1 + "-" + arg2
        elif op == "multiply":
            this_res = arg1 + "*" + arg2
        elif op == "divide":
            this_res = arg1 + "/" + arg2
        elif op == "exp":
            this_res = arg1 + "**" + arg2

        last_res = "(" + this_res + ")"
        result_history.append(last_res)
    return last_res
def convert_percent_to_decimal(expression):
    expression = expression.replace("m1", "-1")
    def replace_percent(match):
        percent_number = float(match.group(1))
        return str(percent_number / 100)
    # 正则表达式模式：匹配带百分号的数字
    pattern = r"(\d+(\.\d*)?)%"
    result = re.sub(pattern, replace_percent, expression)

    return result

def eval_compaqt_epression(data_path):
    same_answer_num = 0
    total_num = {"finqa": 0, "tatqa": 0, "hitab": 0, "multihiertt": 0}
    match_num = {"finqa": 0, "tatqa": 0, "hitab": 0, "multihiertt": 0}
    with open(data_path, 'r') as file:
        for line in file:
            data_dict = json.loads(line)
            output = data_dict["output"]
            pred = data_dict["pred"]
            source = data_dict["source"]
            expression = program_to_expression(output)  # 正确的表达式
            # if expression.count("(") >4:
            #     print(pred)
            total_num[source] += 1
            if expression == pred:
                match_num[source] += 1
            try:
                gd_ans = eval(convert_percent_to_decimal(expression))
            except:
                pass
            try:
                pred_ans = eval(convert_percent_to_decimal(pred))
            except:
                print(pred)  # llama预测出了不合法的表达式
                pass
            if pred_ans == gd_ans:
                same_answer_num += 1
    for key in total_num.keys():
        print(key, match_num[key] / total_num[key])
    print("总正确率", sum(match_num.values()) / sum(total_num.values()))
    print("答案预测正确数", same_answer_num)
    print("总样本数", sum(total_num.values()))
    print("总答案正确率", same_answer_num / sum(total_num.values()))

if __name__ == '__main__':
    data_path = "../data/test_predictions_expression3.json"
    eval_compaqt_epression(data_path)
