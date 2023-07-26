import json
import re

def regularize(expression):
    expression = expression.replace("m1", "-1")
    expression = expression.replace("const_", "")

    # 使用正则表达式匹配百分号数字，并将其转换为小数
    def replace_percent(match):
        percent_number = float(match.group(1))
        return str(percent_number / 100)

    # 匹配百分数
    pattern_percent = r"(\d+(\.\d*)?)%"
    result = re.sub(pattern_percent, replace_percent, expression)

    return result


def extract_single_expression(expression):
    # 将单独的表达式 op(a,b) 抽离出 op,a,b
    parts = expression.split("(")
    operator = parts[0].strip()
    arguments = parts[1].split(")")[0].split(",")
    arguments = [arg.strip() for arg in arguments]
    return operator, arguments[0], arguments[1]


def program_to_expression(program):
    if "table_" in program: # 需要通过表格推理的table_任务不处理
        return program
    program = regularize(program)
    expressions = program.split("),")
    last_res = ""
    result_history = []
    for expression in expressions:
        op, arg1, arg2 = extract_single_expression(expression)
        if arg1[0] == "#":
            arg1_num = int(arg1[1:])
            arg1 = result_history[arg1_num]
        if arg2[0] == "#":
            arg2_num = int(arg2[1:])
            arg2 = result_history[arg2_num]

        this_res = ""
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
        elif op == "greater":
            this_res = arg1 + ">" + arg2

        last_res = "(" + this_res + ")"
        result_history.append(last_res)
    return last_res


def process_file(data_path, out_path):
    with open(data_path, 'r') as file:
        data = json.load(file)

    with open(out_path, 'w+') as file:
        outputs = []
        for one in data:
            program = one["output"]
            expression = program_to_expression(program)
            # if "table_" not in expression:
            #     print(eval(expression))
            output = one
            output["program"] = program
            output["output"] = expression
            outputs.append(output)
        json.dump(outputs, file, indent=4)


if __name__ == '__main__':
    program1 = "subtract(6876,6876), divide(#0,6876), add(1,#1), exp(#2,const_2), multiply(#3,6876)"  # 6876.0
    print(program_to_expression(program1))
    program2 = "multiply(3176, const_m1), divide(#0, 10379)"
    print(program_to_expression(program2))
    program3 = "subtract(36.5%, 19.3%)"
    print(program_to_expression(program3))
    program4 = "subtract(78.0, 75.3), subtract(58.0, 49.9), subtract(54.0, 51.8), add(#0, #1), add(#2, #3)"
    print(program_to_expression(program4))
    program5 = "subtract(1018.5, 907.0), subtract(1033.9, 792.9), greater(#0, #1)"
    print(program_to_expression(program5))
    program6 = "table_max(cumulative foreign currency translation, none)"
    print(program_to_expression(program6))

    # data_path = "../data/train_instruction.json"
    # out_path = "../data/test_ljh_new.json"
    # process_file(data_path,out_path)