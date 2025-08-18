

from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
import os, json, re
import http.client
import os
import time
import json
import re
import ast
import sys
import traceback
from io import StringIO


def message_template(role, new_info):
    new_dict = {'role': role, 'content': new_info}
    return new_dict


client = OpenAI()
# print(os.environ['OPENAI_API_KEY'])

def format_list_string(input_str):
    # 正则匹配大括号内的内容
    match = re.search(r'\{\s*"[^"]+"\s*:\s*\[(.*?)\]\s*\}', input_str)
    if not match:
        return "Invalid input format"

    list_content = match.group(1)  # 获取匹配到的列表内容
    elements = [e.strip() for e in list_content.split(',')]  # 拆分并去除多余空格

    formatted_elements = []
    for elem in elements:
        if not re.match(r'^([\'"])(.*)\1$', elem):  # 检查是否被引号包裹
            elem = f'"{elem}"'  # 添加双引号
        formatted_elements.append(elem)

    return f'{{ "similar_words":[{", ".join(formatted_elements)}]}}'

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_single(messages, mode="", model='gpt-4o', temperature=0,verbose=False):
    if mode == "json":

        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            temperature=temperature,
            messages=messages
        )
    elif mode == 'stream':
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            stream=True,
            max_tokens=2560

        )
        return response
    elif mode == 'json_few_shot':
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=2560

        )
        result= response.choices[0].message.content

        if verbose:print(result)

        return extract_json_and_similar_words(
            result)
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,

        )

    # print(response.choices[0].message.content)
    return response.choices[0].message.content


def general_gpt_without_memory(query='', messages=None, json_mode='', sys_prompt='', temperature=0, verbose=False):
    if isinstance(query, dict):
        query = str(query)
    if query == None:
        return None
    if messages == None:
        messages = []


        messages.append(message_template('system', sys_prompt))
        messages.append(message_template('user', str(query)))
    # result = chat_single(messages, '','gpt-4o-2024-05-13')
    result = chat_single(messages, json_mode,temperature=temperature)
    if verbose:
        print('general_gpt result:', result)
    return result

def extract_json_and_similar_words(text):
    try:

        # 使用正则表达式提取 JSON 部分
            json_match = re.search(r'```json\s*({.*?})\s*```', text, re.DOTALL)

            if not json_match:
                raise ValueError("No JSON data found in the text.")

            # 提取 JSON 字符串
            json_str = json_match.group(1)
            print(json_str)
            # 解析 JSON 字符串为 Python 字典
            data = json.loads(json_str)

            # 提取 'similar_words' 列表

            return data
    except Exception as e:
        print( e)
        if 'similar_words' in text:
            json_match = re.search(r'```json\s*({.*?})\s*```', text, re.DOTALL)

            if not json_match:
                raise ValueError("No JSON data found in the text.")

            # 提取 JSON 字符串
            json_str = json_match.group(1)

            # 解析 JSON 字符串为 Python 字典
            data = json.loads(format_list_string(json_str))

        # 提取 'similar_words' 列表

        return data
def messages_initial_template(ask_prompt,user_query):
    messages=[]
    messages.append(message_template('system',ask_prompt))
    messages.append(message_template('system',user_query))
    return messages
def extract_code_blocks(code_str):
    code_blocks = []
    code_result = []
    if '```python' in code_str:
        parts = code_str.split("```python")
        for part in parts[1:]:  # 跳过第一个部分，因为它在第一个代码块之前
            code_block = part.split("```")[0]



            code_blocks.append(code_block)
        code_str=code_blocks[0]
        for code_part in code_str.split('\n'):
            if 'import' not in code_part and '=' not in code_part and code_part.strip() and 'print' not in code_part and '#' not in code_part:
                code_piece=f'print({code_part})'
            else:
                code_piece=code_part
            code_result.append(code_piece)
        # print(code_result)
        return "\n".join(code_result)
    return code_str
def execute_and_display(code_str, local_vars=None):
    if local_vars is None:
        local_vars = {}

    # 将输出重定向到字符串缓冲区
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        # 解析代码为 AST
        tree = ast.parse(code_str)

        # 如果 AST 的 body 为空，直接返回
        if not tree.body:
            output = sys.stdout.getvalue()
            return output if output else None

        # 分离最后一行（可能是表达式）和前面的语句
        if len(tree.body) > 1:
            exec(compile(ast.Module(tree.body[:-1], []), "<ast>", "exec"), local_vars)
            last_node = tree.body[-1]
        else:
            last_node = tree.body[0]

        # 如果最后一行是表达式，单独执行并获取结果
        if isinstance(last_node, ast.Expr):
            result = eval(compile(ast.Expression(last_node.value), "<ast>", "eval"), local_vars)
            # 获取缓冲区的输出（包括 print 调用和自动显示的内容）
            output = sys.stdout.getvalue()
            # 如果结果不是 None，附加到输出
            if result is not None and hasattr(result, "__str__"):
                output += str(result) + "\n"
            return output if output else result
        else:
            # 如果不是表达式，直接执行整个代码
            exec(code_str, local_vars)
            output = sys.stdout.getvalue()
            return output if output else None

    except Exception:
        # 捕获异常并返回完整的 Traceback 和之前的输出
        output = sys.stdout.getvalue()
        error_msg = traceback.format_exc()
        return output + error_msg if output else error_msg

    finally:
        # 恢复标准输出
        sys.stdout = old_stdout
def extract_words(text, mode='json'):
    # 使用正则表达式提取 JSON 部分
    if mode=='python':
        return extract_code_blocks(text)
    json_match = re.search(r'```%s\s*({.*?})\s*```' % (mode), text, re.DOTALL)

    if not json_match:
        raise ValueError(f"No {mode} data found in the text.")
    # 提取 JSON 字符串

    json_str = json_match.group(1)

    print("extract_words",json_str)
    if mode == 'json':
        # 解析 JSON 字符串为 Python 字典
        data = json.loads(json_str)
    else:
        data = json_str
    # 提取 'similar_words' 列表

    return data
def extract_python_code(input_str):
    # 使用正则表达式匹配 ```python ``` 包裹的代码块
    pattern = r"```python\s*\n(.*?)\n```"
    match = re.search(pattern, input_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError("No ```python``` code block found in the input string.")
def iterative_agent(sys_prompt,query):
#     ask_prompt = """Here’s the revised prompt with an HTML output requirement:
#
# ---
#
# You are a data-processing intelligent assistant designed to analyze CSV files in the current directory. Your task is to process these files step by step using Python and pandas to extract data suitable for visualization (e.g., charts, maps). You must always generate Python code, no matter how simple the request.
#
# For each user query:
#
# 1. **Understand the question** and determine the necessary data processing steps.
# 2. **Always write a Python snippet** for every step, enclosed within triple backticks (```python to ```). The code must use pandas to manipulate and extract relevant data.
# 3. **Assume CSV files exist** and can be loaded with `pd.read_csv("filename.csv")`.
# 4. **Process incrementally**:
#    - Run an **initial analysis** to inspect the data structure.
#    - Based on the results, refine the processing steps with additional code snippets.
#    - Continue iterating until the final dataset is ready for visualization.
# 5. **Output the final answer as JSON**, stored in the variable `final_answer`, ensuring it contains only essential fields. The output must be concise and structured for visualization purposes.
# 6. **Generate an HTML snippet** to display the final results using a suitable visualization (e.g., a chart using Chart.js, a table, or a map). The HTML should be enclosed within triple backticks (```html to ```).
#
# Clearly explain each step before providing the next Python or HTML snippet.
#
# Now, proceed with the user’s query, ensuring every step includes both a Python snippet for data processing and an HTML snippet for the final presentation
# """


    messages = messages_initial_template(sys_prompt, query)
    round_num=0

    while True:
        round_num+=1
        code_result = chat_single(messages)
        print("response", code_result)

        messages.append(message_template('assistant', code_result))

        if 'python' in code_result:
            code_return = str(execute_and_display(extract_python_code(code_result), globals()))
        else:
            code_return = code_result

        raw_return = str(code_return)

        print("code_return", code_return)
        messages.append(message_template('user', code_return))

        if  'final_answer' in code_result:

            if 'traceback' not in code_return.lower():

                break
    return raw_return
