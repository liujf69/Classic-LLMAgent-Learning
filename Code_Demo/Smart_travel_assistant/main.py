from prompts.system_prompt import AGENT_SYSTEM_PROMPT

from tools.simple_tool import get_weather, get_attraction

import re
import os
import argparse
from openai import OpenAI
class OpenAICompatibleClient:
    """
    一个用于调用任何兼容OpenAI接口的LLM服务的客户端。
    """
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用LLM API来生成回应。"""
        print("正在调用大语言模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功。")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return "错误：调用语言模型服务时出错。"

def main(API_KEY: str, BASE_URL: str, MODEL_ID: str, TAVILY_API_KEY: str, USER_PROMPT: str):
    # 将所有工具函数放入一个字典，方便后续调用
    available_tools = {
        "get_weather": get_weather,
        "get_attraction": get_attraction,
    }

    # --- 1. 配置LLM客户端 ---
    API_KEY = API_KEY
    BASE_URL = BASE_URL
    MODEL_ID = MODEL_ID
    TAVILY_API_KEY = TAVILY_API_KEY
    # 请根据您使用的服务，将这里替换成对应的凭证和地址
    os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY

    llm = OpenAICompatibleClient(
        model=MODEL_ID,
        api_key=API_KEY,
        base_url=BASE_URL
    )

    # --- 2. 初始化 ---
    user_prompt = USER_PROMPT
    prompt_history = [f"用户请求: {user_prompt}"]

    print(f"用户输入: {user_prompt}\n" + "="*40)

    # --- 3. 运行主循环 ---
    for i in range(5): # 设置最大循环次数
        print(f"--- 循环 {i+1} ---\n")
        
        # 3.1. 构建Prompt
        full_prompt = "\n".join(prompt_history)
        
        # 3.2. 调用LLM进行思考
        llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
        # 模型可能会输出多余的Thought-Action，需要截断
        match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)
        if match:
            truncated = match.group(1).strip()
            if truncated != llm_output.strip():
                llm_output = truncated
                print("已截断多余的 Thought-Action 对")
        print(f"模型输出:\n{llm_output}\n")
        prompt_history.append(llm_output)
        
        # 3.3. 解析并执行行动
        action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
        if not action_match:
            observation = "错误: 未能解析到 Action 字段。请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式。"
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n" + "="*40)
            prompt_history.append(observation_str)
            continue
        action_str = action_match.group(1).strip()

        if action_str.startswith("Finish"):
            final_answer = re.match(r"Finish\[(.*)\]", action_str).group(1)
            print(f"任务完成，最终答案: {final_answer}")
            break
        
        tool_name = re.search(r"(\w+)\(", action_str).group(1)
        args_str = re.search(r"\((.*)\)", action_str).group(1)
        kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

        if tool_name in available_tools:
            observation = available_tools[tool_name](**kwargs)
        else:
            observation = f"错误：未定义的工具 '{tool_name}'"

        # 3.4. 记录观察结果
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "="*40)
        prompt_history.append(observation_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Smart_travel_assistant')
    parser.add_argument('--API_KEY', default=None, help='LLM API KEY')
    parser.add_argument('--BASE_URL', default=None, help='LLM BASE URL')
    parser.add_argument('--MODEL_ID', default=None, help='LLM MODEL ID')
    parser.add_argument('--TAVILY_API_KEY', default=None, help='TAVILY API KEY')
    parser.add_argument('--USER_PROMPT', default=None, help='USER PROMPT INPUT')

    args = parser.parse_args()
    API_KEY = args.API_KEY
    BASE_URL = args.BASE_URL
    MODEL_ID = args.MODEL_ID
    TAVILY_API_KEY = args.TAVILY_API_KEY
    USER_PROMPT = args.USER_PROMPT
    
    main(API_KEY, BASE_URL, MODEL_ID, TAVILY_API_KEY, USER_PROMPT)
 