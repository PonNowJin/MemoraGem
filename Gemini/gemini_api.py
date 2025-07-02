import os
from dotenv import load_dotenv
import pathlib
from google.genai import types
from google import genai
import httpx
import json
from tools import get_data_path
load_dotenv()


def send_to_gemini(usr_prompt:str, response_file_path:str):
    """使用 gemini 完成 ppt 結構

    Args:
        usr_prompt: 使用者輸入的 prompt
        response_file_path: str
    """
    API_KEY = os.getenv('GEMINI_API_KEY')
    client = genai.Client(api_key=API_KEY)

    # 原始 prompt
    with open("Gemini/prompt.txt", "r", encoding="utf-8") as f:
        prompt = f.read()
        
    # 加入目前使用者提問
    prompt += '使用者提問：' + usr_prompt

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            prompt, 
        ],
        config={
            "response_mime_type": "application/json",
        }
        )

    # print(response.text)

    with open(response_file_path, "w", encoding="utf-8") as f:
        f.write(response.text)
        
    memory_entries = json.loads(response.text)
    # print(memory_entries)
    
    # 處理 big five
    big_five_path = get_data_path(os.path.join('record', 'big_five.json'))
    big_five_data = memory_entries['Big Five 預測']
    
    if os.path.exists(big_five_path):        
        with open(big_five_path, "r", encoding="utf-8") as f:
            load_data = json.load(f)
            load_data['Openness'] = (float(load_data['Openness']) + float(big_five_data['Openness']))/2
            load_data['Conscientiousness'] = (float(load_data['Conscientiousness']) + float(big_five_data['Conscientiousness']))/2
            load_data['Extraversion'] = (float(load_data['Extraversion']) + float(big_five_data['Extraversion']))/2
            load_data['Agreeableness'] = (float(load_data['Agreeableness']) + float(big_five_data['Agreeableness']))/2
            load_data['Neuroticism'] = (float(load_data['Neuroticism']) + float(big_five_data['Neuroticism']))/2
            
        with open(big_five_path, 'w', encoding='utf-8') as f:
            json.dump(load_data, f, ensure_ascii=False, indent=4)
            
    else:
        print(big_five_data)
        with open(big_five_path, 'w', encoding='utf-8') as f:
            json.dump(big_five_data, f, ensure_ascii=False, indent=4)
            


if __name__ == '__main__':
    usr_prompt = '我是一位科學家'
    out_path = 'generate_output.txt'
    send_to_gemini(usr_prompt, out_path)
