import os
from dotenv import load_dotenv
import pathlib
from google.genai import types
from google import genai
import httpx
import json
import numpy as np
import faiss
from Gemini.tools import get_data_path
from sentence_transformers import SentenceTransformer
load_dotenv()

API_KEY = os.getenv('GEMINI_API_KEY')


TEXT_PATH = "record/memory_texts.json"
FAISS_PATH = "record/memory_index.faiss"
BIGFIVE_PATH = "record/big_five.json"


def store_data(usr_prompt:str, response_file_path:str=None):
    """使用 gemini 分析並儲存資料

    Args:
        usr_prompt: 使用者輸入的 prompt
        response_file_path: str
    """
    client = genai.Client(api_key=API_KEY)

    # 原始 prompt
    prompt_path = get_data_path('prompt.txt')
    with open(prompt_path, "r", encoding="utf-8") as f:
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

    if response_file_path:
        with open(response_file_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        
    memory_entries = json.loads(response.text)
    print(memory_entries)
    
    
    # 處理 big five
    big_five_path = get_data_path(BIGFIVE_PATH)
    big_five_data = memory_entries['BigFivePrediction']
    
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
            
            
    # 處理向量資料庫
    summary = memory_entries['Summary']
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(summary)
    if isinstance(vectors, list):  # 有些版本會回傳 list
        vectors = np.array(vectors)

    vectors = vectors.reshape(1, -1)  # 強制轉為 2D
    
    # 向量維度必須與模型輸出一致（MiniLM-L6-v2 是 384 維）
    dim = 384
    index_path = get_data_path(FAISS_PATH)

    # 如果已存在，載入；否則新建
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        index = faiss.IndexFlatL2(dim)
        
    # 加入新數據
    index.add(vectors)
    faiss.write_index(index, index_path)

    # 構造儲存內容（建議保留 raw data）
    stored_entries = [
        {
            "summary": summary,
            "raw": memory_entries
        }
    ]

    # 若檔案存在就 append 新的
    text_path = get_data_path(TEXT_PATH)
    if os.path.exists(text_path):
        with open(text_path, "r") as f:
            existing = json.load(f)
    else:
        existing = []

    existing.extend(stored_entries)

    with open(text_path, "w") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def send_to_gemini(usr_prompt: str, response_file_path:str = None):
    """ 處理使用者問題與回應
    
    """
    # 查詢向量資料庫
    search_results = search_memory(usr_prompt)
    history = ''
    for result in search_results:
        history += result['summary']
        history += ', '
        
    # 抓取 big five
    big_five_path = get_data_path(BIGFIVE_PATH)
    if os.path.exists(big_five_path):
        with open(big_five_path, "r", encoding="utf-8") as f:
            big_five_data = json.load(f)
    else:
        big_five_data = None
    
    # 製作 prompt（沒加入 big five)
    prompt = f"使用者詢問(只需對此回答其餘不用，不用說了解）：{usr_prompt}, 相關歷史紀錄（參考）：{history}"
    
    client = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            prompt, 
        ],
        )
    
    print(response.text)

    if response_file_path:
        with open(response_file_path, "w", encoding="utf-8") as f:
            f.write(response.text)
            
    return {'response': response.text, 'ref': search_results}


def search_memory(query: str, top_k: int = 5, distance_threshold: float = 1.2):
    """ 從記憶中找出 top k 最相關紀錄

    Args:
        query (str): 使用者輸入
        top_k (int): 取前幾筆
        distance_threshold (float): 若距離超過此值，視為不相關

    Returns:
        List[dict]: 匹配到的記憶條目
    """
    # 載入 embedding 模型
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = model.encode([query])  # shape: (1, 384)
    
    # 載入 FAISS 向量索引
    index_path = get_data_path(FAISS_PATH)
    if not os.path.exists(index_path):
        print("尚未建立記憶資料庫")
        return []
    
    index = faiss.read_index(index_path)
    
    # 查詢最近的記憶
    D, I = index.search(query_vector, top_k)
    
    # 載入文字對應內容
    text_path = get_data_path(TEXT_PATH)
    with open(text_path, "r", encoding="utf-8") as f:
        memory_texts = json.load(f)
    
    # 回傳匹配結果
    results = []
    for dist, i in zip(D[0], I[0]):
        if dist < distance_threshold and i < len(memory_texts):
            results.append({
                "score": dist,
                "summary": memory_texts[i]["summary"],
                "raw": memory_texts[i]["raw"]
            })
    
    return results


if __name__ == '__main__':
    usr_prompt = '還記得我的職業嗎？'
    out_path_1 = 'generate_output_1.txt'
    out_path_2 = 'generate_output_2.txt'
    send_to_gemini(usr_prompt, out_path_2)
    store_data(usr_prompt, out_path_1)
