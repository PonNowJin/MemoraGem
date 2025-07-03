import streamlit as st
from Gemini.gemini_api import send_to_gemini, store_data
import os

st.set_page_config(page_title="🧠 MemoraGem", layout="centered")

st.title("🧠 MemoraGem")
st.write("一個會記得你問題的 Gemini 小助手。")

# 初始化對話歷史
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 組對話上下文
history_prompt = ""
count = 0
for chat in st.session_state.chat_history:
    if count <= 10:
        history_prompt += f"使用者：{chat['user']}\nGemini：{chat['ai']}\n"
        count += 1
    else:
        break
    
# 使用者輸入
user_input = st.text_input("💬 請輸入你的問題")

# 顯示按鈕
if st.button("送出") and user_input:
    # 送給 Gemini 回應
    prompt = f"歷史紀錄：{history_prompt}, {user_input}"
    results = send_to_gemini(prompt)
    print(prompt)
    store_data(user_input)

    # 存進對話歷史
    st.session_state.chat_history.append({
        "user": user_input,
        "ai": results['response'],
        "mem_result": results['ref'],
    })

# 顯示聊天歷史
if st.session_state.chat_history:
    st.markdown("### 🗣️ 對話紀錄")
    for chat in st.session_state.chat_history:
        st.markdown(f"**👤 使用者：** {chat['user']}")
        st.markdown(f"**🤖 Gemini：** {chat['ai']}")
        st.markdown(f"** 參考：** {chat['mem_result']}")
        st.markdown("---")
else:
    st.info("請輸入提問內容並送出")
