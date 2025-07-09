import streamlit as st
from Gemini.gemini_api import send_to_gemini, store_data
import os

st.set_page_config(page_title="🧠 MemoraGem", layout="centered")

st.title("🧠 MemoraGem")
st.write("一個會記得你問題的 Gemini 小助手。")

# 初始化對話歷史
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 組對話上下文（最多取 10 條）
history_prompt = ""
for chat in st.session_state.chat_history[-10:]:
    history_prompt += f"使用者：{chat['user']}\nGemini：{chat['ai']}\n"

# 聊天紀錄顯示區塊（可滾動）
chat_container = st.container()

with chat_container:
    st.markdown("### 🗣️ 對話紀錄")
    for chat in st.session_state.chat_history:
        st.markdown(f"**👤 使用者：** {chat['user']}")
        st.markdown(f"**🤖 Gemini：** {chat['ai']}")
        if chat.get("mem_result"):
            st.markdown(f"<div style='color:gray;font-size:0.85em'>🔍 參考：{chat['mem_result']}</div>", unsafe_allow_html=True)
        st.markdown("---")

# 自動捲動到底部的 JS
st.markdown("""
    <script>
        const chatContainer = window.parent.document.querySelector('.main');
        chatContainer.scrollTo(0, chatContainer.scrollHeight);
    </script>
""", unsafe_allow_html=True)

# 使用者輸入欄固定在下方
user_input = st.text_input("💬 請輸入你的問題")

if st.button("送出") and user_input:
    # 加歷史紀錄進 prompt
    contents = []
    for chat in st.session_state.chat_history:
        contents.append({"role": "user", "parts": [chat["user"]]})
        contents.append({"role": "model", "parts": [chat["ai"]]})
    
    results = send_to_gemini(user_input, contents)
    store_data(user_input)

    # 加入對話歷史
    summary = [result['summary'] for result in results['ref']]
    st.session_state.chat_history.append({
        "user": user_input,
        "ai": results['response'],
        "mem_result": summary,
    })

    st.rerun()  # 自動刷新並往下捲
