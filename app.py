import streamlit as st
from Gemini.gemini_api import send_to_gemini, store_data
import os

st.set_page_config(page_title="🧠 MemoraGem", layout="centered")

st.title("🧠 MemoraGem")
st.write("一個會記得你問題的 Gemini 小助手。")

# 使用者輸入
user_input = st.text_input("💬 請輸入你的問題")

# 顯示按鈕
if st.button("送出") and user_input:
    # 送給 Gemini 回應
    gemini_reply = send_to_gemini(user_input)
    store_data(user_input)

    # 顯示結果
    st.markdown("### 🤖 Gemini 的回覆")
    st.write(gemini_reply)

    '''
    if memory_results:
        st.markdown("### 🧠 找到的記憶")
        for m in memory_results:
            st.markdown(f"- {m['summary']}")
    '''
else:
    st.info("請輸入提問內容並送出")

