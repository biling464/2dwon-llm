import streamlit as st
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import pandas as pd
import datetime
import plotly.express as px
import re

# ======== 環境變數讀取與 GPT 初始化 ========
load_dotenv("llm.env")
api_key = os.getenv("OPENAI_API_KEY")
api_version = os.getenv("OPENAI_API_VERSION")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

if not all([api_key, api_version, azure_endpoint, deployment_name]):
    st.error("❌ 無法讀取環境變數，請確認 llm.env 檔案是否與此 .py 同資料夾，且內容無誤")
    st.stop()

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint
)

# ======== 初始化頁面狀態 ========
if "page" not in st.session_state:
    st.session_state.page = "input"
if "num_items" not in st.session_state:
    st.session_state.num_items = 3
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = ""
if "today_data" not in st.session_state:
    st.session_state.today_data = None
if "added_items" not in st.session_state:
    st.session_state.added_items = []

# ======== 主頁：輸入頁面 ========
if st.session_state.page == "input":
    st.set_page_config(page_title="LLM 消費小幫手", page_icon="💸")
    st.title("💸 LLM 消費小幫手")
    st.markdown("請依序輸入每一筆消費項目與金額，然後按下分析")

    items = []
    for i in range(st.session_state.num_items):
        col1, col2 = st.columns([2, 1])
        with col1:
            item = st.text_input(f"項目 {i+1}", key=f"item_{i}")
        with col2:
            price = st.number_input(f"金額", min_value=0, key=f"price_{i}")
        if item and price > 0:
            items.append({"項目": item, "金額": int(price)})

    if st.button("➕ 新增一筆消費欄位"):
        st.session_state.num_items += 1
        st.rerun()

    if st.button("🔍 開始分析"):
        if not items:
            st.warning("請至少輸入一筆有效的消費")
        else:
            combined_input = "、".join([f"{i['項目']} {i['金額']}" for i in items])

            prompt = f"""請將以下花費文字分類，格式如下：\n【花費分類與金額】\n1. 飲食：\n   - 珍奶：55元\n   - 雞排：85元\n   - 合計：140元\n2. 交通：\n   - 捷運車資：45元\n   - 合計：45元\n【總花費】：185元\n消費內容如下：{combined_input}"""

            try:
                response = client.chat.completions.create(
                    model=deployment_name,
                    messages=[
                        {"role": "system", "content": "你是一個精準的消費分類助理，負責把日常花費進行分類並計算總花費"},
                        {"role": "user", "content": prompt}
                    ]
                )
                result = response.choices[0].message.content
                st.session_state.analysis_result = result

                # 解析分類資料（簡化版）
                categories = {}
                current_category = None
                for line in result.splitlines():
                    cat_match = re.match(r"\d+\. (.+?)：", line)
                    item_match = re.match(r"\s*- (.+?)：\d+元", line)
                    if cat_match:
                        current_category = cat_match.group(1)
                    elif item_match and current_category:
                        item_name = item_match.group(1)
                        categories[item_name] = current_category

                # 加入分類與日期
                today = datetime.date.today().isoformat()
                df = pd.DataFrame([{**i, "日期": today, "分類": categories.get(i['項目'], "未分類")} for i in items])
                st.session_state.today_data = df
                st.session_state.added_items = df  # 儲存當次分析資料（用於高亮）

                # 儲存歷史紀錄（避免重複）
                history_path = "history.csv"
                if os.path.exists(history_path):
                    old = pd.read_csv(history_path)
                    new = pd.concat([old, df], ignore_index=True)
                    new.drop_duplicates(subset=["日期", "項目", "金額"], keep="last", inplace=True)
                else:
                    new = df
                new.to_csv(history_path, index=False, encoding='utf-8-sig')

                st.session_state.page = "result"
                st.rerun()

            except Exception as e:
                st.error(f"⚠️ 發生錯誤：{e}")

# ======== 結果頁面 ========
elif st.session_state.page == "result":
    st.set_page_config(page_title="分析結果", page_icon="📊")
    st.title("📊 分析結果與統計圖")

    st.markdown("### 🧾 GPT 分類分析")
    st.markdown(st.session_state.analysis_result)

    # 🔍 AI 消費建議
    st.markdown("---")
    st.markdown("### 🤖 AI 消費提醒")
    if st.session_state.today_data is not None:
        total_by_cat = st.session_state.today_data.groupby("分類")["金額"].sum().sort_values(ascending=False)
        top_cat = total_by_cat.idxmax()
        top_amount = total_by_cat.max()
        st.info(f"💡 今日花費最多為「{top_cat}」類，共 {top_amount} 元，請留意是否為非必要支出。")

    # 🥧 圖表
    st.markdown("---")
    st.markdown("### 🥧 今日消費圓餅圖")
    if st.session_state.today_data is not None:
        fig = px.pie(st.session_state.today_data, names="項目", values="金額", title="今日各項目花費佔比")
        st.plotly_chart(fig)

    # 📈 每日趨勢圖
    st.markdown("---")
    st.markdown("### 📈 每日總花費趨勢圖")
    if os.path.exists("history.csv"):
        history = pd.read_csv("history.csv")
        trend = history.groupby("日期")["金額"].sum().reset_index()
        fig_line = px.line(trend, x="日期", y="金額", markers=True, title="每日總花費")
        st.plotly_chart(fig_line)

    # 📖 歷史紀錄（高亮新增項目）
    st.markdown("---")
    st.markdown("### 📖 歷史消費紀錄")
    if os.path.exists("history.csv"):
        history = pd.read_csv("history.csv")

        def highlight_new_rows(row):
            if st.session_state.added_items is not None:
                for _, new_row in st.session_state.added_items.iterrows():
                    if row["項目"] == new_row["項目"] and row["金額"] == new_row["金額"] and row["日期"] == new_row["日期"]:
                        return ["background-color: #ffd966"] * len(row)
            return [""] * len(row)

        st.dataframe(history.style.apply(highlight_new_rows, axis=1), use_container_width=True)

        csv = history.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button("📥 匯出所有消費紀錄 CSV", csv, file_name="history.csv", mime="text/csv")

    if st.button("🔁 回到輸入頁面"):
        st.session_state.page = "input"
        st.rerun()
