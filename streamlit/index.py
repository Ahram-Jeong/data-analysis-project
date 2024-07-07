import streamlit as st

# --- page
st.set_page_config(page_title = "Index | Jeong Ahram")

st.info("💰 비트코인과 안정자산, 위험자산의 상관관계 분석")
st.write("""
- **데이터 산출 기간**
    - 제도 편입 전 : 2014.09.01 ~ 2019.12.31
    - 제도 편입 후 : 2021.01.01 ~ 2024.06.06
    - 코로나19 기간은 변동성에 대한 상관관계가 명확하지 않아 제외
- **안전자산 비교군** : 금, 달러
- **위험자산 비교군** : 원유
- **데이터 출처** : Yahoo Finance
""")