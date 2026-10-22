import streamlit as st
from src.rag import generate_answer

st.set_page_config(page_title="RAG по статьям Хабра", layout="wide")
st.title("RAG-система на LangChain")
st.write("Гибридный поиск (BM25 FTS5 + FAISS), переранжирование и генерация ответа Llama 3.1.")

question = st.text_input("Введите ваш вопрос:", placeholder="Например: что такое LPQ и HPQ?")

if st.button("Получить ответ") and question.strip():
    with st.spinner("Ищу информацию и формирую ответ..."):
        try:
            answer, documents = generate_answer(question)
        except Exception as e:
            st.error(f"Произошла ошибка: {e}")
            st.stop()

    st.subheader("Ответ")
    st.markdown(answer)

    st.subheader("Найденные документы")
    for i, d in enumerate(documents, 1):
        m = d.metadata
        with st.expander(f"{i}. {m['title']} (score: {m['score']:.3f}, {m['search_type']})"):
            st.markdown(f"[{m['url']}]({m['url']})  \nХабы: {', '.join(m['hubs'])}")
            st.write(d.page_content)
