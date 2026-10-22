import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from src.hybrid_retriever import retriever

llm = ChatOllama(model=os.getenv("LLM_MODEL", "llama3.1:latest"),
                 temperature=0.2, num_ctx=4096)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Ты отвечаешь на вопросы на основе переданного контекста.\n"
     "Используй только информацию из контекста. Если её недостаточно, честно скажи об этом.\n"
     "Отвечай на русском языке, точно и по делу. Ссылайся на источники в виде [номер].\n\n"
     "Контекст:\n{context}"),
    ("human", "Вопрос: {question}")])

answer_chain = prompt | llm | StrOutputParser()


def format_documents(documents):
    return "\n\n".join(f"[{i}] Источник: {d.metadata['title']}\n{d.page_content}"
                       for i, d in enumerate(documents, 1))


def generate_answer(question):
    documents = retriever.invoke(question)
    answer = answer_chain.invoke({"context": format_documents(documents), "question": question})
    return answer, documents


if __name__ == "__main__":
    ans, docs = generate_answer(input("Введите вопрос: "))
    print(ans)
    for i, d in enumerate(docs, 1):
        print(f"{i}. {d.metadata['title']} ({d.metadata['url']})")
