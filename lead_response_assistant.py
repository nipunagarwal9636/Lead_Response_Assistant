import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.schema import StrOutputParser

load_dotenv()
Groq_API_key="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"


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5,
    api_key=Groq_API_key
)


if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )

memory = st.session_state.memory


prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a professional customer response assistant.

Guidelines:
- ALWAYS consider previous conversation context.
- Understand short follow-up replies.
- Respond empathetically.
- Ask relevant questions if needed.
- Avoid hallucinated claims.
- Give safe practical suggestions.
"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{query}")
])

chain = prompt | llm | StrOutputParser()


def evaluate_response(query, response):
    hallucination_score = 0
    risky_words = ["guarantee", "definitely", "always", "never"]

    if any(w in response.lower() for w in risky_words):
        hallucination_score += 0.5

    relevance_score = 1 if any(
        w in response.lower() for w in query.lower().split()[:3]
    ) else 0.7

    completeness_score = 1 if "?" in response else 0.6

    return {
        "hallucination_score": round(hallucination_score, 2),
        "relevance_score": relevance_score,
        "completeness_score": completeness_score
    }


def generate_response(query):

    history = memory.load_memory_variables({})["history"]

    response = chain.invoke({
        "query": query,
        "history": history
    })

    memory.save_context({"input": query}, {"output": response})

    metrics = evaluate_response(query, response)

    return response, metrics


st.set_page_config(page_title="AI Lead Response Assistant")

st.title("AI Lead Response Assistant")
st.caption("Context-aware chatbot with evaluation metrics")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Enter customer enquiry...")

if user_input:
    response, metrics = generate_response(user_input)

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("AI", response, metrics))

for chat in st.session_state.chat_history:
    if len(chat) == 2:
        sender, msg = chat
        with st.chat_message(sender):
            st.write(msg)
    else:
        sender, msg, metrics = chat
        with st.chat_message(sender):
            st.write(msg)
            st.caption(
                f"Hallucination Score: {metrics['hallucination_score']} | "
                f"Relevance Score: {metrics['relevance_score']} | "
                f"Completeness Score: {metrics['completeness_score']}"
            )
