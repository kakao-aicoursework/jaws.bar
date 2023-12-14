import os
from dotenv import load_dotenv
import re
from typing import List
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI

# CHROMA_PERSIST_DIR = os.path.join(DATA_DIR, "upload/chroma-persist")
CHROMA_COLLECTION_NAME = "kakao"
CHROMA_PERSIST_DIR = "upload/chroma-persist"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("api_key")

_db = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=OpenAIEmbeddings(),
    collection_name=CHROMA_COLLECTION_NAME,
)
_retriever = _db.as_retriever()

def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template

def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path)
        ),
        output_key=output_key,
        verbose=True,
    )

def query_db(query: str, use_retriever: bool = False) -> list[str]:
    if use_retriever:
        docs = _retriever.get_relevant_documents(query)
    else:
        docs = _db.similarity_search(query)

    str_docs = [doc.page_content for doc in docs]
    return str_docs

# 데이터 정제 및 저장
def save_data(file_name: str):
    data_file = open(file_name, 'r', encoding='utf-8')
    line = data_file.readline()
    data = {}
    current_section = ""
    while (line):
        parsed_line = re.sub(r'\s+', '', line)
        if parsed_line.startswith("#"):
            current_section = parsed_line[1:]
            data[current_section] = ""
        elif current_section != "":
            data[current_section] += parsed_line
        line = data_file.readline()

    documents = []
    doc_meta = []

    for key, value in data.items():
        document = f"{key}:{value}"
        documents.append(document)
        meta = {
            "subject": file_name.split("/")[-1].split(".")[0],
            "title": key
        }
        doc_meta.append(meta)

    Chroma.from_documents(
        documents,
        OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        meta=doc_meta,
        ids=list(data.keys())
    )

save_data("data/channel.txt")
save_data("data/sync.txt")
save_data("data/social.txt")

llm = ChatOpenAI(temperature=0.1, max_tokens=1000, model="gpt-3.5-turbo")

first_chain = create_chain(
    llm=llm,
    template_path="prompts/first.txt",
    output_key="output"
)
second_chain = create_chain(
    llm=llm,
    template_path="prompts/second.txt",
    output_key="output"
)
default_chain = create_chain(
    llm=llm, template_path="prompts/defatul_response.txt", output_key="output"
)

from langchain.memory import ConversationBufferMemory, FileChatMessageHistory

HISTORY_DIR = "history/"

def load_conversation_history(conversation_id: str):
    file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    return FileChatMessageHistory(file_path)


def log_user_message(history: FileChatMessageHistory, user_message: str):
    history.add_user_message(user_message)


def log_bot_message(history: FileChatMessageHistory, bot_message: str):
    history.add_ai_message(bot_message)


def get_chat_history(conversation_id: str):
    history = load_conversation_history(conversation_id)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="user_message",
        chat_memory=history,
    )

    return memory.buffer

def gernerate_answer(user_message, conversation_id: str='fa1010') -> dict[str, str]:
    history_file = load_conversation_history(conversation_id)

    context = dict(user_message=user_message)
    context["input"] = context["user_message"]
    context["chat_history"] = get_chat_history(conversation_id)

    context["related_documents"] = query_db(context["user_message"])
    has_value = first_chain.run(context)
    answer = ""
    if has_value == "Y":
        answer = second_chain.run(context)
    else:
        answer = default_chain.run(context)

    log_user_message(history_file, user_message)
    log_bot_message(history_file, answer)
    return {"answer": answer}
