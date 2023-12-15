import os
from dotenv import load_dotenv
from typing import List
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from upload import collection, channel_db, channel_retriever, sync_db, sync_retriever, social_db, social_retriever

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("api_key")

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

def query_channel_db(query: str, use_retriever: bool = False) -> list[str]:
    if use_retriever:
        docs = channel_retriever.get_relevant_documents(query)
    else:
        docs = channel_db.similarity_search(query)

    str_docs = [doc.page_content for doc in docs]
    return str_docs

def query_sync_db(query: str, use_retriever: bool = False) -> list[str]:
    if use_retriever:
        docs = sync_retriever.get_relevant_documents(query)
    else:
        docs = sync_db.similarity_search(query)

    str_docs = [doc.page_content for doc in docs]
    return str_docs

def query_social_db(query: str, use_retriever: bool = False) -> list[str]:
    if use_retriever:
        docs = social_retriever.get_relevant_documents(query)
    else:
        docs = social_db.similarity_search(query)

    str_docs = [doc.page_content for doc in docs]
    return str_docs

def query_collection(query: str) -> List[dict]:
    vector_res = collection.query(
            query_texts=[query],
            n_results=3,
        )

    srchres = []
    print(vector_res)
    for v in vector_res['documents'][0]:
        item = v.split(':')
        srchres.append({
            "title" : item[0].strip(),
            "desc" : item[1].strip(),
        })

    return srchres

llm = ChatOpenAI(temperature=0.1, max_tokens=1000, model="gpt-3.5-turbo")

check_channel_chain = create_chain(
    llm=llm, template_path="prompts/check_channel.txt", output_key="output"
)
extract_channel_question_chain = create_chain(
    llm=llm, template_path="prompts/extract_channel_question.txt", output_key="output"
)
check_sync_chain = create_chain(
    llm=llm, template_path="prompts/check_sync.txt", output_key="output"
)
extract_sync_question_chain = create_chain(
    llm=llm, template_path="prompts/extract_sync_question.txt", output_key="output"
)
check_social_chain = create_chain(
    llm=llm, template_path="prompts/check_social.txt", output_key="output"
)
extract_social_question_chain = create_chain(
    llm=llm, template_path="prompts/extract_social_question.txt", output_key="output"
)
default_chain = create_chain(
    llm=llm, template_path="prompts/default_response.txt", output_key="output"
)
final_response_chain = create_chain(
    llm=llm, template_path="prompts/final_response.txt", output_key="output"
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

    context["channel_related_documents"] = []
    context["sync_related_documents"] = []
    context["social_related_documents"] = []

    # context["related_documents"] = query_collection(context["user_message"])
    # print(context["related_documents"])
    is_related_channel = check_channel_chain.run(context)
    if is_related_channel == "Y":
        channel_question = extract_channel_question_chain.run(context)
        print(channel_question)
        context["channel_related_documents"] = query_channel_db(channel_question)
        print(context["channel_related_documents"])

    is_related_sync = check_sync_chain.run(context)
    if is_related_sync == "Y":
        sync_question = extract_sync_question_chain.run(context)
        print(sync_question)
        context["sync_related_documents"] = query_sync_db(sync_question)
        print(context["sync_related_documents"])

    is_related_social = check_social_chain.run(context)
    if is_related_social == "Y":
        social_question = extract_social_question_chain.run(context)
        print(social_question)
        context["social_related_documents"] = query_social_db(social_question)
        print(context["social_related_documents"])

    answer = final_response_chain.run(context)
    print(answer)

    log_user_message(history_file, user_message)
    log_bot_message(history_file, answer)
    return answer
