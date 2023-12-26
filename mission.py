import os
from dotenv import load_dotenv
from typing import List
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from upload import collection, channel_db, channel_retriever, sync_db, sync_retriever, social_db, social_retriever
from langchain.agents.tools import Tool
from langchain.agents import initialize_agent

CUSTOM_PREFIX= """Answer the following questions in as much detail as possible.You have access to the following tools:"""

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

llm = ChatOpenAI(temperature=0.1, max_tokens=1000, model="gpt-3.5-turbo")

tools =[
        Tool(
            name="channel",
            func=query_channel_db,
            description="This is useful when you need information about your KakaoTalk channel. You should ask targeted questions.",
        ),
        Tool(
            name="sync",
            func=query_sync_db,
            description="This is useful when you need information about your KakaoTalk sync. You should ask targeted questions.",
        ),
        Tool(
            name="social",
            func=query_social_db,
            description="This is useful when you need information about your KakaoTalk social. You should ask targeted questions.",
        )
        ]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True, agent_kwargs={'prefix': CUSTOM_PREFIX})

parse_intent_chain = create_chain(
    llm=llm, template_path="prompts/parse_intent.txt", output_key="output"
)

default_response_chain = create_chain(
    llm=llm, template_path="prompts/default_response.txt", output_key="output"
)

answer_chain = create_chain(
    llm=llm, template_path="prompts/answer_with_function_result.txt", output_key="output"
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
    # context["chat_history"] = get_chat_history(conversation_id)
    context["chat_history"] = ""
    context["intent_list"] = read_prompt_template("prompts/intent_list.txt")

    intent = parse_intent_chain.run(context)
    answer = ""

    if intent == "question":
        context["function_result"] = agent.run(user_message)
        answer = answer_chain.run(context)
    else:
        answer = default_response_chain.run(context)

    log_user_message(history_file, user_message)
    log_bot_message(history_file, answer)
    return answer
