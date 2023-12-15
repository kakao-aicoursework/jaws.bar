from langchain.document_loaders import (
    TextLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os
import chromadb
import re

CHROMA_COLLECTION_NAME = "kakao"
CHROMA_PERSIST_DIR = "chromadb"


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("api_key")

channel_db = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=OpenAIEmbeddings(),
    collection_name="channel",
)
channel_retriever = channel_db.as_retriever()

sync_db = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=OpenAIEmbeddings(),
    collection_name="sync",
)
sync_retriever = sync_db.as_retriever()

social_db = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=OpenAIEmbeddings(),
    collection_name="social",
)
social_retriever = social_db.as_retriever()

client = chromadb.PersistentClient()

collection = client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}# l2 is the default #cosine 유사도 사용
)

def upload_embedding_from_file(file_path, collection_name):
    if TextLoader is None:
        raise ValueError("Not supported file type")
    documents = TextLoader(file_path).load()

    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)
    print(docs, end='\n\n\n')

    Chroma.from_documents(
        docs,
        OpenAIEmbeddings(),
        collection_name=collection_name,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print('db success')

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

    collection.add(
        documents=documents,
        metadatas=doc_meta,
        ids=list(map(lambda x:(file_name.split("/")[-1].split(".")[0] + ":" + x),data.keys()))
    )


def main():
    upload_embedding_from_file('data/channel.txt', "channel")
    upload_embedding_from_file('data/sync.txt', "sync")
    upload_embedding_from_file('data/social.txt', "social")
    # save_data('data/channel.txt')
    # save_data('data/sync.txt')
    # save_data('data/social.txt')


if __name__ == "__main__":
    main()