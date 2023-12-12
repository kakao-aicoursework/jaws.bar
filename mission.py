import openai
import tkinter as tk
from tkinter import scrolledtext
import os
from dotenv import load_dotenv
import re
from typing import List
import tiktoken
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("api_key")

llm = ChatOpenAI(temperature=0.8)

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

def build_summarizer(llm):
    system_message = "assistant는 카카오톡이라는 메신저의 싱크 API에 대해서 설명해주는 챗봇이다. assistant는 user의 질문을 함께 받는 데이터를 토대로 잘 대답해라."
    system_message_prompt = SystemMessage(content=system_message)
    
    human_template = "{text}\n---\n위 데이터를 토대로 다음 질문에 대답해 줘. 답변은 짧을수록 좋아.\n---\n{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,
                                                    human_message_prompt])

    chain = LLMChain(llm=llm, prompt=chat_prompt)
    return chain

def truncate_text(text, max_tokens=5000):
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:  # 토큰 수가 이미 3000 이하라면 전체 텍스트 반환
        return text
    return enc.decode(tokens[:max_tokens])

summarizer = build_summarizer(llm)

# 데이터 정제 및 저장
def get_data():
    data_file = open('sync.txt', 'r', encoding='utf-8')
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
    # 벡터로 변환 저장할 텍스트 데이터로 ChromaDB에 Embedding 데이터가 없으면 자동으로 벡터로 변환해서 저장
    documents = []

    for key, value in data.items():
        document = f"{key}:{value}"
        documents.append(document)

    return ' '.join(documents)

data = get_data()

def task(question):

    full_content_truncated = truncate_text(data, max_tokens=3500)

    summary = summarizer.run(text=full_content_truncated, question=question)

    return summary

def send_message(message_log, gpt_model="gpt-3.5-turbo", temperature=0.1):
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=message_log,
        temperature=temperature,
    )
    return response.choices[0].message.content


def main():
    def show_popup_message(window, message):
        popup = tk.Toplevel(window)
        popup.title("")

        # 팝업 창의 내용
        label = tk.Label(popup, text=message, font=("맑은 고딕", 12))
        label.pack(expand=True, fill=tk.BOTH)

        # 팝업 창의 크기 조절하기
        window.update_idletasks()
        popup_width = label.winfo_reqwidth() + 20
        popup_height = label.winfo_reqheight() + 20
        popup.geometry(f"{popup_width}x{popup_height}")

        # 팝업 창의 중앙에 위치하기
        window_x = window.winfo_x()
        window_y = window.winfo_y()
        window_width = window.winfo_width()
        window_height = window.winfo_height()

        popup_x = window_x + window_width // 2 - popup_width // 2
        popup_y = window_y + window_height // 2 - popup_height // 2
        popup.geometry(f"+{popup_x}+{popup_y}")

        popup.transient(window)
        popup.attributes('-topmost', True)

        popup.update()
        return popup

    def on_send():
        user_input = user_entry.get()
        user_entry.delete(0, tk.END)

        if user_input.lower() == "quit":
            window.destroy()
            return
        
        conversation.config(state=tk.NORMAL)  # 이동
        conversation.insert(tk.END, f"You: {user_input}\n", "user")  # 이동
        thinking_popup = show_popup_message(window, "처리중...")
        window.update_idletasks()
        # '생각 중...' 팝업 창이 반드시 화면에 나타나도록 강제로 설정하기

        response = task(user_input)
        thinking_popup.destroy()

        # 태그를 추가한 부분(1)
        conversation.insert(tk.END, f"gpt assistant: {response}\n", "assistant")
        conversation.config(state=tk.DISABLED)
        # conversation을 수정하지 못하게 설정하기
        conversation.see(tk.END)

    window = tk.Tk()
    window.title("GPT AI")

    font = ("맑은 고딕", 10)

    conversation = scrolledtext.ScrolledText(window, wrap=tk.WORD, bg='#f0f0f0', font=font)
    # width, height를 없애고 배경색 지정하기(2)
    conversation.tag_configure("user", background="#c9daf8")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.tag_configure("assistant", background="#e4e4e4")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    # 창의 폭에 맞추어 크기 조정하기(4)

    input_frame = tk.Frame(window)  # user_entry와 send_button을 담는 frame(5)
    input_frame.pack(fill=tk.X, padx=10, pady=10)  # 창의 크기에 맞추어 조절하기(5)

    user_entry = tk.Entry(input_frame)
    user_entry.pack(fill=tk.X, side=tk.LEFT, expand=True)

    send_button = tk.Button(input_frame, text="Send", command=on_send)
    send_button.pack(side=tk.RIGHT)

    window.bind('<Return>', lambda event: on_send())
    window.mainloop()


if __name__ == "__main__":
    main()