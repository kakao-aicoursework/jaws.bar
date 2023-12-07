import json
import openai
import tkinter as tk
import pandas as pd
from tkinter import scrolledtext
import tkinter.filedialog as filedialog
import os
from dotenv import load_dotenv
import chromadb
import re

load_dotenv()
openai.api_key = os.environ.get("api_key")

client = chromadb.PersistentClient()

collection = client.get_or_create_collection(
    name="kakao-channel",
    metadata={"hnsw:space": "cosine"}# l2 is the default #cosine 유사도 사용
)

# 데이터 정제 및 저장
def save_data():
    data_file = open('data.txt', 'r', encoding='utf-8')
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
    doc_meta = []

    for key, value in data.items():
        document = f"{key}:{value}"
        documents.append(document)
        meta = {
            "desc": value
        }
        doc_meta.append(meta)

    # DB 저장
    collection.add(
        documents=documents,
        metadatas=doc_meta,
        ids=list(data.keys())
    )


# response에 CSV 형식이 있는지 확인하고 있으면 저장하기
# def save_to_csv(df):
#     file_path = filedialog.asksaveasfilename(defaultextension='.csv')
#     if file_path:
#         df.to_csv(file_path, sep=';', index=False, lineterminator='\n')
#         return f'파일을 저장했습니다. 저장 경로는 다음과 같습니다. \n {file_path}\n'
#     return '저장을 취소했습니다'


# def save_playlist_as_csv(playlist_csv):
#     if ";" in playlist_csv:
#         lines = playlist_csv.strip().split("\n")
#         csv_data = []

#         for line in lines:
#             if ";" in line:
#                 csv_data.append(line.split(";"))

#         if len(csv_data) > 0:
#             df = pd.DataFrame(csv_data[1:], columns=csv_data[0])
#             return save_to_csv(df)

#     return f'저장에 실패했습니다. \n저장에 실패한 내용은 다음과 같습니다. \n{playlist_csv}'


def send_message(message_log, gpt_model="gpt-3.5-turbo", temperature=0.1):
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=message_log,
        temperature=temperature,
        # functions=functions,
        # function_call='auto',
    )

    # response_message = response["choices"][0]["message"]

    # if response_message.get("function_call"):
    #     available_functions = {
    #         "find_db": find_db,
    #     }
    #     function_name = response_message["function_call"]["name"]
    #     fuction_to_call = available_functions[function_name]
    #     function_args = json.loads(response_message["function_call"]["arguments"])
    #     # 사용하는 함수에 따라 사용하는 인자의 개수와 내용이 달라질 수 있으므로
    #     # **function_args로 처리하기
    #     function_response = fuction_to_call(**function_args)

    #     # 함수를 실행한 결과를 GPT에게 보내 답을 받아오기 위한 부분
    #     message_log.append(response_message)  # GPT의 지난 답변을 message_logs에 추가하기
    #     message_log.append(
    #         {
    #             "role": "function",
    #             "name": function_name,
    #             "content": function_response,
    #         }
    #     )  # 함수 실행 결과도 GPT messages에 추가하기
    #     response = openai.ChatCompletion.create(
    #         model=gpt_model,
    #         messages=message_log,
    #         temperature=temperature,
    #     )  # 함수 실행 결과를 GPT에 보내 새로운 답변 받아오기
    return response.choices[0].message.content


def main():
    save_data()
    message_log = [
        {
            "role": "system",
            "content": '''
            너는 카카오톡이라는 메신저의 채널에 대해서 설명해주는 챗봇이다.
            '''
        }
    ]

    # functions = [
    #     {
    #         "name": "save_playlist_as_csv",
    #         "description": "Saves the given playlist data into a CSV file when the user confirms the playlist.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "playlist_csv": {
    #                     "type": "string",
    #                     "description": "A playlist in CSV format separated by ';'. It must contains a header and the release year should follow the 'YYYY' format. The CSV content must starts with a new line. The header of the CSV file must be in English and it should be formatted as follows: 'Title;Artist;Released'.",
    #                 },
    #             },
    #             "required": ["playlist_csv"],
    #         },
    #     }
    # ]

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

        vector_res = collection.query(
            query_texts=[user_input],
            n_results=1,
        )

        srchres = []
        for v in vector_res['documents'][0]:
            item = v.split(':')
            srchres.append({
                "title" : item[0].strip(),
                "desc" : item[1].strip(),
            })

        message_log.append({"role": "assistant", "content": f"{srchres}"})
        message_log.append({"role": "user", "content": user_input})
        
        response = send_message(message_log)
        thinking_popup.destroy()

        message_log.append({"role": "assistant", "content": response})

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