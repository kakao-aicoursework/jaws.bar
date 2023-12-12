from dto import ChatbotRequest
from samples import list_card
import time
import logging
import openai
from dotenv import load_dotenv
import os
import mission
import requests

# 환경 변수 처리 필요!
load_dotenv()
openai.api_key = os.environ.get("api_key")
logger = logging.getLogger("Callback")

def callback_handler(request: ChatbotRequest) -> dict:
    # ===================== start =================================
    output_text = mission.task(request.userRequest.utterance)

   # 참고링크 통해 payload 구조 확인 가능
    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": output_text
                    }
                }
            ]
        }
    }
    # ===================== end =================================
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/ai_chatbot_callback_guide
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/answer_json_format

    time.sleep(1.0)

    url = request.userRequest.callbackUrl
    print(payload)
    print(len(str(payload)))
    if url:
        with requests.post(url=url, json=payload, verify=False) as resp:
            resp.json()