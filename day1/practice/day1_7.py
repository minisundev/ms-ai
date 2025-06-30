# %%
import os
from glob import glob

from pprint import pprint
import json

import warnings
warnings.filterwarnings("ignore")

from langfuse.langchain import CallbackHandler

# LangChain 콜백 핸들러 생성
langfuse_handler = CallbackHandler()

from langfuse import Langfuse
import os

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

print("✅ 연결 성공?", langfuse.auth_check())  # True가 나와야 정상

# 텍스트 프롬프트 생성
langfuse.create_prompt(
    name="mbti-analyzer-text",  # 프롬프트 이름
    type="text",
    prompt="{mbtiType} 성격 유형에 기반해 이 사람이 {situation} 상황에서 보일 행동을 설명해주세요.",
    labels=["mbti", "analysis", "text"],
    tags=["mbti", "personality", "behavior"],
    config={
        "model": "gpt-4.1-mini",
        "temperature": 0.7
    }
)

# 챗 프롬프트 생성
langfuse.create_prompt(
    name="mbti-analyzer-chat",
    type="chat",
    prompt=[
        {
            "role": "system",
            "content": "당신은 MBTI 성격 분석 전문가입니다. 정확하고 구체적으로 설명해주세요."
        },
        {
            "role": "user",
            "content": "{mbtiType} 유형의 사람은 {situation} 상황에서 어떻게 행동하나요?"
        }
    ], # type: ignore
    labels=["mbti", "analysis", "chat"],
    tags=["mbti", "personality", "chat"],
    config={
        "model": "gpt-4.1-mini",
        "temperature": 0.7
    }
)

# Langfuse는 프롬프트 버전 관리 + 실행 추적 플랫폼이고
# 지금 너는 prompt registry에 프롬프트를 등록한 거야.


# %% 가져와서 실행하기~~
from langfuse import Langfuse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# 1. 프롬프트 불러오기
langfuse = Langfuse()
raw_prompt = langfuse.get_prompt("mbti-analyzer-text", label="latest")  # or version, etc

# 2. LangChain PromptTemplate로 바꾸기
template = PromptTemplate.from_template(raw_prompt.prompt)

# 3. LLM과 연결
llm = ChatOpenAI(model=raw_prompt.config["model"], temperature=raw_prompt.config["temperature"])

# 4. 실행
response = llm.invoke(
    template.format(
        mbtiType="INTP",
        situation="중간 발표에서 조원들이 다 말을 안 듣고 있을 때"
    )
)

print(response.content)

# %%
# %% Chat 프롬프트 가져와서 실행하기~~
from langchain_core.prompts import ChatPromptTemplate

# 1. 프롬프트 불러오기
raw_chat_prompt = langfuse.get_prompt("mbti-analyzer-chat", label="latest")

# 2. ChatPromptTemplate로 변환
chat_prompt = ChatPromptTemplate.from_messages(raw_chat_prompt.prompt)

# 3. LLM 생성 (model + temperature는 config에서)
chat_llm = ChatOpenAI(
    model=raw_chat_prompt.config["model"],
    temperature=raw_chat_prompt.config["temperature"]
)

# 4. 입력 변수 채워서 메시지 포맷팅
messages = chat_prompt.format_messages(
    mbtiType="ENTP",
    situation="프로젝트 마감일이 하루 남았는데 아직 할 일이 많을 때"
)

# 5. 실행
response = chat_llm.invoke(messages)
print(response.content)

# %%
