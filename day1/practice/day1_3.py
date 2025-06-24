# %%
from dotenv import load_dotenv
load_dotenv()

# Langsmith tracing 여부를 확인 (true: langsmith 추척 활성화, false: langsmith 추척 비활성화)
import os
print(os.getenv('LANGSMITH_TRACING'))

# %% 
from langchain_openai import ChatOpenAI

# LLM model
llm = ChatOpenAI(
    model="gpt-4.1-mini", 
    temperature=0.3, 
    top_p=0.95,
    )

# 모델에 프롬프트를 입력
response = llm.invoke("탄소의 원자 번호는 무엇인가요?")

# 응답 객체 확인
response


# %%
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 모델 선언
llm = ChatOpenAI(
    model="gpt-4-1106-preview",  # 최신 지원 모델명으로 교체
    temperature=0.3,
    top_p=0.95,
)

# 템플릿 정의
template = """
당신의 성격은 {character} 입니다. 당신은 {situation} 상황에 처해있습니다.
{situation}에서 어떻게 해야할지의 질문에 대해 대답해주세요.

질문: {question}
답변:"""

prompt = PromptTemplate.from_template(template)

# 출력 파서
output_parser = StrOutputParser()

# 체인 구성
chain = prompt | llm | output_parser

# 실행
response = chain.invoke({
    "character": "이직이 너무 하고싶었던 ENTP",
    "situation": "입사동기끼리 입사 1주년 파티",
    "question": "어떤 축하 케이크가 좋을까"
})

# 결과 출력
print(response)

# %%
