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
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI

# 모델 및 출력 파서
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
parser = StrOutputParser()

# 디버깅용 로그 함수
def log(tag):
    return RunnableLambda(lambda x: (print(f"[{tag}] {x}"), x)[1])

# 후처리: 문장 끝에 "뿅" 붙이기
add_ppyong = RunnableLambda(lambda x: x.strip() + " 뿅")

# 공통 프롬프트 템플릿 (character/situation/question 포함)
template = """
당신은 {character}입니다.
현재 상황은 {situation}입니다.
아래 질문에 대해 어떻게 하면 좋을지 답변해주세요.

질문: {question}
답변:"""
prompt = PromptTemplate.from_template(template)

# 체인 구성 (하나만 만들고 재사용)
chain = prompt | log("🧾 Prompt") | llm | log("📡 LLM") | parser | add_ppyong

# 질문 3개 정의
questions = {
    "스트레스 해소": "스트레스를 덜 받으려면 어떻게 해야 할까?",
    "게임 욕구": "게임이 너무 하고 싶은데 저거 다 하면서 언제 게임도 하지?",
    "청소 회피": "청소가 너무 하기 싫은데 뭘 하면 청소를 안 미루고 할 수 있을까?"
}

# 병렬 입력 구성: 질문만 바꾸고 나머지는 고정
parallel_inputs = {
    name: {
        "character": "무기력한 상태의 INTP",
        "situation": "사내교육, 야근, 스터디 일정이 겹쳐 매우 바쁜 상황",
        "question": q
    }
    for name, q in questions.items()
}

# RunnableParallel 구성: 각 입력에 대해 동일한 체인을 실행
parallel_chain = RunnableParallel({
    name: RunnableLambda(lambda x, q=input_data: q) | chain
    for name, input_data in parallel_inputs.items()
})

# 실행
result = parallel_chain.invoke({})
print("\n 최종 결과:")
for key, value in result.items():
    print(f"- {key}: {value}")



# %%
import asyncio
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

# 모델 및 출력 파서
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
parser = StrOutputParser()

# 후처리: 문장 끝에 "뿅" 붙이기
add_ppyong = RunnableLambda(lambda x: x.strip() + " 뿅")

# 공통 프롬프트 템플릿
template = """
당신은 {character}입니다.
현재 상황은 {situation}입니다.
아래 질문에 대해 어떻게 하면 좋을지 답변해주세요.

질문: {question}
답변:"""
prompt = PromptTemplate.from_template(template)

# 체인 구성
chain = prompt | llm | parser | add_ppyong

# 질문 목록
questions = {
    "스트레스 해소": "스트레스를 덜 받으려면 어떻게 해야 할까?",
    "게임 욕구": "게임이 너무 하고 싶은데 저거 다 하면서 언제 게임도 하지?",
    "청소 회피": "청소가 너무 하기 싫은데 뭘 하면 청소를 안 미루고 할 수 있을까?"
}

# 공통 입력값
base_input = {
    "character": "무기력한 상태의 INTP",
    "situation": "사내교육, 야근, 스터디 일정이 겹쳐 매우 바쁜 상황"
}

# 비동기 실행 함수
async def run_parallel():
    tasks = [
        chain.ainvoke({**base_input, "question": question})
        for question in questions.values()
    ]
    results = await asyncio.gather(*tasks)
    return dict(zip(questions.keys(), results))

# 실제 실행
if __name__ == "__main__":
    final_result = asyncio.run(run_parallel())

    print("\n✅ 최종 결과:")
    for key, value in final_result.items():
        print(f"- {key}: {value}")
