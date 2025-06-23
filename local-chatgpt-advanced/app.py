from typing import Literal, Optional
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
# from langchain.schema import SystemMessage, HumanMessage
# from langchain.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END

import torch

# 전역 디바이스 변수 선언
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1. Define State
class GraphState(BaseModel):
    question: str
    q_type: Optional[Literal["about_address", "search", "general"]] = None
    context: Optional[str] = None
    answer: Optional[str] = None
    relevance: Optional[Literal["grounded", "notGrounded", "notSure"]] = None

# Step 2. Define Output Schema for LLM
class QTypeSchema(BaseModel):
    q_type: Literal["about_address", "search", "general"]
    
    # model_config = {
    #     'from_attributes': True
    # }

parser = PydanticOutputParser(pydantic_object=QTypeSchema)

from langchain_community.llms import Ollama
chat = Ollama(model="llama3.2-vision")

# Step 3. Decision Maker Node
prompt = PromptTemplate.from_template(
    """
    You are a classification model that must decide the type of question.
    Choose one of the following:
    - "about_address"
    - "search"
    - "general"

    Respond only in JSON like: {{"q_type": "search"}}

    Question:
    {question}
    """
)

decision_chain = prompt | chat | parser

def decision_maker(state: GraphState) -> GraphState:
    output: QTypeSchema = decision_chain.invoke({"question": state.question})
    state_dict = state.dict()
    state_dict["q_type"] = output.q_type
    return GraphState(**state_dict)

# Step 4. Conditional Routing Logic
def decision_making(state: GraphState) -> str:
    match state.q_type:
        case "about_address":
            return "about_address"
        case "search":
            return "search"
        case "general":
            return "general"
        case _:
            raise ValueError(f"Unexpected q_type: {state.q_type}")


# Step 5. Dummy Node Examples
def retrieve_document(state: GraphState) -> GraphState:
    print("[retrieve_document] 문서 검색 중")
    # 실제 DB/벡터 스토어에서 검색된 내용이라고 가정
    context = f"서울의 본사 주소는 ... 입니다."
    return state.copy(update={"context": context})


def general_llm(state: GraphState) -> GraphState:
    print("[general_llm] LLM 응답 생성 중")
    response = chat.invoke(state.question)  # Ollama LLM
    return state.copy(update={"answer": response})

def llm_answer(state: GraphState) -> GraphState:
    print("[llm_answer] context 기반 LLM 응답 생성 중")
    prompt = f"""문맥: {state.context}\n질문: {state.question}\n답변:"""
    response = chat.invoke(prompt)
    return state.copy(update={"answer": response})


def search_on_web(state: GraphState) -> GraphState:
    print("[search_on_web] 웹 검색 결과를 context에 저장")
    context = f"{state.question}에 대한 검색 결과: ..."
    return state.copy(update={"context": context})


def relevance_check(state: GraphState) -> GraphState:
    print("[relevance_check] context의 신뢰도 판단")
    judge_prompt = f"""문맥이 질문에 적절한가요?

    문맥:
    {state.context}

    질문:
    {state.question}

    답변: grounded / notGrounded / notSure
    """
    result = chat.invoke(judge_prompt)
    result = result.strip().lower()

    if "grounded" in result:
        relevance = "grounded"
    elif "notgrounded" in result:
        relevance = "notGrounded"
    else:
        relevance = "notSure"

    return state.copy(update={"relevance": relevance})


def is_relevant(state: GraphState) -> str:
    match state.relevance:
        case "grounded": return "llm_answer"
        case "notGrounded" | "notSure": return "search_on_web"
        case _: raise ValueError(f"Unexpected relevance: {state.relevance}")

# Step 6. Build Graph
workflow = StateGraph(GraphState)
workflow.set_entry_point("decision_maker")

workflow.add_node("decision_maker", decision_maker)
workflow.add_node("retrieve", retrieve_document)
workflow.add_node("general_llm", general_llm)
workflow.add_node("llm_answer", llm_answer)
workflow.add_node("search_on_web", search_on_web)
workflow.add_node("relevance_check", relevance_check)

workflow.add_conditional_edges("decision_maker", decision_making, {
    "about_address": "retrieve",
    "search": "search_on_web",
    "general": "general_llm"
})

workflow.add_edge("retrieve", "relevance_check")
workflow.add_edge("search_on_web", "relevance_check")
workflow.add_conditional_edges("relevance_check", is_relevant, {
    "grounded": "llm_answer",
    "notGrounded": "search_on_web",
    "notSure": "search_on_web"
})

workflow.add_edge("llm_answer", END)
workflow.add_edge("general_llm", END)

# Compile
app = workflow.compile()

# Chainlit Integration
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_translator():
    model_name = "yanolja/EEVE-Korean-Instruct-2.8B-v1.0"
    # removed  trust_remote_code=True parameter. and then it work. i don't know.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(DEVICE)
    
    return tokenizer, model

tokenizer, model = load_translator()

def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    print(f"input : {text}")
    if src_lang.startswith("kor") and tgt_lang.startswith("eng"):
        prompt = f"{text} → 영어로 번역해줘."
    elif src_lang.startswith("eng") and tgt_lang.startswith("kor"):
        prompt = f"Translate to Korean: {text}"
    else:
        raise ValueError("지원하지 않는 언어 쌍입니다.")

    print("start - model input, tokenizer")
    model_inputs = tokenizer(prompt, return_tensors="pt")
    model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}  # ⬅ 반드시 옮기기
    print('model, generating')
    print(model_inputs)
    outputs = model.generate(**model_inputs, max_new_tokens=256)
    print('model generate complete, batch decoding')
    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f'output_text : {output_text}')
    
    # 프롬프트 제거하고 응답만 추출
    return output_text.split("Assistant:")[-1].strip()


import chainlit as cl

@cl.on_message
async def run_graph(message: cl.Message):
    print(f"current device: {DEVICE}")
    print("원본 입력:", message.content)

    translated_input = translate(message.content, 'kor_Kore', 'eng_Latn')
    print("번역된 영어 입력:", translated_input)

    if not translated_input.strip() or translated_input.strip() == ".":
        await cl.Message(content="입력 내용을 인식하지 못했습니다. 다시 시도해주세요.").send()
        return

    state = GraphState(question=translated_input)
    result = app.invoke(state)
    print("Graph 실행 결과:", result)

    answer = result.get("answer") or "[No response]"
    print("최종 영어 응답:", answer)

    translated_output = translate(answer, 'eng_Latn', 'kor_Kore')
    print("최종 한국어 응답:", translated_output)

    await cl.Message(content=translated_output).send()

