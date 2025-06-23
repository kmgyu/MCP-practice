from typing import Literal, Optional
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
# from langchain.schema import SystemMessage, HumanMessage
# from langchain.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END

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
    return state.copy(update={"context": "[dummy retrieved context]"})

def general_llm(state: GraphState) -> GraphState:
    return state.copy(update={"answer": "[dummy general answer]"})

def llm_answer(state: GraphState) -> GraphState:
    return state.copy(update={"answer": "[dummy llm answer based on context]"})

def search_on_web(state: GraphState) -> GraphState:
    return state.copy(update={"context": "[dummy web search results]"})

def relevance_check(state: GraphState) -> GraphState:
    return state.copy(update={"relevance": "grounded"})

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
def load_translator():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    model_name = "NHNDQ/nllb-finetuned-en2ko"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("translation", model=model, tokenizer=tokenizer, device=0)

translator = load_translator()

def translate(text, src_lang, tgt_lang):
    return translator(text, src_lang=src_lang, tgt_lang=tgt_lang)[0]["translation_text"]

import chainlit as cl

@cl.on_message
async def run_graph(message: cl.Message):
    translated_input = translate(message.content, 'kor_Kore', 'eng_Latn')
    
    if not translated_input.strip() or translated_input.strip() == ".":
        await cl.Message(content="입력 내용을 번역할 수 없습니다. 다시 입력해 주세요.").send()
        return

    state = GraphState(question=translated_input)
    result = app.invoke(state)
    answer = result.get("answer") or "[No response]"

    translated_output = translate(answer, 'eng_Latn', 'kor_Kore')
    await cl.Message(content=translated_output).send()
