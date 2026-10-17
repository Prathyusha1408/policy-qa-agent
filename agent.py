"""Step 2: the LangGraph agent.
 
retrieve -> grade_documents -> generate            (relevant context found)
                            -> rewrite_query -> retrieve   (nothing relevant, retry once)
                            -> fallback          (still nothing: say "not found")
"""
from pathlib import Path
from typing import List, Optional, TypedDict
 
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
 
import config
 
 
class GraphState(TypedDict, total=False):
    question: str                 # current search query (may be rewritten)
    original_question: str        # what the user actually asked
    history: List[BaseMessage]
    documents: List[Document]     # retrieved chunks
    relevant_docs: List[Document] # chunks that passed grading
    answer: str
    sources: List[str]
    retries: int
    fallback: bool
 
 
class Grade(BaseModel):
    relevant: bool = Field(description="True if the passage helps answer the question")
 
 
GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You grade whether a policy document passage is relevant to a user's question. "
     "Answer relevant=true only if the passage states a specific fact needed to answer the "
     "question, even if it answers only part of it. Answer relevant=false if the passage is "
     "merely on the same general topic but does not state the fact the question asks about."),
    ("human", "Question: {question}\n\nPassage:\n{passage}"),
])
 
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's question as a better search query for a policy document "
     "database. Use likely formal terms and synonyms. Return only the new query."),
    ("human", "Question: {question}\nPrevious query (found nothing useful): {previous_query}"),
])
 
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about policy documents using ONLY the context below. "
     "Cite the source after each claim in the form [file.pdf, p.3]. "
     "If the context only partly answers the question, answer what you can and "
     "say what is missing. If the question asks for a specific deadline, amount, or rule "
     "that the context does not explicitly state, say the documents don't specify it. "
     "Never substitute a different deadline or rule from a related process, and never "
     "invent details.\n\nContext:\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])
 
 
def source_label(doc: Document) -> str:
    name = Path(doc.metadata.get("source", "unknown")).name
    return f"{name}, p.{int(doc.metadata.get('page', 0)) + 1}"
 
 
def format_context(docs: List[Document]) -> str:
    return "\n\n".join(f"[{source_label(d)}]\n{d.page_content}" for d in docs)
 
 
def get_retriever():
    config.require_api_key()
    store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(model=config.EMBEDDING_MODEL),
        persist_directory=str(config.PERSIST_DIR),
    )
    return store.as_retriever(search_kwargs={"k": config.TOP_K})
 
 
def build_graph(llm=None, retriever=None):
    """Compile the agent. llm / retriever can be injected (used for testing)."""
    if llm is None:
        config.require_api_key()
        llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0)
    if retriever is None:
        retriever = get_retriever()
 
    grader = GRADE_PROMPT | llm.with_structured_output(Grade)
    rewriter = REWRITE_PROMPT | llm | StrOutputParser()
    generator = ANSWER_PROMPT | llm
 
    # ----- nodes -----
    def retrieve(state: GraphState) -> dict:
        return {"documents": retriever.invoke(state["question"])}
 
    def grade_documents(state: GraphState) -> dict:
        docs = state["documents"]
        inputs = [
            {"question": state["original_question"], "passage": d.page_content}
            for d in docs
        ]
        grades = grader.batch(inputs) if inputs else []
        return {"relevant_docs": [d for d, g in zip(docs, grades) if g.relevant]}
 
    def rewrite_query(state: GraphState) -> dict:
        new_query = rewriter.invoke({
            "question": state["original_question"],
            "previous_query": state["question"],
        })
        return {"question": new_query.strip(), "retries": state.get("retries", 0) + 1}
 
    def generate(state: GraphState) -> dict:
        docs = state["relevant_docs"]
        response = generator.invoke({
            "context": format_context(docs),
            "history": state.get("history", []),
            "question": state["original_question"],
        })
        sources = list(dict.fromkeys(source_label(d) for d in docs))
        return {"answer": response.content, "sources": sources, "fallback": False}
 
    def fallback(state: GraphState) -> dict:
        return {"answer": config.NOT_FOUND_MESSAGE, "sources": [], "fallback": True}
 
    # ----- routing -----
    def route_after_grading(state: GraphState) -> str:
        if state["relevant_docs"]:
            return "generate"
        if state.get("retries", 0) < config.MAX_QUERY_REWRITES:
            return "rewrite_query"
        return "fallback"
 
    # ----- wiring -----
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate)
    graph.add_node("fallback", fallback)
 
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"generate": "generate", "rewrite_query": "rewrite_query", "fallback": "fallback"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)
    return graph.compile()
 
 
def ask(graph, question: str, history: Optional[List[BaseMessage]] = None) -> dict:
    """Run one question through the agent and return a clean result dict."""
    final = graph.invoke({
        "question": question,
        "original_question": question,
        "history": (history or [])[-config.MAX_HISTORY_MESSAGES:],
        "retries": 0,
    })
    return {
        "answer": final["answer"],
        "sources": final.get("sources", []),
        "fallback": final.get("fallback", False),
        "retries": final.get("retries", 0),
        "relevant_chunks": len(final.get("relevant_docs", [])),
        "final_query": final["question"],
    }
 
 
if __name__ == "__main__":
    # Quick terminal chat, similar to the original RAG demo
    from langchain_core.messages import AIMessage, HumanMessage
 
    agent = build_graph()
    chat_history: List[BaseMessage] = []
    print("Policy Q&A Agent ready. Type 'exit' to quit.")
    while True:
        user_input = input("\nUser: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        result = ask(agent, user_input, chat_history)
        chat_history += [HumanMessage(content=user_input), AIMessage(content=result["answer"])]
        print(f"Assistant: {result['answer']}")
        if result["sources"]:
            print(f"Sources: {', '.join(result['sources'])}")
        if result["retries"]:
            print(f"(query rewritten {result['retries']}x -> '{result['final_query']}')")
 