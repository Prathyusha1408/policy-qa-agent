"""Step 3: Streamlit chat UI.  Run with:  streamlit run app.py"""
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

import config
import ingest
from agent import ask, build_graph

st.set_page_config(page_title="Policy Q&A Agent", page_icon="📄")
st.title("📄 Policy Q&A Agent")
st.caption("RAG + LangGraph · OpenAI · ChromaDB")

config.DATA_DIR.mkdir(exist_ok=True)

# ---------- Sidebar: documents ----------
with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload policy PDFs", type="pdf", accept_multiple_files=True)
    for f in uploads or []:
        (config.DATA_DIR / f.name).write_bytes(f.getbuffer())

    pdfs = sorted(p.name for p in config.DATA_DIR.glob("*.pdf"))
    st.write(f"**{len(pdfs)}** PDF(s) in data/")
    for name in pdfs:
        st.caption(f"• {name}")

    if st.button("Build / rebuild index", disabled=not pdfs):
        with st.spinner("Embedding documents..."):
            n_chunks = ingest.build_index()
        st.cache_resource.clear()
        st.success(f"Indexed {n_chunks} chunks")

    if st.button("Clear chat"):
        st.session_state.pop("messages", None)
        st.rerun()

if not config.index_exists():
    st.info("Add PDFs in the sidebar and click **Build / rebuild index** to start.")
    st.stop()


@st.cache_resource
def get_agent():
    return build_graph()


agent = get_agent()
st.session_state.setdefault("messages", [])

# ---------- Chat ----------
def render(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        meta = msg.get("meta")
        if meta:
            with st.expander("Sources & agent trace"):
                if meta["sources"]:
                    st.write("**Sources:** " + "; ".join(meta["sources"]))
                st.write(f"Relevant chunks used: {meta['relevant_chunks']}")
                st.write(f"Query rewrites: {meta['retries']}")
                if meta["retries"]:
                    st.write(f"Final search query: `{meta['final_query']}`")
                if meta["fallback"]:
                    st.write("Fallback triggered: no relevant evidence found.")


for m in st.session_state.messages:
    render(m)

if question := st.chat_input("Ask a question about your policy documents"):
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    render(user_msg)

    history = [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in st.session_state.messages[:-1]
    ]
    with st.spinner("Searching and grading the documents..."):
        result = ask(agent, question, history)

    bot_msg = {"role": "assistant", "content": result["answer"], "meta": result}
    st.session_state.messages.append(bot_msg)
    render(bot_msg)
