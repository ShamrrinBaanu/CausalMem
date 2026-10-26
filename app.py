import logging

import streamlit as st

from chat_service import process_chat
from config import settings
from llm_service import user_facing_error
from mem0_engine import get_memories
from temporal_engine import current_states, timeline
from visualization import graph_figure

logger = logging.getLogger("causalmem.mem0")

st.set_page_config(page_title="CausalMem",page_icon="◉",layout="wide")
st.markdown("<style>.stApp{background:#0b1220}.block-container{max-width:1450px;padding-top:2rem}.hero{padding:1.2rem 1.5rem;border:1px solid #243247;border-radius:16px;background:linear-gradient(110deg,#111d32,#122b3b)}.muted{color:#9fb0c4}</style>",unsafe_allow_html=True)
st.markdown("<div class='hero'><h1>◉ CausalMem</h1><div class='muted'>Persistent personal memory with explicit, evidence-linked state transitions.</div></div>",unsafe_allow_html=True)
st.caption("Legacy Streamlit view. The React console is the primary interface; see README.md.")
user_id=st.sidebar.text_input("User ID",value=settings.default_user,help="Memory and causal history use this user identifier.").strip() or settings.default_user
st.sidebar.caption(f"LLM: {settings.provider} / {settings.model}")
if settings.provider=="openai" and not settings.api_key: st.sidebar.warning("Set OPENAI_API_KEY in .env to enable extraction and answers.")
left,right=st.columns([1.2,1])
with left:
    st.subheader("Chat with your memory")
    chats=st.session_state.setdefault("chats",{})
    chat=chats.setdefault(user_id,[])
    for item in chat:
        with st.chat_message(item["role"]): st.write(item["content"])
    prompt=st.chat_input("Tell me something or ask about your history…")
    if prompt:
        chat.append({"role":"user","content":prompt})
        try:
            response=process_chat(prompt,user_id)
            answer=response["answer"]
            if response["response_type"]=="answer":
                st.session_state.setdefault("last_queries",{})[user_id]={
                    "analysis":response["query_analysis"],"evidence":response["evidence"],
                    "answer_diagnostics":response["diagnostics"]}
            else:
                st.session_state["last_ingest"]={
                    "memories":response["memories_added"],"states":response["states"],
                    "transitions":response["transitions"]}
        except Exception as exc:
            logger.error("Chat operation failed (source=%s, type=%s)",
                         getattr(exc,"source","chat"),
                         getattr(exc,"cause_type",type(exc).__name__))
            answer=user_facing_error(exc)
        chat.append({"role":"assistant","content":answer}); st.rerun()
with right:
    st.subheader("Mem0 memories")
    try:
        memories=get_memories(user_id)
        for m in memories[-8:][::-1]: st.info(f"**{m.get('memory','')}**\n\nID: `{m.get('id','')}` · {m.get('created_at','')}")
        if not memories: st.caption("No memories yet. Tell the chat something you want to remember.")
    except Exception as exc:
        logger.error("Mem0 memory listing failed (%s)", type(exc).__name__, exc_info=True)
        st.warning(f"Mem0 unavailable ({type(exc).__name__}): {user_facing_error(exc)}")
    st.subheader("Active states")
    states=current_states(user_id)
    if not states: st.caption("No states yet.")
    for s in states: st.write(f"🟢 **{s.state_value}** · {s.state_type} · {s.confidence:.0%}")
    with st.expander("State history / timeline"):
        history=timeline(user_id)
        for s in history: st.write(f"{s.valid_from.isoformat()} — {s.state_value} ({s.status})")
        if not history: st.caption("No timeline yet.")
    st.subheader("Causal graph")
    st.plotly_chart(graph_figure(user_id),use_container_width=True)
    with st.expander("Query analysis and evidence"):
        data=st.session_state.get("last_queries",{}).get(user_id)
        if data:
            st.json({"query_type":data["analysis"]["category"],"retrieval":data["evidence"]["retrieval_sources"]})
            st.write("Answer validation (text-free)")
            st.json(data.get("answer_diagnostics",{}))
            st.write("Retrieved Mem0 memories");st.json(data["evidence"]["memories"])
            st.write("States, transitions, and path");st.json({"states":data["evidence"]["states"],"transitions":data["evidence"]["transitions"],"causal_path":data["evidence"]["causal_path"]})
        else: st.caption("Ask a question to inspect routing and evidence.")
