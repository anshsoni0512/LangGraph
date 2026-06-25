# Complete LangGraph Guide

A progressive series of Jupyter notebooks exploring LangChain and LangGraph — from basic LLM calls to a full Corrective RAG (CRAG) pipeline.

## Tech Stack

- **LangGraph / LangChain** — graph-based agent orchestration
- **Groq** — LLM inference (LLaMA 3.1 8B Instant)
- **Pinecone** — vector database (AWS serverless)
- **HuggingFace** — embeddings (`BAAI/bge-small-en-v1.5`)
- **Pydantic** — structured LLM output

## Setup

```bash
python -m venv meraenv
meraenv\Scripts\activate

pip install langgraph langchain langchain-groq langchain-pinecone
pip install langchain-huggingface sentence-transformers
pip install langchain-experimental pymupdf python-dotenv pinecone
```

Create a `.env` file in the root:
```
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

---

## Notebooks

---

### `0_Output_of_llm.ipynb` — LLM Basics & Structured Output
- Initialize `ChatGroq` and invoke the LLM
- Parse `.content` from `AIMessage` response
- Use `with_structured_output(PydanticModel)` to get typed responses
- Use `Literal` types to constrain LLM output to fixed values

---

### `1_simple_llm_workflow.ipynb` — First LangGraph Workflow
- Define state with `TypedDict`
- Create a single node function that reads and updates state
- Connect nodes: `START → node → END` using `add_edge()`
- Compile with `.compile()` and invoke with `.invoke()`

**Key concept:** Every node takes state as input and returns a dict of updated fields only.

---

### `2_prompt_chain_workflow.ipynb` — Sequential Prompt Chaining
- Multi-node linear pipeline: `outline → blog → evaluate`
- Each node enriches the state with new fields
- Demonstrates how state accumulates across steps

**Key concept:** Chain nodes to break complex tasks into focused steps.

---

### `3_parallel_workflow.ipynb` — Parallel Node Execution
- Multiple nodes start from `START` simultaneously
- Uses `Annotated[list[int], operator.add]` to merge results from parallel branches
- All parallel branches converge at a final node

**Key concept:** `Annotated` with `operator.add` is how LangGraph aggregates values from parallel nodes.

---

### `4_conditional_workflow.ipynb` — Conditional Routing
- `add_conditional_edges(node, routing_fn)` routes to different nodes based on state
- Routing function returns the name of the next node as a string
- Real example: sentiment classification routes to different response handlers

**Key concept:** Routing functions return node names — this is how branching logic works in LangGraph.

---

### `5_Iterative_workflow.ipynb` — Loops & Iterative Refinement
- Conditional edge loops back to an earlier node for refinement
- Iteration counter in state prevents infinite loops
- Pattern: `generate → evaluate → optimize → evaluate` (loop until approved or max iterations)

**Key concept:** Loops are just conditional edges that point backwards in the graph.

---

### `6_tools_langraph.ipynb` — Tool Calling
- Define custom tools with `@tool` decorator
- Bind tools to LLM: `llm.bind_tools(tools)`
- `ToolNode` from prebuilt handles tool execution automatically
- `tools_condition` routes between chat and tools nodes

**Key concept:** `tools_condition` returns `"tools"` if the LLM called a tool, or `END` if it responded directly.

---

### `7_chatbot_langraph.ipynb` — Stateful Chatbot
- State uses `Annotated[list[BaseMessage], add_messages]` — `add_messages` appends instead of overwriting
- `MemorySaver` checkpointer keeps conversation in memory
- `thread_id` in config identifies separate conversations
- Chatbot is an agent with tool access

**Key concept:** `add_messages` is a reducer that appends new messages to the list rather than replacing it.

---

### `8_sqlite3.ipynb` — SQLite Persistence
- `SqliteSaver` replaces `MemorySaver` for disk-based persistence
- Chat history survives kernel restarts
- Same `thread_id` in config retrieves full conversation history

**Key concept:** Swap `MemorySaver` → `SqliteSaver` for production-grade persistence with zero code change.

---

### `9_subgraph.ipynb` — Subgraphs
- Define a smaller `StateGraph` for a reusable subtask (e.g. translation)
- Call `subflow.invoke(...)` from inside a parent node
- State is mapped manually between parent and subgraph

**Key concept:** Subgraphs encapsulate logic into reusable components that can be called from any parent node.

---

### `10_subgraph_combine.ipynb` — Subgraph as Node
- Add a compiled subgraph directly as a node: `graph.add_node("name", subgraph)`
- Cleaner than manual invocation — LangGraph handles state passing automatically
- Enables true graph composition

**Key concept:** `add_node("name", compiled_subgraph)` is the preferred way to nest graphs.

---

### `11_shortmemory.ipynb` — Short-Term Memory (Message Trimming)
- `trim_messages()` limits conversation history by token budget
- `strategy='last'` keeps most recent messages
- Prevents context window overflow in long conversations

**Key concept:** Always trim messages before invoking the LLM in long-running conversations.

---

### `12_longmemory.ipynb` — Long-Term Memory (Semantic Store)
- `InMemoryStore` stores facts/preferences with namespaces
- `store.put(namespace, key, value)` to save memories
- `store.search(namespace, query=..., limit=N)` for semantic retrieval using embeddings
- Memories persist across conversations

**Key concept:** Namespace as `("users", "user_id")` isolates each user's memories.

---

### `13_summarization.ipynb` — Conversation Summarization
- When message count exceeds threshold, a summarization node fires
- Old messages are deleted with `RemoveMessage(id=...)`
- Summary is stored in state and injected as context in future turns

**Key concept:** Summarization compresses history — keeps context without blowing up the token count.

---

### `14_HITL_langraph.ipynb` — Human-in-the-Loop
- `interrupt("message")` pauses graph execution and waits for human input
- Graph state is saved by checkpointer during the pause
- `Command(resume=user_input)` resumes execution from where it stopped
- Real example: LLM drafts a post, human approves/rejects before it's published

**Key concept:** `interrupt()` + `Command(resume=...)` is the standard HITL pattern in LangGraph.

---

### `15_crag.ipynb` — Corrective RAG (CRAG) — Full Pipeline

Full RAG system with document quality correction using LangGraph.

**Pipeline:**
```
retrieve → score → [Correct]   → refine → generate_answer
                 → [Incorrect] → web_search → refine → generate_answer
                 → [Ambiguous] → ambiguous  → refine → generate_answer
```

**Key components:**

| Component | Detail |
|-----------|--------|
| PDF Loader | `PyMuPDFLoader` — 49-page LangChain/LangGraph interview Q&A book |
| Chunking | `SemanticChunker` with `percentile=95` breakpoints → 198 chunks |
| Embeddings | `BAAI/bge-small-en-v1.5` (384 dimensions) |
| Vector DB | Pinecone serverless index on AWS |
| Doc Scoring | LLM grades each chunk 0.0–1.0 using structured output (`Score` Pydantic model) |
| Verdict | `Correct` if any score > 0.7, `Incorrect` if all < 0.3, else `Ambiguous` |
| Sentence Filter | Decomposes chunks into sentences, LLM judges each `True/False` for relevance |
| Answer Gen | Cleaned, filtered context passed to LLM for final answer |

---

### `practice.ipynb` — LinkedIn Post Generator (Practice)
- Parallel nodes from `START`: generate post + summarize topic simultaneously
- Structured evaluation gives a rating (1–10) with feedback
- Iterative loop: optimize → re-evaluate until rating > 8 or 5 iterations reached

---

## Common Patterns

```python
# State definition
class MyState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # appends
    scores: Annotated[list[int], operator.add]            # merges from parallel nodes

# Node function
def my_node(state: MyState):
    return {"field": new_value}   # only return what changed

# Graph
graph = StateGraph(MyState)
graph.add_node("node1", fn1)
graph.add_edge(START, "node1")
graph.add_conditional_edges("node1", routing_fn)
workflow = graph.compile(checkpointer=MemorySaver())
workflow.invoke({"question": "..."}, config={"configurable": {"thread_id": "1"}})
```

---

## Knowledge Base

`Langchain and LangGraph Interview QA.pdf` — 100 interview Q&As used as the knowledge base in `15_crag.ipynb`.

