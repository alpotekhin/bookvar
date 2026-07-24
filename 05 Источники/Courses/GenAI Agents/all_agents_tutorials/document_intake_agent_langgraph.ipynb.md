---
title: "Document Intake Agent: LLM-Ready Markdown from Any Office Document"
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: bd681451b254ac1a790e947b581d3997ab35013d
---

> [!note] Original source material
> This page preserves [`all_agents_tutorials/document_intake_agent_langgraph.ipynb`](https://github.com/NirDiamant/GenAI_Agents/blob/bd681451b254ac1a790e947b581d3997ab35013d/all_agents_tutorials/document_intake_agent_langgraph.ipynb) from
> *GenAI Agents* at commit `bd681451b254ac1a790e947b581d3997ab35013d`. License:
> [Custom non-commercial license](https://github.com/NirDiamant/GenAI_Agents/blob/bd681451b254ac1a790e947b581d3997ab35013d/LICENSE). Bookvar changed only
> publication markup, link paths, and characters required for safe rendering.



### Overview

Agents in production receive files, not clean prompts: a quarterly report as docx, a deck as pptx, a spreadsheet as xlsx, a contract as pdf. Language models read none of these natively. This tutorial builds a document intake agent with LangGraph that accepts an arbitrary incoming file, routes it through the right converter, and grounds its answers in a faithful markdown version of the document instead of a lossy text scrape.

The conversion step runs as a tool call against the [hushvert conversion API](https://hushvert.com/for-developers), a hosted service for exactly the formats that are painful to convert in pure Python (native office documents and PDFs with real layouts). The same capability is available to MCP hosts like Claude Code and Cursor through a one-block config, shown near the end.

### Motivation

The naive way to feed a docx or pdf to a model is to rip the raw text out and paste it into the prompt. That silently destroys the structure business documents live on:

- **Tables** collapse into an undifferentiated stream of cell values, so "which region grew fastest" becomes a token-counting guessing game.
- **Reading order** breaks on multi-column layouts, interleaving sentences from different columns.
- **Headings and lists** flatten away, so the model loses the document's own organization.

Converting to markdown first preserves that structure in a form models are heavily trained on: pipe tables stay tables, headings stay headings, and the agent can quote figures instead of hallucinating around them.

![What each path preserves](https://github.com/NirDiamant/GenAI_Agents/raw/bd681451b254ac1a790e947b581d3997ab35013d/images/document-intake-table-loss.svg)

### Key Components

1. **Format router**: inspects the incoming file and decides whether to read it directly (already text), convert it (office documents, PDF), or refuse it with a useful message. The routing table comes from the converter's live formats endpoint, not a hardcoded list.
2. **Conversion tool**: a small client for the hushvert conversion API following its submit, upload, poll, download flow.
3. **Analyst node**: gpt-4o-mini answering questions grounded only in the converted markdown.
4. **LangGraph state graph** wiring the three together with one conditional edge.

### Agent Architecture

![Document Intake Agent](https://github.com/NirDiamant/GenAI_Agents/raw/bd681451b254ac1a790e947b581d3997ab35013d/images/document-intake-agent.svg)

The routing decision is deterministic on purpose. A ReAct-style agent could decide "should I convert this?" with an LLM call on every run, but format routing is a lookup, not a judgment call: putting it in a conditional edge makes the behavior testable and saves a model round trip on every document.

## Setting Up the Environment

### Install dependencies

```python
!pip install -q langgraph langchain-openai langchain-core requests python-docx python-dotenv
```

### Configure API keys

Two keys are needed:

- **OpenAI**: for the analyst model (gpt-4o-mini).
- **hushvert**: create a free account and an API key at [hushvert.com/developers/keys](https://hushvert.com/developers/keys). Sign-in is a one-time email code (no password), and keys (they look like `hv_live_...`) require a confirmed email. New accounts include a free monthly allowance of server conversions, so this tutorial runs end to end without payment.

```python
import os
from getpass import getpass

from dotenv import load_dotenv

# check for .env file at the repo root
if os.path.exists("../.env"):
    load_dotenv("../.env")

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")
if not os.environ.get("HUSHVERT_API_KEY"):
    os.environ["HUSHVERT_API_KEY"] = getpass("Enter your hushvert API key (hv_live_...): ")
```

### Import libraries

```python
import pathlib
import re
import time
import zipfile

import requests
from typing_extensions import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
```

## The Conversion Layer

### Discover what the API converts

The formats endpoint is public and returns every server-side pair as a slug like `docx-to-md`, together with its size cap and credit cost. We fetch it once and treat it as the routing table, so the agent's idea of "convertible" can never drift from what the API actually supports.

```python
HUSHVERT_API = "https://hushvert.com/api"

pairs = requests.get(f"{HUSHVERT_API}/v1/formats", timeout=30).json()["pairs"]
server_pairs = {p["pair"] for p in pairs}

md_inputs = sorted(p["from"] for p in pairs if p["to"] == "md")
print(f"{len(pairs)} server pairs available")
print("Convertible to markdown:", ", ".join(md_inputs))
```

### The conversion client

The API is a three-step flow, the standard shape for services that move real file bytes:

1. **Submit** the pair and byte count. The response carries a `jobId` and a presigned `uploadUrl`.
2. **Upload** the bytes with a plain HTTP PUT. They go directly to object storage, not through the API host.
3. **Poll** the job until it reports `done`, then download the result from the presigned `downloadUrl`.

The optional `Idempotency-Key` header makes retries safe: a resubmission with the same key returns the same job and is never charged twice. Pass a stable key if you wrap this call in retry logic.

![The conversion flow](https://github.com/NirDiamant/GenAI_Agents/raw/bd681451b254ac1a790e947b581d3997ab35013d/images/document-intake-conversion-flow.svg)

```python
def convert_file(path: str, to: str, idempotency_key: str | None = None) -> bytes:
    """Convert a local file via the hushvert API: submit, upload, poll, download."""
    src = pathlib.Path(path)
    data = src.read_bytes()
    pair = f"{src.suffix.lstrip('.').lower()}-to-{to}"

    headers = {"Authorization": f"Bearer {os.environ['HUSHVERT_API_KEY']}"}
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key

    # 1. Declare the pair and byte count; get a job id and a presigned upload URL.
    submit = requests.post(
        f"{HUSHVERT_API}/v1/conversions",
        json={"pair": pair, "bytes": len(data)},
        headers=headers,
        timeout=30,
    )
    submit.raise_for_status()
    job = submit.json()

    # 2. Upload the bytes straight to object storage.
    requests.put(job["uploadUrl"], data=data, timeout=300).raise_for_status()

    # 3. Poll until done, then download the result.
    while True:
        poll = requests.get(
            f"{HUSHVERT_API}/v1/conversions/{job['jobId']}",
            headers=headers,
            timeout=30,
        )
        poll.raise_for_status()
        state = poll.json()
        if state["status"] == "done":
            result = requests.get(state["downloadUrl"], timeout=300)
            result.raise_for_status()
            return result.content
        if state["status"] == "failed":
            raise RuntimeError(f"Conversion failed: {state.get('error')}")
        time.sleep(2)
```

## Building the Agent

### Define the agent state

The state carries the file path and question in, and the route, markdown, and answer out. Every node reads what it needs and returns only the keys it changes.

```python
class AgentState(TypedDict):
    file_path: str
    question: str
    source_format: str
    route: str
    markdown: str
    answer: str
```

### Node: inspect and route

Three outcomes: the file is already text (read it), the live formats list says it converts to markdown (convert it), or neither (refuse with guidance). Nothing here calls a model.

```python
NATIVE_TEXT = {"md", "markdown", "txt"}


def inspect_file(state: AgentState) -> dict:
    ext = pathlib.Path(state["file_path"]).suffix.lstrip(".").lower()
    if ext in NATIVE_TEXT:
        route = "native"
    elif f"{ext}-to-md" in server_pairs:
        route = "convert"
    else:
        route = "unsupported"
    return {"source_format": ext, "route": route}
```

### Node: convert to markdown

The node is a thin wrapper around the client above: convert to `md`, decode, store in state.

```python
def convert_document(state: AgentState) -> dict:
    markdown = convert_file(state["file_path"], to="md").decode("utf-8")
    return {"markdown": markdown}
```

### Nodes: read native text and refuse unsupported formats

Files that are already markdown or plain text skip conversion entirely. Anything unroutable gets a refusal that tells the sender exactly what would have worked, built from the same live formats list.

```python
def read_native(state: AgentState) -> dict:
    text = pathlib.Path(state["file_path"]).read_text(encoding="utf-8")
    return {"markdown": text}


def refuse_unsupported(state: AgentState) -> dict:
    supported = sorted({p.split("-to-")[0] for p in server_pairs if p.endswith("-to-md")})
    return {
        "answer": (
            f"'{state['source_format']}' is not a format I can read. "
            f"Send markdown or plain text directly, or one of: {', '.join(supported)}."
        )
    }
```

### Node: analyze with the LLM

The analyst is deliberately constrained: answer from the document, quote figures exactly, admit when the document does not contain the answer. Grounding instructions like these only work when the document the model sees actually preserves the structure the question is about, which is the entire point of the conversion stage.

```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

analyst_prompt = ChatPromptTemplate.from_template(
    "You are a careful document analyst. Answer the question using only the "
    "document below. Quote figures exactly as they appear. If the document "
    "does not contain the answer, say so.\n\n"
    "Document (markdown):\n{markdown}\n\n"
    "Question: {question}"
)


def analyze(state: AgentState) -> dict:
    chain = analyst_prompt | llm
    response = chain.invoke(
        {"markdown": state["markdown"], "question": state["question"]}
    )
    return {"answer": response.content}
```

### Assemble the graph

One conditional edge does all the routing; both reading paths converge on the analyst.

```python
workflow = StateGraph(AgentState)

workflow.add_node("inspect", inspect_file)
workflow.add_node("convert", convert_document)
workflow.add_node("read_native", read_native)
workflow.add_node("refuse", refuse_unsupported)
workflow.add_node("analyze", analyze)

workflow.add_edge(START, "inspect")
workflow.add_conditional_edges(
    "inspect",
    lambda state: state["route"],
    {"convert": "convert", "native": "read_native", "unsupported": "refuse"},
)
workflow.add_edge("convert", "analyze")
workflow.add_edge("read_native", "analyze")
workflow.add_edge("refuse", END)
workflow.add_edge("analyze", END)

app = workflow.compile()
```

### Visualize the workflow

```python
from IPython.display import Image, display

display(Image(app.get_graph().draw_mermaid_png()))
```

## Usage Example

### Create a realistic sample document

To keep the tutorial self-contained we generate the incoming file ourselves: a small quarterly report in docx with a heading, a paragraph, and a real Word table. This is exactly the kind of document that breaks naive extraction.

```python
from docx import Document

doc = Document()
doc.add_heading("Q3 Financial Summary", level=1)
doc.add_paragraph(
    "Revenue grew in three of four regions. The board flagged the West region "
    "for review after its second consecutive quarterly decline."
)

table = doc.add_table(rows=5, cols=3)
table.style = "Table Grid"
rows = [
    ("Region", "Revenue ($M)", "Change vs Q2"),
    ("North", "128.4", "+6.1%"),
    ("South", "95.2", "+2.8%"),
    ("East", "143.9", "+11.4%"),
    ("West", "61.7", "-4.3%"),
]
for r, row in enumerate(rows):
    for c, text in enumerate(row):
        table.rows[r].cells[c].text = text

doc.save("q3_report.docx")
print("Wrote q3_report.docx")
```

### Run the agent

One invocation: the graph inspects the file, converts it to markdown through the API, and answers a question that can only be answered correctly by reading the table.

```python
result = app.invoke(
    {
        "file_path": "q3_report.docx",
        "question": "Which region grew fastest in Q3, and which region should worry us?",
    }
)

print(result["answer"])
```

### What the model actually saw

The reason the answer is grounded: the table survived conversion as a GFM pipe table, so rows and columns still mean something.

```python
print(result["markdown"])
```

## Comparison with Naive Extraction

The intake agent's whole value is what it refuses to lose, so let's measure the loss directly. The cell below implements the "just rip the text out" baseline for docx: unzip the container, strip the XML tags, keep whatever falls out.

```python
def naive_docx_text(path: str) -> str:
    """The 'just rip the text out' baseline: read the XML, strip the tags."""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    return " ".join(re.sub(r"<[^>]+>", " ", xml).split())

print(naive_docx_text("q3_report.docx")[:320])
```

Every number survives, but the table does not: `North 128.4 +6.1% South 95.2 +2.8% ...` is a word soup where the association between region, revenue, and change now depends on the model counting tokens correctly. In the converted markdown the same data is a pipe table the model can read row by row.

The failure gets worse on PDFs. Glyph-level text extraction has no notion of layout, so a two-column page comes out interleaved, with sentence fragments from the left column spliced into the right. Conversion chains that reconstruct the document, rather than scrape its glyphs, preserve column reading order; that difference, not raw text recall, is usually what separates a usable ingestion pipeline from a broken one.

![Two-column reading order](https://github.com/NirDiamant/GenAI_Agents/raw/bd681451b254ac1a790e947b581d3997ab35013d/images/document-intake-reading-order.svg)

## Using the Same Capability from Claude Code, Cursor, and Other MCP Hosts

The graph above is the pattern for embedding document intake inside your own Python agents. If your agent is an MCP host (Claude Code, Cursor, Cline, Zed, Claude Desktop), you do not need any of this plumbing: the same conversion API ships as an MCP server, [`@hushvert/mcp`](https://www.npmjs.com/package/@hushvert/mcp), and the whole integration is one config block:

```json
{
  "mcpServers": {
    "hushvert": {
      "command": "npx",
      "args": ["-y", "@hushvert/mcp"],
      "env": { "HUSHVERT_API_KEY": "hv_live_your_key_here" }
    }
  }
}
```

The server exposes `convert_file`, `convert_poll`, `list_formats`, and `check_usage` tools. Ask the agent to "convert quarterly.pptx to markdown" and it performs the same submit, upload, poll, download round trip we implemented by hand, then writes the result next to the input.

For a general introduction to the protocol itself, see the [MCP tutorial](https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/mcp-tutorial.ipynb) in this repo.

## Additional Considerations

- **Metering and quotas.** Server conversions are billed per use: a free monthly allowance, then prepaid credits. `GET /v1/usage` returns the remaining allowance and balance; have batch jobs check it up front. Each pair also carries a size cap (returned by the formats endpoint as `freeMaxBytes`), and oversized uploads are rejected with HTTP 413 rather than charged.
- **Only pay for what needs a server.** Images, HEIC, audio, archives, and PDF page operations convert client-side in the browser with the open-source [`@hushvert/engine`](https://github.com/hushvert/engine) package (MIT, WebAssembly, the file never leaves the device). If you are building a web app rather than a Python agent, route those pairs through the engine for free and keep the API for the office-document formats a browser cannot do. The MCP server enforces this split: it refuses client-convertible pairs and points to the engine.
- **Privacy and retention.** Uploaded inputs are deleted the moment a conversion finishes; outputs expire after about an hour. Do not treat the download URL as storage: fetch the result and persist it yourself.
- **Failure handling.** A job ends `done` or `failed`, and `failed` carries a specific error. A scanned PDF with no text layer, for example, fails up front with a clear message and no charge, instead of returning empty markdown. Combine that with an `Idempotency-Key` and retries become boring, which is what you want.
- **Where this pattern goes next.** Here the converted markdown feeds a single question. In a real pipeline it is what you chunk and embed for RAG, diff against last quarter's version, or archive next to the original.

## Conclusion

We built an agent whose first move with any incoming file is to normalize it into LLM-ready markdown, and whose answers are grounded in structure that naive extraction throws away. Three ideas are worth keeping even if you swap every component:

1. **Route deterministically.** Format routing is a lookup against a live capability list, not a judgment call for the model.
2. **Convert, do not scrape.** Tables, headings, and reading order are where document questions are answered; a converter that reconstructs the document beats an extractor that strips it.
3. **One capability, two surfaces.** The same conversion service serves a Python graph you own and, through MCP, agents you do not (Claude Code, Cursor, and friends), so document intake stays consistent across your stack.

## References

- [hushvert conversion API](https://hushvert.com/for-developers): API documentation, key creation, and format list
- [`@hushvert/mcp`](https://www.npmjs.com/package/@hushvert/mcp): the MCP server from the config section
- [`@hushvert/engine`](https://github.com/hushvert/engine): the open-source browser-side engine for client-convertible formats
