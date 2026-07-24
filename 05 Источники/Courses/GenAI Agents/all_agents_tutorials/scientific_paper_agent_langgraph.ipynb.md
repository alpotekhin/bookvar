---
title: "Scientific paper agent using LangGraph"
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: bd681451b254ac1a790e947b581d3997ab35013d
---

> [!note] Original source material
> This page preserves [`all_agents_tutorials/scientific_paper_agent_langgraph.ipynb`](https://github.com/NirDiamant/GenAI_Agents/blob/bd681451b254ac1a790e947b581d3997ab35013d/all_agents_tutorials/scientific_paper_agent_langgraph.ipynb) from
> *GenAI Agents* at commit `bd681451b254ac1a790e947b581d3997ab35013d`. License:
> [Custom non-commercial license](https://github.com/NirDiamant/GenAI_Agents/blob/bd681451b254ac1a790e947b581d3997ab35013d/LICENSE). Bookvar changed only
> publication markup, link paths, and characters required for safe rendering.



## Overview



This project implements an intelligent research assistant that helps users navigate, understand, and analyze scientific literature using LangGraph and advanced language models. By combining various academic API with sophisticated paper processing techniques, it creates a seamless experience for researchers, students, and professionals working with academic papers.

> NOTE: The presented workflow is not domain specific: each step in the graph can be adapted to a different domain by simply changing the prompts.

## Motivation

Research literature review represents a significant time investment in R&D, with studies showing that researchers spend 30-50% of their time reading, analyzing, and synthesizing academic papers. This challenge is universal across the research community. While thorough literature review is crucial for advancing science and technology, the current process remains inefficient and time-consuming.

Key challenges include:
- Extensive time commitment (30-50% of R&D hours) dedicated to reading and processing papers
- Inefficient search processes across fragmented database ecosystems
- Complex task of synthesizing and connecting findings across multiple papers
- Resource-intensive maintenance of comprehensive literature reviews
- Ongoing effort required to stay current with new publications

## Key components

 1. State-Driven Workflow Engine 
    - StateGraph Architecture: Five-node system for orchestrated research 
    - Decision Making Node: Query intent analysis and routing 
    - Planning Node: Research strategy formulation
    - Tool Execution Node: Paper retrieval and processing 
    - Judge Node: Quality validation and improvement cycles 

2. Paper Processing Integration 
    - Source Integration, CORE API for comprehensive paper access 
    - Document Processing, PDF content extraction, Text structure preservation 

3. Analysis Workflow 
    - State-aware processing pipeline 
    - Multi-step validation gates 
    - Quality-focused improvement cycles 
    - Human-in-the-loop validation options

An overview of the workflow is shown below:

![image](https://i.ibb.co/0BBzkcb/mermaid-diagram-2024-11-17-195744.png)]

## Method details

1. The system requires 
    - OpenAI API key to access GPT 4o. This model was chosen after comparing its performance with other, open-source alternatives (in particular Llama 3). However, any other LLM with tool calling capabilities can be used.
    - CORE API key for paper retrieval. CORE is one of the larges online repositories for scientific papers, counting over 136 million papers, and offers a free API for personal use. A key can be requested [here](https://core.ac.uk/services/api#form).

2. Technical Architecture: 
    - LangGraph for state orchestration.
    - PDFplumber for document processing.
    - Pydantic for structured data handling.

> Acknowledgment: Special thanks to CORE API for enabling academic paper access.

---

## Setup

This cell installs the required dependencies.

```python
! pip install --upgrade --quiet langchain==0.2.16 langchain-community==0.2.16 langchain-openai==0.1.23 langgraph==0.2.18 langsmith==0.1.114 pdfplumber python-dotenv
```

This cell imports the required libraries and sets the environment variables.

```python
import io
import json
import os
import urllib3
import time

import pdfplumber
from dotenv import load_dotenv
from IPython.display import display, Markdown
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing import Annotated, ClassVar, Sequence, TypedDict, Optional

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

# You can set your own keys here
os.environ["OPENAI_API_KEY"] = "sk-proj-..."
os.environ["CORE_API_KEY"] = "..."
```

## Prompts

This cell contains the prompts used in the workflow.

The `agent_prompt` contains a section explaining how to use complex queries with the CORE API, enabling the agent to solve more complex tasks.

```python
# Prompt for the initial decision making on how to reply to the user
decision_making_prompt = """
You are an experienced scientific researcher.
Your goal is to help the user with their scientific research.

Based on the user query, decide if you need to perform a research or if you can answer the question directly.
- You should perform a research if the user query requires any supporting evidence or information.
- You should answer the question directly only for simple conversational questions, like "how are you?".
"""

# Prompt to create a step by step plan to answer the user query
planning_prompt = """
# IDENTITY AND PURPOSE

You are an experienced scientific researcher.
Your goal is to make a new step by step plan to help the user with their scientific research .

Subtasks should not rely on any assumptions or guesses, but only rely on the information provided in the context or look up for any additional information.

If any feedback is provided about a previous answer, incorportate it in your new planning.


# TOOLS

For each subtask, indicate the external tool required to complete the subtask. 
Tools can be one of the following:
{tools}
"""

# Prompt for the agent to answer the user query
agent_prompt = """
# IDENTITY AND PURPOSE

You are an experienced scientific researcher. 
Your goal is to help the user with their scientific research. You have access to a set of external tools to complete your tasks.
Follow the plan you wrote to successfully complete the task.

Add extensive inline citations to support any claim made in the answer.


# EXTERNAL KNOWLEDGE

## CORE API

The CORE API has a specific query language that allows you to explore a vast papers collection and perform complex queries. See the following table for a list of available operators:

| Operator       | Accepted symbols         | Meaning                                                                                      |
|---------------|-------------------------|----------------------------------------------------------------------------------------------|
| And           | AND, +, space          | Logical binary and.                                                                           |
| Or            | OR                     | Logical binary or.                                                                            |
| Grouping      | (...)                  | Used to prioritise and group elements of the query.                                           |
| Field lookup  | field_name:value       | Used to support lookup of specific fields.                                                    |
| Range queries | fieldName(>, <,>=, <=) | For numeric and date fields, it allows to specify a range of valid values to return.         |
| Exists queries| _exists_:fieldName     | Allows for complex queries, it returns all the items where the field specified by fieldName is not empty. |

Use this table to formulate more complex queries filtering for specific papers, for example publication date/year.
Here are the relevant fields of a paper object you can use to filter the results:
{
  "authors": [{"name": "Last Name, First Name"}],
  "documentType": "presentation" or "research" or "thesis",
  "publishedDate": "2019-08-24T14:15:22Z",
  "title": "Title of the paper",
  "yearPublished": "2019"
}

Example queries:
- "machine learning AND yearPublished:2023"
- "maritime biology AND yearPublished>=2023 AND yearPublished<=2024"
- "cancer research AND authors:Vaswani, Ashish AND authors:Bello, Irwan"
- "title:Attention is all you need"
- "mathematics AND _exists_:abstract"
"""

# Prompt for the judging step to evaluate the quality of the final answer
judge_prompt = """
You are an expert scientific researcher.
Your goal is to review the final answer you provided for a specific user query.

Look at the conversation history between you and the user. Based on it, you need to decide if the final answer is satisfactory or not.

A good final answer should:
- Directly answer the user query. For example, it does not answer a question about a different paper or area of research.
- Answer extensively the request from the user.
- Take into account any feedback given through the conversation.
- Provide inline sources to support any claim made in the answer.

In case the answer is not good enough, provide clear and concise feedback on what needs to be improved to pass the evaluation.
"""
```

## Utility classes and functions

This cell contains the utility classes and functions used in the workflow. It includes a wrapper around the CORE API, the Pydantic models for the input and output of the nodes, and a few general-purpose functions.

The `CoreAPIWrapper` class includes a retry mechanism to handle transient errors and make the workflow more robust.

```python
class CoreAPIWrapper(BaseModel):
    """Simple wrapper around the CORE API."""
    base_url: ClassVar[str] = "https://api.core.ac.uk/v3"
    api_key: ClassVar[str] = os.environ["CORE_API_KEY"]

    top_k_results: int = Field(description = "Top k results obtained by running a query on Core", default = 1)

    def _get_search_response(self, query: str) -> dict:
        http = urllib3.PoolManager()

        # Retry mechanism to handle transient errors
        max_retries = 5    
        for attempt in range(max_retries):
            response = http.request(
                'GET',
                f"{self.base_url}/search/outputs", 
                headers={"Authorization": f"Bearer {self.api_key}"}, 
                fields={"q": query, "limit": self.top_k_results}
            )
            if 200 <= response.status < 300:
                return response.json()
            elif attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 2))
            else:
                raise Exception(f"Got non 2xx response from CORE API: {response.status} {response.data}")

    def search(self, query: str) -> str:
        response = self._get_search_response(query)
        results = response.get("results", [])
        if not results:
            return "No relevant results were found"

        # Format the results in a string
        docs = []
        for result in results:
            published_date_str = result.get('publishedDate') or result.get('yearPublished', '')
            authors_str = ' and '.join([item['name'] for item in result.get('authors', [])])
            docs.append((
                f"* ID: {result.get('id', '')},\n"
                f"* Title: {result.get('title', '')},\n"
                f"* Published Date: {published_date_str},\n"
                f"* Authors: {authors_str},\n"
                f"* Abstract: {result.get('abstract', '')},\n"
                f"* Paper URLs: {result.get('sourceFulltextUrls') or result.get('downloadUrl', '')}"
            ))
        return "\n-----\n".join(docs)

class SearchPapersInput(BaseModel):
    """Input object to search papers with the CORE API."""
    query: str = Field(description="The query to search for on the selected archive.")
    max_papers: int = Field(description="The maximum number of papers to return. It's default to 1, but you can increase it up to 10 in case you need to perform a more comprehensive search.", default=1, ge=1, le=10)

class DecisionMakingOutput(BaseModel):
    """Output object of the decision making node."""
    requires_research: bool = Field(description="Whether the user query requires research or not.")
    answer: Optional[str] = Field(default=None, description="The answer to the user query. It should be None if the user query requires research, otherwise it should be a direct answer to the user query.")

class JudgeOutput(BaseModel):
    """Output object of the judge node."""
    is_good_answer: bool = Field(description="Whether the answer is good or not.")
    feedback: Optional[str] = Field(default=None, description="Detailed feedback about why the answer is not good. It should be None if the answer is good.")

def format_tools_description(tools: list[BaseTool]) -> str:
    return "\n\n".join([f"- {tool.name}: {tool.description}\n Input arguments: {tool.args}" for tool in tools])

async def print_stream(app: CompiledStateGraph, input: str) -> Optional[BaseMessage]:
    display(Markdown("## New research running"))
    display(Markdown(f"### Input:\n\n{input}\n\n"))
    display(Markdown("### Stream:\n\n"))

    # Stream the results 
    all_messages = []
    async for chunk in app.astream({"messages": [input]}, stream_mode="updates"):
        for updates in chunk.values():
            if messages := updates.get("messages"):
                all_messages.extend(messages)
                for message in messages:
                    message.pretty_print()
                    print("\n\n")
 
    # Return the last message if any
    if not all_messages:
        return None
    return all_messages[-1]
```

## Agent state

This cell defines the agent state, which contains the following information:
- `requires_research`: Whether the user query requires research or not.
- `num_feedback_requests`: The number of times the LLM asked for feedback.
- `is_good_answer`: Whether the LLM's final answer is good or not.
- `messages`: The conversation history between the user and the LLM.

```python
class AgentState(TypedDict):
    """The state of the agent during the paper research process."""
    requires_research: bool = False
    num_feedback_requests: int = 0
    is_good_answer: bool = False
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

## Agent tools

This cell defines the tools available to the agent. The toolkit contains a tool to search for scientific papers using the CORE API, a tool to download a scientific paper from a given URL, and a tool to ask for human feedback.

To make the paper download more robust, the tool includes a retry mechanism, similar to the one used for the CORE API, as well as a mock browser header to avoid 403 errors.

```python
@tool("search-papers", args_schema=SearchPapersInput)
def search_papers(query: str, max_papers: int = 1) -> str:
    """Search for scientific papers using the CORE API.

    Example:
    {"query": "Attention is all you need", "max_papers": 1}

    Returns:
        A list of the relevant papers found with the corresponding relevant information.
    """
    try:
        return CoreAPIWrapper(top_k_results=max_papers).search(query)
    except Exception as e:
        return f"Error performing paper search: {e}"

@tool("download-paper")
def download_paper(url: str) -> str:
    """Download a specific scientific paper from a given URL.

    Example:
    {"url": "https://sample.pdf"}

    Returns:
        The paper content.
    """
    try:        
        http = urllib3.PoolManager(
            cert_reqs='CERT_NONE',
        )
        
        # Mock browser headers to avoid 403 error
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }
        max_retries = 5
        for attempt in range(max_retries):
            response = http.request('GET', url, headers=headers)
            if 200 <= response.status < 300:
                pdf_file = io.BytesIO(response.data)
                with pdfplumber.open(pdf_file) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() + "\n"
                return text
            elif attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 2))
            else:
                raise Exception(f"Got non 2xx when downloading paper: {response.status_code} {response.text}")
    except Exception as e:
        return f"Error downloading paper: {e}"

@tool("ask-human-feedback")
def ask_human_feedback(question: str) -> str:
    """Ask for human feedback. You should call this tool when encountering unexpected errors."""
    return input(question)

tools = [search_papers, download_paper, ask_human_feedback]
tools_dict = {tool.name: tool for tool in tools}
```

## Workflow nodes

This cell defines the nodes of the workflow. Note how the `judge_node` is configured to end the execution if the LLM failed to provide a good answer twice to keep latency acceptable.

```python
# LLMs
base_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
decision_making_llm = base_llm.with_structured_output(DecisionMakingOutput)
agent_llm = base_llm.bind_tools(tools)
judge_llm = base_llm.with_structured_output(JudgeOutput)

# Decision making node
def decision_making_node(state: AgentState):
    """Entry point of the workflow. Based on the user query, the model can either respond directly or perform a full research, routing the workflow to the planning node"""
    system_prompt = SystemMessage(content=decision_making_prompt)
    response: DecisionMakingOutput = decision_making_llm.invoke([system_prompt] + state["messages"])
    output = {"requires_research": response.requires_research}
    if response.answer:
        output["messages"] = [AIMessage(content=response.answer)]
    return output

# Task router function
def router(state: AgentState):
    """Router directing the user query to the appropriate branch of the workflow."""
    if state["requires_research"]:
        return "planning"
    else:
        return "end"

# Planning node
def planning_node(state: AgentState):
    """Planning node that creates a step by step plan to answer the user query."""
    system_prompt = SystemMessage(content=planning_prompt.format(tools=format_tools_description(tools)))
    response = base_llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

# Tool call node
def tools_node(state: AgentState):
    """Tool call node that executes the tools based on the plan."""
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_dict[tool_call["name"]​].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}

# Agent call node
def agent_node(state: AgentState):
    """Agent call node that uses the LLM with tools to answer the user query."""
    system_prompt = SystemMessage(content=agent_prompt)
    response = agent_llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

# Should continue function
def should_continue(state: AgentState):
    """Check if the agent should continue or end."""
    messages = state["messages"]
    last_message = messages[-1]

    # End execution if there are no tool calls
    if last_message.tool_calls:
        return "continue"
    else:
        return "end"

# Judge node
def judge_node(state: AgentState):
    """Node to let the LLM judge the quality of its own final answer."""
    # End execution if the LLM failed to provide a good answer twice.
    num_feedback_requests = state.get("num_feedback_requests", 0)
    if num_feedback_requests >= 2:
        return {"is_good_answer": True}

    system_prompt = SystemMessage(content=judge_prompt)
    response: JudgeOutput = judge_llm.invoke([system_prompt] + state["messages"])
    output = {
        "is_good_answer": response.is_good_answer,
        "num_feedback_requests": num_feedback_requests + 1
    }
    if response.feedback:
        output["messages"] = [AIMessage(content=response.feedback)]
    return output

# Final answer router function
def final_answer_router(state: AgentState):
    """Router to end the workflow or improve the answer."""
    if state["is_good_answer"]:
        return "end"
    else:
        return "planning"
```

## Workflow definition

This cell defines the workflow using LangGraph.

```python
# Initialize the StateGraph
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("decision_making", decision_making_node)
workflow.add_node("planning", planning_node)
workflow.add_node("tools", tools_node)
workflow.add_node("agent", agent_node)
workflow.add_node("judge", judge_node)

# Set the entry point of the graph
workflow.set_entry_point("decision_making")

# Add edges between nodes
workflow.add_conditional_edges(
    "decision_making",
    router,
    {
        "planning": "planning",
        "end": END,
    }
)
workflow.add_edge("planning", "agent")
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": "judge",
    },
)
workflow.add_conditional_edges(
    "judge",
    final_answer_router,
    {
        "planning": "planning",
        "end": END,
    }
)

# Compile the graph
app = workflow.compile()
```

## Example usecase for PhD academic research

This cell tests the workflow with several example queries. These queries are designed to evaluate the agent on the following aspects:
- Completing tasks that are representative of the work a PhD researcher might need to perform.
- Addressing more specific tasks that require researching papers within a defined timeframe.
- Tackling tasks across multiple areas of research.
- Critically evaluating its own responses by sourcing specific information from the papers.

```python
test_inputs = [
    "Download and summarize the findings of this paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11379842/pdf/11671_2024_Article_4070.pdf",

    "Can you find 8 papers on quantum machine learning?",

    """Find recent papers (2023-2024) about CRISPR applications in treating genetic disorders, 
    focusing on clinical trials and safety protocols""",

    """Find and analyze papers from 2023-2024 about the application of transformer architectures in protein folding prediction, 
    specifically looking for novel architectural modifications with experimental validation."""
]

# Run tests and store the results for later visualisation
outputs = []
for test_input in test_inputs:
    final_answer = await print_stream(app, test_input)
    outputs.append(final_answer.content)
```

# Display results

This cell displays the results of the test queries for a more compact visualisation of the results.

```python
for input, output in zip(test_inputs, outputs):
    display(Markdown(f"## Input:\n\n{input}\n\n"))
    display(Markdown(f"## Output:\n\n{output}\n\n"))
```

## Input:

Download and summarize the findings of this paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11379842/pdf/11671_2024_Article_4070.pdf



## Output:

The paper titled "Advances, limitations and perspectives in the use of celecoxib-loaded nanocarriers in therapeutics of cancer" reviews the development and potential of celecoxib (CXB)-loaded nanocarriers in cancer treatment. Celecoxib is a selective COX-2 inhibitor used in cancer therapy, but its use is limited by the need for high doses, which can cause severe side effects. Nanocarriers offer a promising solution by improving the drug's biopharmaceutical properties, allowing for controlled release and targeted delivery.

### Key Findings:

1. **Nanocarrier Types and Materials**: 
   - CXB-loaded nanocarriers are primarily based on polymers and lipids, using materials like poly(lactic-co-glycolic acid) (PLGA), cholesterol, phospholipids, and poly(ethylene glycol) (PEG).
   - These carriers enhance drug solubility, stability, and bioavailability, and can be engineered for targeted delivery to tumor sites.

2. **Advancements in Formulations**:
   - Recent developments include the use of cell surface ligands, co-delivery of synergistic agents, and materials that provide imaging capabilities.
   - The combination of CXB with other anti-inflammatory drugs or apoptosis inducers shows promise in enhancing therapeutic effects.

3. **Clinical and Preclinical Studies**:
   - The research is mostly in preclinical stages, with no current clinical trials using CXB-loaded nanocarriers for cancer treatment.
   - In vivo studies have increased since 2017, indicating progress towards potential clinical applications.

4. **Challenges and Future Directions**:
   - The main challenges include CXB's low water solubility and the complexity of scaling up nanocarrier production for clinical use.
   - Future research should focus on optimizing nanocarrier design for stability, targeting, and controlled release, as well as exploring synergistic drug combinations.

5. **Potential Impact**:
   - CXB-loaded nanocarriers could significantly enhance the efficacy of cancer treatments by improving drug delivery and reducing side effects.
   - The ability of CXB to potentiate the effects of established chemotherapeutic agents is a major clinical advancement.

The paper highlights the potential of nanotechnology to revolutionize cancer therapy by enabling more effective and less harmful treatment options through the use of CXB-loaded nanocarriers.



## Input:

Can you find 8 papers on quantum machine learning?



## Output:

Here are 8 papers on quantum machine learning:

1. **Quantum Circuit Learning**
   - **Authors**: Mitarai, Kosuke; Negoro, Makoto; Kitagawa, Masahiro; Fujii, Keisuke
   - **Published Date**: April 23, 2019
   - **Abstract**: This paper proposes a classical-quantum hybrid algorithm for machine learning on near-term quantum processors, called quantum circuit learning. The framework allows a quantum circuit to learn tasks by tuning parameters, circumventing high-depth circuits. Theoretical and numerical simulations show that a quantum circuit can approximate nonlinear functions.
   - **URL**: [Quantum Circuit Learning](http://arxiv.org/abs/1803.00745)

2. **Quantum Machine Learning**
   - **Authors**: Biamonte, Jacob; Wittek, Peter; Pancotti, Nicola; Rebentrost, Patrick; Wiebe, Nathan; Lloyd, Seth
   - **Published Date**: May 10, 2018
   - **Abstract**: This paper explores the potential of quantum computers to outperform classical computers on machine learning tasks. It discusses the challenges and paths towards solutions in quantum machine learning.
   - **URL**: [Quantum Machine Learning](http://arxiv.org/abs/1611.09347)

3. **The Power of One Qubit in Machine Learning**
   - **Authors**: Ghobadi, Roohollah; Oberoi, Jaspreet S.; Zahedinejhad, Ehsan
   - **Published Date**: June 8, 2019
   - **Abstract**: This paper proposes a kernel-based quantum machine learning algorithm that can be implemented on near-term quantum devices, using deterministic quantum computing with one qubit.
   - **URL**: [The Power of One Qubit in Machine Learning](http://arxiv.org/abs/1905.01390)

4. **Quantum Machine Learning: A Classical Perspective**
   - **Authors**: Ciliberto, Carlo; Herbster, Mark; Ialongo, Alessandro Davide; Pontil, Massimiliano; Rocchetto, Andrea; Severini, Simone; Wossnig, Leonard
   - **Published Date**: February 13, 2018
   - **Abstract**: This review discusses the potential of quantum computation to speed up classical machine learning algorithms, highlighting the limitations and advantages of quantum resources for learning problems.
   - **URL**: [Quantum Machine Learning: A Classical Perspective](http://arxiv.org/abs/1707.08561)

5. **Quantum Machine Learning Over Infinite Dimensions**
   - **Authors**: Lau, Hoi-Kwan; Pooser, Raphael; Siopsis, George; Weedbrook, Christian
   - **Published Date**: November 14, 2016
   - **Abstract**: This paper generalizes quantum machine learning to infinite-dimensional systems, presenting subroutines for quantum machine learning algorithms on continuous-variable quantum computers.
   - **URL**: [Quantum Machine Learning Over Infinite Dimensions](http://arxiv.org/abs/1603.06222)

6. **Experimental Demonstration of Quantum Learning Speed-up with Classical Input Data**
   - **Authors**: Lee, Joong-Sung; Bang, Jeongho; Hong, Sunghyuk; Lee, Changhyoup; Seol, Kang Hee; Lee, Jinhyoung; Lee, Kwang-Geol
   - **Published Date**: November 22, 2018
   - **Abstract**: This paper demonstrates a quantum-classical hybrid machine learning approach, showing a quantum learning speed-up of approximately 36% compared to classical machines.
   - **URL**: [Experimental Demonstration of Quantum Learning Speed-up](http://arxiv.org/abs/1706.01561)

7. **Quantum-Enhanced Machine Learning**
   - **Authors**: Dunjko, Vedran; Taylor, Jacob M.; Briegel, Hans J.
   - **Published Date**: October 26, 2016
   - **Abstract**: This work proposes a systematic approach to machine learning from the perspective of quantum information, covering supervised, unsupervised, and reinforcement learning.
   - **URL**: [Quantum-Enhanced Machine Learning](http://arxiv.org/abs/1610.08251)

8. **An Efficient Quantum Algorithm for Generative Machine Learning**
   - **Authors**: Gao, Xun; Zhang, Zhengyu; Duan, Luming
   - **Published Date**: November 6, 2017
   - **Abstract**: This paper proposes a quantum algorithm for generative machine learning, showing exponential improvements in training and inference over classical algorithms.
   - **URL**: [An Efficient Quantum Algorithm for Generative Machine Learning](http://arxiv.org/abs/1711.02038)

These papers cover a range of topics within quantum machine learning, from theoretical frameworks to experimental demonstrations.



## Input:

Find recent papers (2023-2024) about CRISPR applications in treating genetic disorders, 
    focusing on clinical trials and safety protocols



## Output:

Here are some recent papers (2023-2024) on CRISPR applications in treating genetic disorders, focusing on clinical trials and safety protocols:

1. **CRISPR-Cas9 Gene Editing Tool: Potential Treatment for Sickle Cell Disease**
   - **Authors**: Young, Brittany
   - **Published Date**: April 25, 2024
   - **Abstract**: Not provided.
   - **URL**: [Read the paper](https://digitalcommons.sacredheart.edu/cgi/viewcontent.cgi?article=2403&context=acadfest)

2. **Progress and Harmonization of Gene Editing to Treat Human Diseases: Proceeding of COST Action CA21113 GenE-HumDi**
   - **Authors**: Cavazza, Alessia; González Martínez, Coral; Sánchez Martín, Rosario María; Martín Molina, Francisco; Benabdellah, Karim; COST Action CA21113
   - **Published Date**: December 12, 2023
   - **Abstract**: This publication discusses the efforts of the GenE-HumDi network to expedite the application of genome editing for therapeutic purposes in treating human diseases. It covers aspects like enhancing genome editing technologies, assessing delivery systems, addressing safety concerns, promoting clinical translation, and developing regulatory guidelines.
   - **URL**: [Read the paper](https://digibug.ugr.es/bitstream/10481/86007/1/1-s2.0-S2162253123002846-main.pdf)

3. **Germline Genome Editing of Human IVF Embryos Should Not Be Subject to Overly Stringent Restrictions**
   - **Authors**: Smith, Kevin
   - **Published Date**: July 5, 2024
   - **Abstract**: This paper critiques the restrictive criteria for germline genome editing, advocating for a balanced approach that weighs potential benefits against risks. It suggests that ethical oversight combined with genetic scrutiny can enable responsible use of the technology.
   - **URL**: [Read the paper](https://rke.abertay.ac.uk/files/81349925/Smith_GermlineGenomeEditing_Publised_2024.pdf)

4. **Balancing Progress and Ethics: Exploring the Science and Ethics of Gene Editing: Literature Review**
   - **Authors**: Burgess, Jackson
   - **Published Date**: January 1, 2024
   - **Abstract**: Not provided.
   - **URL**: [Read the paper](https://scholarworks.uni.edu/cgi/viewcontent.cgi?article=1929&context=hpt)

These papers provide insights into the current state of CRISPR technology in clinical settings, focusing on safety protocols and ethical considerations.



## Input:

Find and analyze papers from 2023-2024 about the application of transformer architectures in protein folding prediction, 
    specifically looking for novel architectural modifications with experimental validation.



## Output:

I found several papers from 2023 that discuss the application of transformer architectures in protein folding prediction, with a focus on novel architectural modifications and experimental validation. Here are some of the most relevant papers:

1. **Protein tertiary structure prediction and refinement using deep learning**
   - **Authors**: Wu, Tianqi
   - **Published Date**: 2023-01-08
   - **Abstract**: This paper discusses the development of a method called TransPross, which applies a 1D transformer network and attention mechanism for protein secondary structure prediction. It also introduces ATOMRefine, a novel end-to-end protein structure refinement tool. The paper emphasizes the use of deep learning techniques in improving protein structure prediction.
   - **URL**: [Read the paper](https://mospace.umsystem.edu/xmlui/bitstream/10355/94100/1/WuTianqiResearch.pdf)

2. **Enhancing the Protein Tertiary Structure Prediction by Multiple Sequence Alignment Generation**
   - **Authors**: Zhang, Le; Chen, Jiayang; Shen, Tao; Li, Yu; Sun, Siqi
   - **Published Date**: 2023-06-02
   - **Abstract**: This paper introduces MSA-Augmenter, a novel generative language model that uses protein-specific attention mechanisms to generate novel protein sequences. These sequences enhance the accuracy of structural property predictions, especially when existing sequences lack homologous families.
   - **URL**: [Read the paper](http://arxiv.org/abs/2306.01824)

3. **HelixFold-Single: MSA-free Protein Structure Prediction by Using Protein Language Model as an Alternative**
   - **Authors**: Fang, Xiaomin; Wang, Fan; Liu, Lihang; He, Jingzhou; Lin, Dayong; Xiang, Yingfei; Zhang, Xiaonan; Wu, Hua; Li, Hui; Song, Le
   - **Published Date**: 2023-02-21
   - **Abstract**: This paper presents HelixFold-Single, which combines a large-scale protein language model with AlphaFold2's geometric learning capabilities. It aims to predict protein structures using only primary sequences, bypassing the need for multiple sequence alignments (MSAs).
   - **URL**: [Read the paper](http://arxiv.org/abs/2207.13921)

4. **Integration of persistent Laplacian and pre-trained transformer for protein solubility changes upon mutation**
   - **Authors**: Wee, JunJie; Chen, Jiahui; Xia, Kelin; Wei, Guo-Wei
   - **Published Date**: 2023-11-02
   - **Abstract**: This work integrates persistent Laplacians and pre-trained Transformers to predict protein solubility changes upon mutation. The model outperforms existing methods and improves the state-of-the-art by up to 15%.
   - **URL**: [Read the paper](http://arxiv.org/abs/2310.18760)

These papers provide insights into the latest advancements in using transformer architectures for protein folding prediction, with a focus on novel modifications and experimental validation. You can explore these papers further to gain a deeper understanding of the specific architectural innovations and their experimental outcomes.



---

## Comparative Analysis

In this comprehensive analysis, we evaluated our Scientific Paper Agent against two leading AI knowledge co pilots : Microsoft Copilot and Perplexity AI. Using a standardized query - "Find 8 papers on quantum machine learning" - we conducted a detailed comparison across multiple dimensions to understand the strengths, limitations, and optimal use cases for each system.


#### Test Case Implementation

We implemented a controlled test using the same research query across all three platforms:
- Query: "Find 8 papers on quantum machine learning"
- Sample Size: Multiple test runs to ensure consistency
- Evaluation Time: Early 2024
- Metrics Tracked: Response time, metadata quality, and result structure

#### Key Findings

While our agent demonstrated superior academic rigor and metadata completeness, taking approximately 30 seconds per query, competitors like Microsoft Copilot (2 seconds) and Perplexity AI (4-5 seconds) showed advantages in response speed. This tradeoff between speed and depth reflects different design philosophies and target use cases.

The comparative analysis reveals a clear differentiation in approaches:
- Our Agent: Optimized for thorough academic research with comprehensive validation
- Microsoft Copilot: Focused on rapid information retrieval and general overview
- Perplexity AI: Balanced approach with emphasis on source verification

### Microsoft  copilot results 

![image](https://i.ibb.co/y4Zf4Pc/Screenshot-2024-11-17-at-21-40-21.png)]

### Perplexity AI results
![image](https://i.ibb.co/n1rr7kW/Screenshot-2024-11-17-at-21-40-42.png)]

### Metrics Comparsion
![image](https://i.ibb.co/5KbTmFq/Screenshot-2024-11-17-at-22-03-43.png)

Here we present a comprehensive comparison between our research assistant agent and leading platforms (Microsoft Copilot and Perplexity AI). Using a standardized query - "Find 8 papers on quantum machine learning" - we evaluated performance across key metrics including response time, metadata quality, and academic value. Our analysis reveals distinct trade-offs: while our agent takes longer to process (30s vs. 2-5s), it provides significantly more detailed metadata, validated sources, and structured academic output. The comparison table above breaks down these differences across multiple dimensions, helping users choose the right tool for their specific research needs - whether it's quick exploration (where Copilot excels) or deep academic research (where our agent shows its strengths).

---

##  Limitations

1. Technical Limitations
    - API rate limits for paper access
    - Handle time for large PDFs
    - Limited to publicly accessible papers
  
2. Functional Limitations
    - No support for image analysis in papers
    - Limited context window for very long papers
    - Cannot perform mathematical derivations
    - Language constraints for non-English papers


## Potential Improvements:

1. Technical Improvements
    - Implement parallel processing for multiple papers
    - Add caching system for frequently accessed papers
    - Integrate multiple academic APIs for broader coverage
    - Implement batch processing for large datasets

2. Functional Improvements
    - Add support for figure and table extraction
    - Implement cross-referencing between papers
    - Add citation network analysis
    - Include domain-specific validation rules
        
3. User Experience
    - Add interactive feedback mechanisms
    - Implement progress tracking
    - Add customizable validation criteria
    - Include export options for research summaries
        
   
## Specific Use Cases:

1. Academic Research, Literature review and paper analysis.
    - Comprehensive search
    - Citation tracking
    - Cross-reference validation

2. Industry Research, Technical documentation and patent analysis.
    - Focused search
    - Technical specification extraction
    - Competitive analysis

3. Educational, Student research assistance.
    - Simplified explanations
    - Learning resource identification
    - Guided research process

---

## Conclusion

This implementation demonstrates how state-driven architectures can transform academic paper analysis. By combining LangGraph's orchestration capabilities with robust API integrations, we've created a system that maintains research rigor while automating key aspects of paper processing. The workflow's emphasis on validation and quality control ensures reliable research outputs while significantly streamlining the paper analysis process.

![](https://europe-west1-genai-agents-views-tracker.cloudfunctions.net/genai-agents-tracker?notebook=all-agents-tutorials--scientific-paper-agent-langgraph)
