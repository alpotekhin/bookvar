---
title: "**Taskifier - Intelligent Task Allocation & Management**"
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: bd681451b254ac1a790e947b581d3997ab35013d
---

> [!note] Original source material
> This page preserves [`all_agents_tutorials/taskifier.ipynb`](https://github.com/NirDiamant/GenAI_Agents/blob/bd681451b254ac1a790e947b581d3997ab35013d/all_agents_tutorials/taskifier.ipynb) from
> *GenAI Agents* at commit `bd681451b254ac1a790e947b581d3997ab35013d`. License:
> [Custom non-commercial license](https://github.com/NirDiamant/GenAI_Agents/blob/bd681451b254ac1a790e947b581d3997ab35013d/LICENSE). Bookvar changed only
> publication markup, link paths, and characters required for safe rendering.



## Tech Stack
* Langchain
* LangGraph
* Tavily

## Overview

Taskifier presents an agent that helps manage task management for productivity optimization. This tutorial utilizes Langchain and LangGraph to build a regulated pipeline for such purpose. It encompasses:
 - context breakdown & analysis
 - external resource retrieval (web search)
 - discretization of information

## Motivation

In the world of workforce, procrastination and messy workflow is a common phenomenon, particularly with college students or non-administration level personnel in workplace. This is often due to the lack of clarity in objectives with the tasks given to them. Suppose a SWE in a startup was given a task to build a sign-in page for a web app. Things get messy and discouraging when the SWE was trying to start coding and asked questions like, "should I build an auth server first or should I create the front end first?". Those questions can branch off to smaller sub-questions, leaving the task puzzling and therefore driving procrastination. This phenomenon is highly replicable across different industries as well.

This projects aim at assisting in the analysis and organization of tasks that users need to complete. It utilizes the LLM's ability to qualitatively dissect information for such purpose. It will involve some behavioral analysis that adjusts the workstyle according to underlying patterns of how users approach different tasks, and thereby return an optimal workflow suggestion.

## Key Components

1. Data Ingestion: Gathers data for approach analysis
2. State Graph: Orchestrates steps from analysis to personalized generation
3. Tavily Web Query: Searches for information on the task to maximize task proficiency
4. LLM Model: Generates plans and analyzes approach

## Method Details

The system follows a step-by-step approach to personalize approaching plan for queried task:
1. Approach Analysis: Breakdown how the user tends to carry out tasks (a step by step person? a plan-first-then-build approach?)
2. Information Gathering: Retrieval of information related to the task in virtue of understanding what is necessary for completing the task
3. Customized Approach Generation: Given the analyzed style, the LLM generates a customized approach according to the style.

## Program Visualization

```python
display(
    Image(
        app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
)
```

```text
<IPython.core.display.Image object>
```

![[Assets/Sources/GenAI Agents/all_agents_tutorials/taskifier/cell-012-output-01.png]]

## Conclusion

This notebook exhibits the organized pipeline using LangGraph to induce step by step breakdown and generation of optimal response based on the user's preferences. It enables potential applications across different fields and different characters to optimize their workflow and productivity. Further analysis involving quantitative analysis can be used but given time limitation, we let LLM tackle the analysis of approach and yield the complete plan accordingly. Future improvements can involve behavioral analysis of decision making in quantitative terms, having multiple personas of different work attitudes and approach styles and match the user's preferences to the most similar personas, pivoting from user's feedbacks on generated response and tuning the style preference accordingly, etc.

***

## Installation
We will be using LangChain & LangGraph for building ensembles of agents & controlling their workflow.

```python
%%capture --no-stderr
!pip install langchain langgraph tavily-python
```

## Importations
**Make sure you have the OpenAI and Tavily API Keys as part of your environment variables!**

```python
import os
from typing import TypedDict, Annotated, List
from langgraph.graph import START, StateGraph, END

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_openai import ChatOpenAI
from langchain_core.runnables.graph import MermaidDrawMethod
from IPython.display import display, Image, Markdown

from tavily import TavilyClient

import os
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY')
```

## State Definitions
Here we define the states for the agent workflow. The states w

```python
class ApproachState(TypedDict):
    plan: str  # detailed workflow of the approach
    style: str # style description of the approach
    task: str # user's input of task
    details: str # internet retrieval of task specs
    history: str # description of history approaches
```

## LLM & Tavily Initialization

```python
# Initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-4o-mini")
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
```

## Component Functions
Define functions for...
* Internet Query of Task Specs [retrieval]
* Compare User Approach Preference vs Personas Approach Preference [filter approach]

```python
from tavily import TavilyClient

def approach_analysis(approach: ApproachState) -> ApproachState:
    """Retrieve history approach and let LLM do a qualitative analysis on user approach preference."""
    history = ""
    for h in os.listdir(f"{os.getcwd()}/history"):
        if (h[-4:] == ".txt"):
            with open(os.path.join(os.getcwd(), f"history/{h}")) as f:
                content = f.readlines()
            history = f"{history}\n{content[0]}"

    approach['history'] = history

    prompt = ChatPromptTemplate.from_template(
        "Analyze the work style the following summary of work history portrays. "
        "Provide a brief summary the preference in work style."
        "\n\nWork History: {history}"
    )
    style = llm.invoke(prompt.format(history=approach['history']))
    approach['style'] = style
    return approach

# def extract_aim(approach: ApproachState) -> ApproachState:
#     """Get key aims from the task query."""

def task_manifest(approach: ApproachState) -> ApproachState:
    """use Tavily to look up information on the task."""

    search_foundation = "What are the steps for the following task? {task}"
    search_query = search_foundation.format(task=approach["task"])

    searches = tavily_client.search(search_query, max_results=10)

    details = ""

    for result in searches['results']:
        if details == "":
            details = result['content']
        else:
            details = f"{details} {result['content']}"

    approach["details"] = details

    return approach

def result_approach(approach: ApproachState) -> ApproachState:
    prompt = ChatPromptTemplate.from_template(
        "Give me a plan of steps to carry out the following task with custom work styles specified."
        "You have to pay extra attention to Work Style mentioned below and adjust the plan accordingly."
        "\n\nTask: {task}\n\nDetails: {details}\n\nWork Style: {style}\n\n"
        "The output must be a numbered list of steps with explanation of why it is needed, what to do and how it considers the Work Style."
    )

    suggestion = llm.invoke(prompt.format(task=approach["task"], details=approach["details"], style=approach["style"]))

    approach['plan'] = suggestion

    return approach
```

## Graph Workflow Building
Now we can start to structure the workflow and organize them in order!

```python
# Initialize the StateGraph
workflow = StateGraph(ApproachState)

# Add nodes to the graph
workflow.add_node("approach_analysis", approach_analysis)
workflow.add_node("task_knowledge_retrieval", task_manifest)
workflow.add_node("customized_approach_generation", result_approach)

# Define and add conditional edges
workflow.add_edge("approach_analysis", "task_knowledge_retrieval")
workflow.add_edge("task_knowledge_retrieval", "customized_approach_generation")

# Set the entry point
workflow.set_entry_point("approach_analysis")

# Set the exit point
workflow.add_edge("customized_approach_generation", END)

# Compile the graph
app = workflow.compile()
```

## Agent Calling Function
This function will be used to induce the entire workflow!

```python
def approach(task: str) -> ApproachState:
    init_approach = ApproachState(
        task=task,
        plan="",
        style="",
        history="",
        details=""
    )

    response = app.invoke(init_approach)
    return response
```

### **🚀🚀🚀🚀🚀🚀🚀🚀 Great! Now we can start the inference and see how the workflow performs! 🚀🚀🚀🚀🚀🚀🚀🚀**

## Example
This is an example where the user hopes to build a smoke detector that is futuristic in design and accessible for installation!

```python
query = """
    I want to build a smoke detector device! I am visioning it with futuristic design and hope to maximize the ability to install it anywhere. Perhaps keep it small and energy efficient for that purpose!
    """

## Some history is being fed into the Agent! It helps the agent understand users' approach preferences!
generated_plan = approach(task=query)


print(f"Task:\n")
print(f"{generated_plan['task']}\n")
print(f"Style:\n")
print(f"{generated_plan['style']}\n")
print(f"Steps:\n")
generated_plan['plan'].pretty_print()
```

```text
Task:


    I want to build a smoke detector device! I am visioning it with futuristic design and hope to maximize the ability to install it anywhere. Perhaps keep it small and energy efficient for that purpose!


Style:

content="The work history summary portrays a systematic and pragmatic work style. The individual demonstrates a preference for tackling tasks in a structured manner, starting with simpler challenges to build confidence and familiarity before progressing to more complex issues. This approach reflects a methodical mindset and a desire to establish a strong foundation before confronting difficulties.\n\nIn the context of their venture into the medical IT field, the individual shows a proactive attitude by prioritizing regulatory challenges from the FDA and EMA, indicating a preference for addressing potential obstacles early on to avoid complications later. This indicates a forward-thinking approach and an inclination to mitigate risks.\n\nWhen it comes to job applications, the individual prefers to avoid extensive, time-consuming responses, suggesting a more straightforward, efficiency-driven work style. This preference for brevity indicates a focus on practicality and a desire to streamline processes, potentially valuing results over exhaustive detail.\n\nOverall, the individual's work style can be characterized as organized, proactive, and efficiency-oriented, with an emphasis on tackling tasks in a logical sequence and streamlining efforts to achieve goals." additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 205, 'prompt_tokens': 211, 'total_tokens': 416, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_0ba0d124f1', 'finish_reason': 'stop', 'logprobs': None} id='run-b1bdb405-a180-4354-bb70-fb68da28538e-0' usage_metadata={'input_tokens': 211, 'output_tokens': 205, 'total_tokens': 416, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}

Steps:

==================================[1m Ai Message [0m==================================

Here’s a structured plan to build a futuristic, energy-efficient smoke detector device, tailored to your systematic and pragmatic work style:

### Step 1: Define the Requirements
**Why It’s Needed:** Establishing clear requirements helps in laying a strong foundation for the project. It ensures all stakeholders have a unified vision and expectations are aligned.

**What to Do:**
- Gather information on necessary features (e.g., smoke detection technology, IoT integration, energy efficiency).
- Outline specifications such as size, power source, and design aesthetics.

**How It Considers the Work Style:** This step addresses your preference for a structured approach by starting with a clear understanding of what needs to be achieved before diving into complex design and engineering tasks.

### Step 2: Research and Select Components
**Why It’s Needed:** Choosing the right components is crucial for performance, energy efficiency, and integration capabilities.

**What to Do:**
- Investigate various sensor technologies (e.g., photoelectric, ionization).
- Evaluate energy-efficient microcontrollers and power sources (e.g., rechargeable batteries, solar).
- Explore sustainable materials for the casing (e.g., bamboo, recycled plastics).

**How It Considers the Work Style:** This step allows you to systematically review available options and make informed decisions, ensuring that you avoid potential complications later in the design process.

### Step 3: Create a Prototype Design
**Why It’s Needed:** A prototype helps visualize the final product and identify any design flaws or areas for improvement.

**What to Do:**
- Use CAD software to create a 3D model of the smoke detector.
- Ensure the design is compact and aesthetically futuristic.
- Plan for modularity to allow for future upgrades and integrations.

**How It Considers the Work Style:** Designing a prototype focuses on practicality and efficiency, allowing for straightforward adjustments and refinements before the full-scale development begins.

### Step 4: Develop Software for Smart Integration
**Why It’s Needed:** Software is essential for the smart functionality of the device, enabling features like remote notifications and data monitoring.

**What to Do:**
- Develop a mobile app or integrate with existing smart home systems.
- Implement real-time monitoring and data analytics for energy management.
- Ensure compliance with relevant regulations (e.g., safety standards).

**How It Considers the Work Style:** Addressing software development early on mitigates risks associated with regulatory compliance and ensures that the device can seamlessly integrate with other smart home technologies.

### Step 5: Build and Test the Prototype
**Why It’s Needed:** Testing is critical to verify that the device functions as intended and meets safety standards.

**What to Do:**
- Assemble the prototype using the selected components.
- Conduct thorough testing for smoke detection efficacy and energy consumption.
- Gather feedback from potential users for further refinement.

**How It Considers the Work Style:** This step emphasizes an organized approach to problem-solving, allowing you to identify and address issues in a logical and systematic manner.

### Step 6: Refine the Design Based on Testing Feedback
**Why It’s Needed:** Refinement ensures the final product meets user needs and performs optimally.

**What to Do:**
- Analyze the testing data and user feedback to identify areas for improvement.
- Make necessary adjustments to the design and functionality.
- Re-test the updated prototype.

**How It Considers the Work Style:** This iterative process aligns with your efficiency-driven mindset, focusing on continuous improvement and ensuring that the final product is both effective and market-ready.

### Step 7: Plan for Production and Market Launch
**Why It’s Needed:** A well-thought-out plan for production and marketing is essential for successful product launch and scalability.

**What to Do:**
- Identify manufacturing partners and production processes that adhere to sustainable practices.
- Develop a marketing strategy highlighting the energy-efficient and smart features of the smoke detector.
- Set a timeline for launch and distribution.

**How It Considers the Work Style:** This structured approach to planning emphasizes practicality and efficiency, ensuring that all aspects of the product’s lifecycle are considered and managed in an organized manner.

### Step 8: Monitor and Iterate Post-Launch
**Why It’s Needed:** Continuous monitoring post-launch helps in identifying any performance issues and user feedback for future improvements.

**What to Do:**
- Collect data on device performance and customer satisfaction.
- Address any issues promptly and plan for future iterations of the product based on user input.

**How It Considers the Work Style:** This proactive approach to monitoring reflects your preference for mitigating risks early, ensuring long-term success and user satisfaction.

By following this structured plan, you will effectively manage the complexity of building a smart smoke detector while adhering to your organized, pragmatic work style.
```

![](https://europe-west1-genai-agents-views-tracker.cloudfunctions.net/genai-agents-tracker?notebook=all-agents-tutorials--taskifier)
