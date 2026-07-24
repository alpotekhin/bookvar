---
title: "Project Manager Assistant Agent"
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: bd681451b254ac1a790e947b581d3997ab35013d
---

> [!note] Original source material
> This page preserves [`all_agents_tutorials/project_manager_assistant_agent.ipynb`](https://github.com/NirDiamant/GenAI_Agents/blob/bd681451b254ac1a790e947b581d3997ab35013d/all_agents_tutorials/project_manager_assistant_agent.ipynb) from
> *GenAI Agents* at commit `bd681451b254ac1a790e947b581d3997ab35013d`. License:
> [Custom non-commercial license](https://github.com/NirDiamant/GenAI_Agents/blob/bd681451b254ac1a790e947b581d3997ab35013d/LICENSE). Bookvar changed only
> publication markup, link paths, and characters required for safe rendering.



## Overview
This tutorial demonstrates how to create a an AI agent that assists in project management tasks including creating actionable tasks from a given project description, identify the dependency within the tasks, create a task execution schedule for the project, and assign the individual tasks to project members based on their expertise and experience. This application uses a combination of custom functions, structured output, and an agent that can streamline the project management, in particular during the project initation / setup process.

## Motivation
Effective management is key to project success. It already starts, with the setup of the project, definition of tasks, scheduling and allocation of team members. However, with increasing complexity of the project, this task becomes more and more challenging and time consuming frequently requiring days of planning involving several members of the organization.

The ```Project Manager Assistant Agent``` was created to transfrom how projects are initiated by introducing automation, intelligence, and precision into the process. It enables project managers to seamless translate project description into structure, actionable plans, mapping dependencies for better workflow alignment and assign tasks based on team members' expertise and experience. In addition, it creates risks scores for the individual tasks allowing overall project risk assessment. This overall project risk score is used as part of a self-reflection (along insights generations on the actual plan) to further improve the schedule and task assignment to reduce the project risks. (see details on the implementation).

### Benefits
This AI-driven approach reduces the burden of manual planning and eliminates redundancies, allowing project managers to shift their focus to higher-level strategy and decision. making.

Example visaulized output for an agent derived project plan:
![Project Manager Assistant Agent Gantt-chart](https://github.com/NirDiamant/GenAI_Agents/raw/bd681451b254ac1a790e947b581d3997ab35013d/images/project_manager_assistant_agent_ganttchart.png)

### Key Components
<ol>
    <li>
        <b>LangGraph</b>: Organize the overall workflow of the agent, managing the flow of information between different nodes and stages of the agent
    </li>
    <li>
        <b>GPT-4o-mini</b>: Extracts actionable items from the project description, map dependencies, schedules tasks, allocate tasks to team, and assess the risks
    </li>

</ol>

### Key Classes
<b>State Management</b>
<br/>
<ol>
    <li>
        <b>AgentState</b>
        <ul>
            <li><b>project_description</b>: Description of the project</li>
            <li><b>team</b>: The team available</li>
            <li><b>tasks</b>: List of tasks</li>
            <li><b>dependencies</b>: List of dependencies of tasks</li>
            <li><b>schedule</b>: Project schedule created by the agent</li>
            <li><b>task_allocations</b>: List of team member and task allocated to them</li>
            <li><b>risks</b>: Risks associated with the tasks</li>
            <li><b>iteration_number</b>: Number of times the agent will go into self-reflection</li>
            <li><b>max_iteration</b>: The maximum number of time it can go into feedback loop to improve itself</li>
            <li><b>insights</b>: Insights generated to improve the response</li>
            <li><b>schedule_iteration</b>: Schedule per iteration </li>
            <li><b>task_allocations_iteration</b>: Task allocation per iteration</li>
            <li><b>risks_iteration</b>: Risks per task per iteration</li>
            <li><b>project_risk_score_iterations</b>: Projecr risk score per iteration</li>
        </ul>
    </li>
    <li>
        <b>Task</b>
        <ul>
            <li><b>id</b>: Id for the task</li>
            <li><b>task_name</b>: Name of the task</li>
            <li><b>task_description</b>: Description of task</li>
            <li><b>estimated_day</b>: Estimated time the task will take to complete</li>
        </ul>
    </li>
    <li>
        <b>TaskDependency</b>
        <ul>
            <li><b>task</b>: The task</li>
            <li><b>dependent_task</b>: Dependent tasks</li>
        </ul>
    </li>
    <li>
        <b>Team Member</b>
        <ul>
            <li><b>name</b>: Name of the team member</li>
            <li><b>profile</b>: Profile of the member</li>
        </ul>
    </li>
    <li>
        <b>Team</b>
        <ul>
            <li><b>team_member</b>: Members in the team</li>
        </ul>
    </li>
    <li>
        <b>TaskAllocation</b>
        <ul>
            <li><b>task</b>: A task </li>
            <li><b>team_member</b>: To whom the task is allocated</li>
        </ul>
    </li>
    <li>
        <b>TaskSchedule</b>
        <ul>
            <li><b>task</b>: A task </li>
            <li><b>start_day</b>: When the task should start</li>
            <li><b>end_day</b>: When the task will end</li>
        </ul>
    </li>
    <li>
        <b>TaskList</b>
        <ul>
            <li><b>tasks</b>: List of tasks</li>
        </ul>
    </li>
    <li>
        <b>DependencyList</b>
        <ul>
            <li><b>dependencies</b>: List of tasks with their dependent tasks</li>
        </ul>
    </li>
    <li>
        <b>Schedule</b>
        <ul>
            <li><b>schedule</b>: List of task schedule</li>
        </ul>
    </li>
    <li>
        <b>TaskAllocationList</b>
        <ul>
            <li><b>task_allocations</b>: List of allocated tasks</li>
        </ul>
    </li>
    <li>
        <b>Risk</b>
        <ul>
            <li><b>task</b>: A task</li>
            <li><b>score</b>: Risk score for that task</li>
        </ul>
    </li>
    <li>
        <b>RiskList</b>
        <ul>
            <li><b>risks</b>: List of risks</li>
        </ul>
    </li>
</ol>

### Key Nodes and Functions
<ol>
    <li>
        <b>task_generation</b>: Node that will extract tasks from given project description
    </li>
    <li>
        <b>task_dependencies</b>: Node that will evaluate the dependencies between the tasks
    </li>
    <li>
        <b>task_scheduler</b>: Node that will schedule tasks based on dependencies and team availability
    </li>
    <li>
        <b>task_allocator</b>: Node that will allocate tasks to team members
    </li>
    <li>
        <b>risk_assessor</b>: Node that analyze risk associated with schedule and allocation of task
    </li>
    <li>
        <b>insight_generator</b>: Node that generate insights from the schedule, task allocation and risk associated
    </li>
    <li>
        <b>router</b>: A helper function that will route the agent to the appropriate node based on the project description
    </li>
</ol>

## Visual Representation of the Agent

<img src="https://github.com/NirDiamant/GenAI_Agents/raw/bd681451b254ac1a790e947b581d3997ab35013d/images/project_manager_assistant_agent.svg" alt="Project Manager Assistant" width="500" height="500">

## Implementation
In the following section, we provide a detailed overview how to implement the Project Manager Assistant Agent.

### Install and import tutorial specific libraries

```python
!pip install langchain langgraph langchain-openai pandas plotly networkx pyvis openai python-dotenv -q
```

```python
import os
import uuid
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from typing import List, TypedDict
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.graph import StateGraph, START,END
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display, Markdown, HTML
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Define your model provider
model_provider = 'Azure' # 'Azure' or 'OpenAI'
```

### Instantiate LLM model

```python
# Based on model_provider load the language model
if model_provider == 'Azure':
    """
    Define your environmental variables under .venv:
        - AZURE_OPENAI_API_KEY
        - OPENAI_API_VERSION
        - AZURE_OPENAI_ENDPOINT
    """
    llm = AzureChatOpenAI(
        deployment_name='gpt-4o-mini',  # Your actual deployment name
    )
elif model_provider == 'OpenAI':
    """
    Define your environmental variables under .venv:
        - OPENAI_API_KEY
        - OPENAI_API_BASE
    """
    llm = ChatOpenAI(model="gpt-4o-mini")
else:
    print('Implement your own llm loader')
```

```python
# Test your LLM
llm.invoke("Hello, how are you?")
```

```text
AIMessage(content="Hello! I'm just a computer program, so I don't have feelings, but I'm here and ready to help you. How can I assist you today?", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 30, 'prompt_tokens': 13, 'total_tokens': 43, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_04751d0b65', 'prompt_filter_results': [{'prompt_index': 0, 'content_filter_results': {'hate': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'filtered': False, 'detected': False}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}}}], 'finish_reason': 'stop', 'logprobs': None, 'content_filter_results': {'hate': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}}}, id='run-9bbab967-9fae-47d5-84bd-a1b38555873a-0', usage_metadata={'input_tokens': 13, 'output_tokens': 30, 'total_tokens': 43, 'input_token_details': {}, 'output_token_details': {}})
```

In the following section, we will step-by-step implement:
- The data classes,
- Agent state,
- Nodes and function (used as a router),
- and the workflow of the agent.

So let's create all required data classes first:

```python
# Data Models
class Task(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Unique identifier for the task")
    task_name: str = Field(description="Name of the task")
    task_description: str = Field(description="Description of the task")
    estimated_day: int = Field(description="Estimated number of days to complete the task")

class TaskList(BaseModel):
    tasks: List[Task] = Field(description="List of tasks")

class TaskDependency(BaseModel):
    """Task dependency model"""
    task: Task = Field(description="Task")
    dependent_tasks: List[Task] = Field(description="List of dependent tasks")

class TeamMember(BaseModel):
    name: str = Field(description="Name of the team member")
    profile: str = Field(description="Profile of the team member")

class Team(BaseModel):
    team_members: List[TeamMember] = Field(description="List of team members")

# Iterative assessment
class TaskAllocation(BaseModel):
    """Task allocation class"""
    task: Task = Field(description="Task")
    team_member: TeamMember = Field(description="Team members assigned to the task")

class TaskSchedule(BaseModel):
    """Schedule schedule class"""
    task: Task = Field(description="Task")
    start_day: int = Field(description="Start day of the task")
    end_day: int = Field(description="End day of the task")

# Lists
class DependencyList(BaseModel):
    """List of task dependencies"""
    dependencies: List[TaskDependency] = Field(description="List of task dependencies")

class Schedule(BaseModel):
    """List of task schedules"""
    schedule: List[TaskSchedule] = Field(description="List of task schedules")

class TaskAllocationList(BaseModel):
    """List of task allocations"""
    task_allocations: List[TaskAllocation] = Field(description="List of task allocations")

# Iteration
class TaskAllocationListIteration(BaseModel):
    """List of task allocations for each iteration"""
    task_allocations_iteration: List[TaskAllocationList] = Field(description="List of task allocations for each iteration")

class ScheduleIteration(BaseModel):
    """List of task schedules for each iteration"""
    schedule: List[Schedule] = Field(description="List of task schedules for each iteration")

class Risk(BaseModel):
    """Risk of a task"""
    task: Task = Field(description="Task")
    score: str = Field(description="Risk associated with the task")

class RiskList(BaseModel):
    """List of risks for each iteration"""
    risks: List[Risk] = Field(description="List of risks")

class RiskListIteration(BaseModel):
    """List of risks for each iteration"""
    risks_iteration: List[RiskList] = Field(description="List of risks for each iteration")
```

In the next step, let's create the AgentState. The `schedule_iteration`, `task_allocations_iteration`, `risks_iteration` are introduced to generate structured 'memory' for the self-reflection cycles.

```python
class AgentState(TypedDict):
    """The project manager agent state."""
    project_description: str
    team: Team
    tasks: TaskList
    dependencies: DependencyList
    schedule: Schedule
    task_allocations: TaskAllocationList
    risks: RiskList
    iteration_number: int
    max_iteration: int
    insights: List[str]
    schedule_iteration: List[Schedule]
    task_allocations_iteration: List[TaskAllocationList]
    risks_iteration: List[RiskListIteration]
    project_risk_score_iterations: List[int]
```

Well done, let's create the required nodes. As a quick recap:

 In LangGraph a node is defined as a function which has an argument the `AgentState`. Within the node certain attributes of the field state is updated which at the end of the node is returned to the workflow manager and passed to the next node. Inside the nodes, LLM's are used to generate (non)-structured response.

In this tutorial we have implemented the nodes based on th following pattern:

```
def task_generation_node(state: AgentState):
 """LangGraph node that will extract tasks from given project description"""
    description = state["project_description"]
    prompt = f"""You are an experienced project description analyzer. Analyze the
    project description '{description}' and create a list of actionable and
    realistic tasks with estimated time (in days) to complete each task.
    If the task takes longer than 5 days, break it down into independent smaller tasks.
    """
    structure_llm = llm.with_structured_output(TaskList)
    tasks: TaskList = structure_llm.invoke(prompt)
    state['tasks'] = tasks
    return state
```
In almost all nodes, we used:
- `llm.with_structured_output(<structure>)` - generating structured output.

The .with_structured_output() method enables models with native APIs for structured outputs, such as function calling or JSON mode, to reliably produce outputs as objects based on a defined schema. The schema can be specified using a TypedDict, JSON Schema, or a Pydantic class, determining whether the output is a dictionary or a Pydantic object.

The only exception is the insight_generation_node where only `str` as requested from the llm and the required interface only `llm.invoke(prompt)`

```python
# Workflow Nodes
def task_generation_node(state: AgentState):
    """LangGraph node that will extract tasks from given project description"""
    description = state["project_description"]
    prompt = f"""
        You are an expert project manager tasked with analyzing the following project description: {description}
        Your objectives are to:
        1. **Extract Actionable Tasks:**
            - Identify and list all actionable and realistic tasks necessary to complete the project.
            - Provide an estimated number of days required to complete each task.
        2. **Refine Long-Term Tasks:**
            - For any task estimated to take longer than 5 days, break it down into smaller, independent sub-tasks.
        **Requirements:** - Ensure each task is clearly defined and achievable.
            - Maintain logical sequencing of tasks to facilitate smooth project execution."""

    structure_llm = llm.with_structured_output(TaskList)
    tasks: TaskList = structure_llm.invoke(prompt)
    return {"tasks": tasks}

def task_dependency_node(state: AgentState):
    """Evaluate the dependencies between the tasks"""
    tasks = state["tasks"]
    prompt = f"""
        You are a skilled project scheduler responsible for mapping out task dependencies.
        Given the following list of tasks: {tasks}
        Your objectives are to:
            1. **Identify Dependencies:**
                - For each task, determine which other tasks must be completed before it can begin (blocking tasks).
            2. **Map Dependent Tasks:**
                - For every task, list all tasks that depend on its completion.
        """
    structure_llm = llm.with_structured_output(DependencyList)
    dependencies: DependencyList = structure_llm.invoke(prompt)
    return {"dependencies": dependencies}

def task_scheduler_node(state: AgentState):
    """LangGraph node that will schedule tasks based on dependencies and team availability"""
    dependencies = state["dependencies"]
    tasks = state["tasks"]
    insights = state["insights"] #"" if state["insights"] is None else state["insights"].insights[-1]
    prompt = f"""
        You are an experienced project scheduler tasked with creating an optimized project timeline.
        **Given:**
            - **Tasks:** {tasks}
            - **Dependencies:** {dependencies}
            - **Previous Insights:** {insights}
            - **Previous Schedule Iterations (if any):** {state["schedule_iteration"]}
        **Your objectives are to: **
            1. **Develop a Task Schedule:**
                - Assign start and end days to each task, ensuring that all dependencies are respected.
                - Optimize the schedule to minimize the overall project duration.
                - If possible parallelize the tasks to reduce the overall project duration.
                - Try not to increase the project duration compared to previous iterations.
            2. **Incorporate Insights:**
                - Utilize insights from previous iterations to enhance scheduling efficiency and address any identified issues.
        """
    schedule_llm = llm.with_structured_output(Schedule)
    schedule: Schedule = schedule_llm.invoke(prompt)
    state["schedule"] = schedule
    state["schedule_iteration"].append(schedule)
    return state

def task_allocation_node(state: AgentState):
    """LangGraph node that will allocate tasks to team members"""
    tasks = state["tasks"]
    schedule = state["schedule"]
    team = state["team"]
    insights = state["insights"] #"" if state["insights"] is None else state["insights"].insights[-1]
    prompt = f"""
        You are a proficient project manager responsible for allocating tasks to team members efficiently.
        **Given:**
            - **Tasks:** {tasks}
            - **Schedule:** {schedule}
            - **Team Members:** {team}
            - **Previous Insights:** {insights}
            - **Previous Task Allocations (if any):** {state["task_allocations_iteration"]}
        **Your objectives are to:**
            1. **Allocate Tasks:**
                - Assign each task to a team member based on their expertise and current availability.
                - Ensure that no team member is assigned overlapping tasks during the same time period.
            2. **Optimize Assignments:**
                - Utilize insights from previous iterations to improve task allocations.
                - Balance the workload evenly among team members to enhance productivity and prevent burnout.
                **Constraints:**
                    - Each team member can handle only one task at a time.
                    - Assignments should respect the skills and experience of each team member.
        """
    structure_llm = llm.with_structured_output(TaskAllocationList)
    task_allocations: TaskAllocationList = structure_llm.invoke(prompt)
    state["task_allocations"] = task_allocations
    state["task_allocations_iteration"].append(task_allocations)
    return state

def risk_assessment_node(state: AgentState):
    """LangGraph node that analyse risk associated with schedule and allocation of task"""
    schedule = state["schedule"]
    task_allocations=state["task_allocations"]
    prompt = f"""
        You are a seasoned project risk analyst tasked with evaluating the risks associated with the current project plan.
        **Given:**
            - **Task Allocations:** {task_allocations}
            - **Schedule:** {schedule}
            - **Previous Risk Assessments (if any):** {state['risks_iteration']}
        **Your objectives are to:**
            1. **Assess Risks:**
                - Analyze each allocated task and its scheduled timeline to identify potential risks.
                - Consider factors such as task complexity, resource availability, and dependency constraints.
            2. **Assign Risk Scores:**
            - Assign a risk score to each task on a scale from 0 (no risk) to 10 (high risk).
            - If a task assignment remains unchanged from a previous iteration (same team member and task), retain the existing risk score to ensure consistency.
            - If the team member has more time between tasks - assign lower risk score for the tasks
            - If the task is assigned to a more senior person - assign lower risk score for the tasks
            3. **Calculate Overall Project Risk:**
            - Sum the individual task risk scores to determine the overall project risk score.
        """
    structure_llm = llm.with_structured_output(RiskList)
    risks: RiskList = structure_llm.invoke(prompt)
    project_task_risk_scores = [int(risk.score) for risk in risks.risks]
    project_risk_score = sum(project_task_risk_scores)
    state["risks"] = risks
    state["project_risk_score"] = project_risk_score
    state["iteration_number"] += 1
    state["project_risk_score_iterations"].append(project_risk_score)
    state["risks_iteration"].append(risks)
    return state

def insight_generation_node(state: AgentState):
    """LangGraph node that generate insights from the schedule, task allocation, and risk associated"""
    schedule = state["schedule"]
    task_allocations=state["task_allocations"]
    risks = state["risks"]
    prompt = f"""
        You are an expert project manager responsible for generating actionable insights to enhance the project plan.
        **Given:**
            - **Task Allocations:** {task_allocations}
            - **Schedule:** {schedule}
            - **Risk Analysis:** {risks}
        **Your objectives are to:**
            1. **Generate Critical Insights:**
            - Analyze the current task allocations, schedule, and risk assessments to identify areas for improvement.
            - Highlight any potential bottlenecks, resource conflicts, or high-risk tasks that may jeopardize project success.
            2. **Recommend Enhancements:**
            - Suggest adjustments to task assignments or scheduling to mitigate identified risks.
            - Propose strategies to optimize resource utilization and streamline workflow.
                **Requirements:**
                - Ensure that all recommendations aim to reduce the overall project risk score.
                - Provide clear and actionable suggestions that can be implemented in subsequent iterations.
        """
    insights = llm.invoke(prompt).content
    return {"insights": insights}
```

The proposed agentic workflow contains a conditional routing in which the logic was built around the overall risk score of the project plan. The task scheduling and task assignment is carried out at least twice in a 'self-reflection' in order to minimize the overall project risk assigned in each iteration as part of the `risk_assessment_node`. If the risk was reduced the agent finishes its task, otherwise tries to self-reflect using an `insight_generation_node` from which the insights fed back to the scheduler_node.

```python
def router(state: AgentState):
    """LangGraph node that will route the agent to the appropriate node based on the project description"""
    max_iteration = state["max_iteration"]
    iteration_number = state["iteration_number"]

    if iteration_number < max_iteration:
        if len(state["project_risk_score_iterations"])>1:
            if state["project_risk_score_iterations"][-1] < state["project_risk_score_iterations"][0]:
                return END
            else:
                return "insight_generator"
        else:
            return "insight_generator"
    else:
        return END
```

As a last remainign step, let's create an agentic workflow using LangGraph.

```python
# Instantiate the workflow
workflow = StateGraph(AgentState)

# Add nodes to the workflow
workflow.add_node("task_generation", task_generation_node)
workflow.add_node("task_dependencies", task_dependency_node)
workflow.add_node("task_scheduler", task_scheduler_node)
workflow.add_node("task_allocator", task_allocation_node)
workflow.add_node("risk_assessor", risk_assessment_node)
workflow.add_node("insight_generator", insight_generation_node)

# Add edges to the workflow
workflow.set_entry_point("task_generation")
workflow.add_edge("task_generation", "task_dependencies")
workflow.add_edge("task_dependencies", "task_scheduler")
workflow.add_edge("task_scheduler", "task_allocator")
workflow.add_edge("task_allocator", "risk_assessor")
workflow.add_conditional_edges("risk_assessor", router, ["insight_generator", END])
workflow.add_edge("insight_generator", "task_scheduler")

# Set up memory
memory = MemorySaver()

# Compile the workflow
graph_plan = workflow.compile(checkpointer=memory)
```

```python
# To visualize the created workflow, we can use
display(Image(graph_plan.get_graph(xray=1).draw_mermaid_png()))
```

```text
<IPython.core.display.Image object>
```

![[Assets/Sources/GenAI Agents/all_agents_tutorials/project_manager_assistant_agent/cell-026-output-01.png]]

## Usage Example

#### Initiate of the AgentState by defining an input
Our Project Manager Agent requires a project description (str) and a team (Team) input next to other initialization parameters.
In this tutorial we provide two dummy input under `data`. The team is defined as a csv file, composed of 2 colums: <i>Name</i>,<i>Profile Description</i>

```python
def get_project_description(file_path:str):
    """Read the project description from the file"""
    with open(file_path, 'r') as file:
            content = file.read()

    return content

def get_team(file_path:str):
    """Read the team members from the CSV file"""
    team_df = pd.read_csv(file_path)
    team_members = [
            TeamMember(name=row['Name'], profile=row['Profile Description'])
            for _, row in team_df.iterrows()
        ]
    team = Team(team_members=team_members)

    return team

project_description = get_project_description("../data/project_manager_assistant/project_description.txt")
team = get_team("../data/project_manager_assistant/team.csv")

print(project_description)
print(team)
```

```text
Our business aims to deliver a chatbot application for our customers to ensure 24/7 support and advice on product choices.
team_members=[TeamMember(name='Alice', profile=' Alice is a Frontend Developer skilled in HTML CSS JavaScript and React.'), TeamMember(name='Bob', profile=' Bob is a Backend Developer proficient in Python Django SQL and RESTful APIs.'), TeamMember(name='Charlie', profile=' Charlie is a Project Manager experienced in Agile methodologies team leadership project planning and risk management.'), TeamMember(name='David', profile=' David is a Full Stack Developer with expertise in both frontend (HTML CSS JavaScript) and backend (Node.js MongoDB) technologies.'), TeamMember(name='Eve', profile=' Eve is a DevOps Engineer skilled in CI/CD pipelines Docker Kubernetes and cloud services like AWS and Azure.'), TeamMember(name='Frank', profile=' Frank is a Junior Frontend Developer with knowledge in HTML CSS JavaScript and basic React.'), TeamMember(name='Grace', profile=' Grace is a Senior Data Scientist with expertise in machine learning data analysis Python R and big data technologies like Hadoop and Spark.')]
```

```python
# Definition of the AgentState
state_input = {
    "project_description": project_description,
    "team": team,
    "insights": "",
    "iteration_number": 0,
    "max_iteration": 3,
    "schedule_iteration": [],
    "task_allocations_iteration": [],
    "risks_iteration": [],
    "project_risk_score_iterations": []
}
```

```python
# Invoke the agent
config = {"configurable": {"thread_id": "1"}}
for event in graph_plan.stream(state_input, config, stream_mode=["updates"]):
    "Print the different nodes as the agent progresses"
    print(f"Current node: {next(iter(event[1]))}")
```

```text
Current node: task_generation
Current node: task_dependencies
Current node: task_scheduler
Current node: task_allocator
Current node: risk_assessor
Current node: insight_generator
Current node: task_scheduler
Current node: task_allocator
Current node: risk_assessor
Current node: insight_generator
Current node: task_scheduler
Current node: task_allocator
Current node: risk_assessor
```

```python
# Retrive the final state
final_state = graph_plan.get_state(config).values
print(final_state['iteration_number'])
print(final_state['project_risk_score_iterations'])
```

```text
3
[38, 38, 38]
```

As we can see from the final_state the agentic workflow ended up having at least 2 iterations, and under `project_risk_score_iterations` we may see evolution of the project risk score. Even if the overall risk score does not descrease between iterations (additional prompt engineering can potentially improve it), we can assume that by increasing the project timeline (e.g., iteration #2) - the risk should descrease.

#### Visualize the results
To better understand the output of the agentic workflow - here we provide visualizations for:
- Project plan - Gantt-char for all created scenarios

```python
# Visalize Project timeline
number_of_iterations = final_state['iteration_number']

for i in range(number_of_iterations):
    ## Tasks schedule
    task_schedules = final_state['schedule_iteration'][i].schedule

    t = []
    # Iterate over the task_schedules and append each task's data to the DataFrame
    for task_schedule in task_schedules:
        t.append([
            task_schedule.task.task_name,
            task_schedule.start_day,
            task_schedule.end_day
        ])

    df_schedule = pd.DataFrame(t,columns=['task_name', 'start', 'end'])

    ## Tasks allocation
    task_allocations = final_state['task_allocations_iteration'][i].task_allocations

    t = []
    # Iterate over the task_schedules and append each task's data to the DataFrame
    for task_allocation in task_allocations:
        t.append([
            task_allocation.task.task_name,
            task_allocation.team_member.name
        ])

    df_allocation = pd.DataFrame(t,columns=['task_name', 'team_member'])

    df = df_allocation.merge(df_schedule, on='task_name')

    import plotly.express as px

    from datetime import datetime, timedelta
    # Get the current date
    current_date = datetime.today()

    # Convert start and end offsets to actual dates
    df['start'] = df['start'].apply(lambda x: current_date + timedelta(days=x))
    df['end'] = df['end'].apply(lambda x: current_date + timedelta(days=x))

    df.rename(columns={'team_member': 'Team Member'}, inplace=True)
    df.sort_values(by='Team Member', inplace=True)
    # Create a Gantt chart
    fig = px.timeline(df, x_start="start", x_end="end", y="task_name", color="Team Member", title=f"Gantt Chart - Iteration:{i+1} ")

    # Update layout for better visualization
    fig.update_layout(
        xaxis_title="Timeline",
        yaxis_title="Tasks",
        yaxis=dict(autorange="reversed"),  # Reverse the y-axis to have tasks in the vertical side
        title_x=0.5
    )

    # Show the plot
    fig.show()
```

## Comparison

As a comparison, we have implemented a simple agent composed of only 1 node aiming create a project plan as a 1-short.
The input remained the `project_description` and the `team`. The prompt became the combination of the each node - except the risk and insight generations.
The single node agent with the simplied stage could create a project plan.

However, the expected complex structured response may not always be achieved by the LLM, resulting in error at downstream processing steps. In addition, a multi-agent solution allows for incorporation of self-reflection at different stages of the planning (in the future even human-in-the-loop) even including new information via user interactions (e.g. sickness , holidays of people team members).

```python
class ProjectPlan(BaseModel):
    tasks: TaskList
    dependencies: DependencyList
    schedule: Schedule
    task_allocations: TaskAllocationList

class SimpleAgentState(TypedDict):
    """The project manager agent state."""
    project_description: str
    team: Team
    tasks: TaskList
    dependencies: DependencyList
    schedule: Schedule
    task_allocations: TaskAllocationList

def project_plan_generation_node(state: SimpleAgentState):
    """LangGraph node that will extract tasks from given project description"""
    description = state["project_description"]
    team = state["team"]
    prompt = f"""You are an experienced project description analyzer, who needs to create a project plan.
        Create the project plan using the following steps:
        - Analyze the project description '{description}' and create a list of actionable and realistic tasks with estimated time (in days) to complete each task. If the task takes longer than 5 days, break it down into independent smaller tasks.
        - Assess dependency between tasks. For each task, identify the blocking tasks. Provide for each task the list of dependent tasks.
        - Schedule tasks based on the dependencies.
        - Allocate tasks to team members {team} based on their skills and availability, such that there is no overlapping task assigned for a team member. Ensure that no team member has 2 tasks assigned for the same time period.
    """
    structure_llm = llm.with_structured_output(ProjectPlan)
    project_plan: ProjectPlan = structure_llm.invoke(prompt)
    print(project_plan)
    return {"tasks": project_plan.tasks, "dependencies": project_plan.dependencies, "schedule": project_plan.schedule, "task_allocations": project_plan.task_allocations}


# Instantiate the workflow
simple_workflow = StateGraph(SimpleAgentState)

# Add nodes to the workflow
simple_workflow.add_node("create_project_plan", project_plan_generation_node)

# Add edges to the workflow
simple_workflow.set_entry_point("create_project_plan")
simple_workflow.add_edge("create_project_plan", END)

# Set up memory
simple_memory = MemorySaver()

# Compile the workflow
simple_graph_plan = simple_workflow.compile(checkpointer=memory)

# To visualize the created workflow, we can use
display(Image(simple_graph_plan.get_graph(xray=1).draw_mermaid_png()))
```

```text
<IPython.core.display.Image object>
```

![[Assets/Sources/GenAI Agents/all_agents_tutorials/project_manager_assistant_agent/cell-037-output-01.png]]

```python
# Invoke the agent
config = {"configurable": {"thread_id": "2"}}
for event in simple_graph_plan.stream(state_input, config, stream_mode=["updates"]):
    "Print the different nodes as the agent progresses"
    print(f"Current node: {next(iter(event[1]))}")

simple_final_state = simple_graph_plan.get_state(config).values
```

```text
tasks=TaskList(tasks=[Task(id=UUID('4addec1e-c3a4-41b8-886e-3bfecc9e4ebe'), task_name='Requirements Gathering', task_description='Gather requirements from stakeholders about the chatbot features and functionalities.', estimated_day=3), Task(id=UUID('cd3b2ccd-a274-4122-87e5-792c416e63bc'), task_name='Design Chatbot Architecture', task_description='Create the architecture of the chatbot application including frontend and backend design.', estimated_day=5), Task(id=UUID('e41834d4-afe8-46b9-8519-e4a2cd8aae15'), task_name='Frontend Development', task_description='Develop the user interface for the chatbot using HTML, CSS, and JavaScript.', estimated_day=7), Task(id=UUID('7b2710d8-b9e9-4028-8a26-435e6e89159c'), task_name='Backend Development', task_description='Set up the backend for the chatbot using Python and Django.', estimated_day=7), Task(id=UUID('232113dc-aef2-4c86-b8ba-43747e58d264'), task_name='Integrate Frontend and Backend', task_description='Integrate the frontend and backend components of the chatbot application.', estimated_day=3), Task(id=UUID('ca7f94ed-4eb3-4af0-88c5-c0b0aa478168'), task_name='Testing and QA', task_description='Conduct testing and quality assurance for the chatbot application.', estimated_day=4), Task(id=UUID('de60cddf-9831-4307-a5eb-cd19f2475bc6'), task_name='Deployment', task_description='Deploy the chatbot application to a cloud platform.', estimated_day=3), Task(id=UUID('a7f21afa-d2bb-4cb2-9f91-8985eeaac71b'), task_name='Post-Deployment Support', task_description='Provide post-deployment support and bug fixes for the chatbot application.', estimated_day=3)]) dependencies=DependencyList(dependencies=[TaskDependency(task=Task(id=UUID('d2654503-74f3-432b-b455-f9024f688a43'), task_name='Requirements Gathering', task_description='Gather requirements from stakeholders about the chatbot features and functionalities.', estimated_day=3), dependent_tasks=[Task(id=UUID('c2541e7a-c96e-4024-83d2-15c2716c75d0'), task_name='Design Chatbot Architecture', task_description='Create the architecture of the chatbot application including frontend and backend design.', estimated_day=5)]), TaskDependency(task=Task(id=UUID('f75fd162-0c48-406c-954c-7bfb55533a5a'), task_name='Design Chatbot Architecture', task_description='Create the architecture of the chatbot application including frontend and backend design.', estimated_day=5), dependent_tasks=[Task(id=UUID('2d58b977-bf73-4969-b569-ea5796e25431'), task_name='Frontend Development', task_description='Develop the user interface for the chatbot using HTML, CSS, and JavaScript.', estimated_day=7), Task(id=UUID('97a551b1-bf24-47a9-b1a3-353cbde711e2'), task_name='Backend Development', task_description='Set up the backend for the chatbot using Python and Django.', estimated_day=7)]), TaskDependency(task=Task(id=UUID('05d335b9-7733-49c7-962b-be3d81d1dfed'), task_name='Backend Development', task_description='Set up the backend for the chatbot using Python and Django.', estimated_day=7), dependent_tasks=[Task(id=UUID('0d5d648f-d12a-4d00-9357-61468a1fd9aa'), task_name='Integrate Frontend and Backend', task_description='Integrate the frontend and backend components of the chatbot application.', estimated_day=3)]), TaskDependency(task=Task(id=UUID('8340fd6d-e8a5-430b-97ae-46de2141c0f4'), task_name='Frontend Development', task_description='Develop the user interface for the chatbot using HTML, CSS, and JavaScript.', estimated_day=7), dependent_tasks=[Task(id=UUID('80382386-eacc-45b3-b147-af6b3c8a1ebf'), task_name='Integrate Frontend and Backend', task_description='Integrate the frontend and backend components of the chatbot application.', estimated_day=3)]), TaskDependency(task=Task(id=UUID('fa43ac3a-01f9-4cad-b900-cb33a08e2323'), task_name='Integrate Frontend and Backend', task_description='Integrate the frontend and backend components of the chatbot application.', estimated_day=3), dependent_tasks=[Task(id=UUID('54068596-0ac9-4b36-8963-930bbc5f89a1'), task_name='Testing and QA', task_description='Conduct testing and quality assurance for the chatbot application.', estimated_day=4)]), TaskDependency(task=Task(id=UUID('4b82aeb1-8d35-41a1-8924-55509f91b8bf'), task_name='Testing and QA', task_description='Conduct testing and quality assurance for the chatbot application.', estimated_day=4), dependent_tasks=[Task(id=UUID('be3cbd0a-5e34-4cd4-a9ba-7b7ecd252074'), task_name='Deployment', task_description='Deploy the chatbot application to a cloud platform.', estimated_day=3)]), TaskDependency(task=Task(id=UUID('a436ba1a-5804-4ca2-9534-7ff3e93e3864'), task_name='Deployment', task_description='Deploy the chatbot application to a cloud platform.', estimated_day=3), dependent_tasks=[Task(id=UUID('673909e0-3806-4e4d-9fd4-120d57dbf912'), task_name='Post-Deployment Support', task_description='Provide post-deployment support and bug fixes for the chatbot application.', estimated_day=3)])]) schedule=Schedule(schedule=[TaskSchedule(task=Task(id=UUID('d4d98552-61a5-433c-8454-83197da75e51'), task_name='Requirements Gathering', task_description='Gather requirements from stakeholders about the chatbot features and functionalities.', estimated_day=3), start_day=1, end_day=3), TaskSchedule(task=Task(id=UUID('f13cf1a8-2aeb-48ed-9dc9-35d6767c0a0a'), task_name='Design Chatbot Architecture', task_description='Create the architecture of the chatbot application including frontend and backend design.', estimated_day=5), start_day=4, end_day=8), TaskSchedule(task=Task(id=UUID('403fbfb2-ae48-4591-89de-ccb0bcea0122'), task_name='Frontend Development', task_description='Develop the user interface for the chatbot using HTML, CSS, and JavaScript.', estimated_day=7), start_day=9, end_day=15), TaskSchedule(task=Task(id=UUID('98016950-0f73-44b8-adbf-3c117d8f6685'), task_name='Backend Development', task_description='Set up the backend for the chatbot using Python and Django.', estimated_day=7), start_day=9, end_day=15), TaskSchedule(task=Task(id=UUID('2bf0bc35-71bf-4c01-834c-7c84cc57902f'), task_name='Integrate Frontend and Backend', task_description='Integrate the frontend and backend components of the chatbot application.', estimated_day=3), start_day=16, end_day=18), TaskSchedule(task=Task(id=UUID('60d20758-3067-46fa-b425-d6d67b666942'), task_name='Testing and QA', task_description='Conduct testing and quality assurance for the chatbot application.', estimated_day=4), start_day=19, end_day=22), TaskSchedule(task=Task(id=UUID('e2ef4752-0679-4caf-a773-3b06c94ac7f3'), task_name='Deployment', task_description='Deploy the chatbot application to a cloud platform.', estimated_day=3), start_day=23, end_day=25), TaskSchedule(task=Task(id=UUID('ba7a7fe3-094d-490b-bc47-bd54ca25d04b'), task_name='Post-Deployment Support', task_description='Provide post-deployment support and bug fixes for the chatbot application.', estimated_day=3), start_day=26, end_day=28)]) task_allocations=TaskAllocationList(task_allocations=[TaskAllocation(task=Task(id=UUID('eb47af24-a4d0-469a-945a-53be9baa4492'), task_name='Requirements Gathering', task_description='Gather requirements from stakeholders about the chatbot features and functionalities.', estimated_day=3), team_member=TeamMember(name='Charlie', profile=' Charlie is a Project Manager experienced in Agile methodologies team leadership project planning and risk management.')), TaskAllocation(task=Task(id=UUID('403a8a30-c1c9-4aec-a61f-3dfef18b0eb7'), task_name='Design Chatbot Architecture', task_description='Create the architecture of the chatbot application including frontend and backend design.', estimated_day=5), team_member=TeamMember(name='Bob', profile=' Bob is a Backend Developer proficient in Python Django SQL and RESTful APIs.')), TaskAllocation(task=Task(id=UUID('2a734eee-cbfa-49e8-b3ba-4e54eafa51c3'), task_name='Frontend Development', task_description='Develop the user interface for the chatbot using HTML, CSS, and JavaScript.', estimated_day=7), team_member=TeamMember(name='Alice', profile=' Alice is a Frontend Developer skilled in HTML CSS JavaScript and React.')), TaskAllocation(task=Task(id=UUID('e7ebeb83-5dc2-4b67-b8b3-a633f1daf551'), task_name='Backend Development', task_description='Set up the backend for the chatbot using Python and Django.', estimated_day=7), team_member=TeamMember(name='David', profile=' David is a Full Stack Developer with expertise in both frontend (HTML CSS JavaScript) and backend (Node.js MongoDB) technologies.')), TaskAllocation(task=Task(id=UUID('9d5f9e25-ab62-4dd0-b8eb-19538d3c9a87'), task_name='Integrate Frontend and Backend', task_description='Integrate the frontend and backend components of the chatbot application.', estimated_day=3), team_member=TeamMember(name='Frank', profile=' Frank is a Junior Frontend Developer with knowledge in HTML CSS JavaScript and basic React.')), TaskAllocation(task=Task(id=UUID('a0996072-a35e-499a-903e-a7ce04c2c681'), task_name='Testing and QA', task_description='Conduct testing and quality assurance for the chatbot application.', estimated_day=4), team_member=TeamMember(name='Eve', profile=' Eve is a DevOps Engineer skilled in CI/CD pipelines Docker Kubernetes and cloud services like AWS and Azure.')), TaskAllocation(task=Task(id=UUID('8db9696e-d68a-41cd-a884-388c8804b737'), task_name='Deployment', task_description='Deploy the chatbot application to a cloud platform.', estimated_day=3), team_member=TeamMember(name='Charlie', profile=' Charlie is a Project Manager experienced in Agile methodologies team leadership project planning and risk management.')), TaskAllocation(task=Task(id=UUID('f113d200-54b2-4593-858d-0ba7f3c92bb3'), task_name='Post-Deployment Support', task_description='Provide post-deployment support and bug fixes for the chatbot application.', estimated_day=3), team_member=TeamMember(name='Grace', profile=' Grace is a Senior Data Scientist with expertise in machine learning data analysis Python R and big data technologies like Hadoop and Spark.'))])
Current node: create_project_plan
```

```python
## Tasks schedule
task_schedules = simple_final_state['schedule'].schedule

t = []
# Iterate over the task_schedules and append each task's data to the DataFrame
for task_schedule in task_schedules:
    t.append([
        task_schedule.task.task_name,
        task_schedule.start_day,
        task_schedule.end_day
    ])

df_schedule = pd.DataFrame(t,columns=['task_name', 'start', 'end'])

## Tasks allocation
task_allocations = simple_final_state['task_allocations'].task_allocations

t = []
# Iterate over the task_schedules and append each task's data to the DataFrame
for task_allocation in task_allocations:
    t.append([
        task_allocation.task.task_name,
        task_allocation.team_member.name
    ])

df_allocation = pd.DataFrame(t,columns=['task_name', 'team_member'])

df = df_allocation.merge(df_schedule, on='task_name')

import plotly.express as px

from datetime import datetime, timedelta
# Get the current date
current_date = datetime.today()

# Convert start and end offsets to actual dates
df['start'] = df['start'].apply(lambda x: current_date + timedelta(days=x))
df['end'] = df['end'].apply(lambda x: current_date + timedelta(days=x))

df.rename(columns={'team_member': 'Team Member'}, inplace=True)
df.sort_values(by='Team Member', inplace=True)
# Create a Gantt chart
fig = px.timeline(df, x_start="start", x_end="end", y="task_name", color="Team Member", title=f"Gantt Chart - Oneshot Project Plan")

# Update layout for better visualization
fig.update_layout(
    xaxis_title="Timeline",
    yaxis_title="Tasks",
    yaxis=dict(autorange="reversed"),  # Reverse the y-axis to have tasks in the vertical side
    title_x=0.5
)

# Show the plot
fig.show()
```

## Additional Considerations

Improvement Possibilities:
- Incorporating a 'human-in-the-loop' mechanism as part of the self-reflection process can significantly enhance the system's effectiveness. This approach allows for the introduction of additional, real-time information about the schedule and the availability or status of team members. For instance, if a team member is currently sick, the agent would typically still assign tasks to them due to a lack of awareness. By integrating human oversight, such critical updates can be communicated to the system, ensuring that tasks are reassigned appropriately and the workload is distributed more efficiently.
Limitations of the Approach:
- Incorporating an optimizer based on extracted features generated by the LLM can provide better and more reproducible scheduling and task allocation. So only leveraging the agent to produce structured content from the project description, task dependencies and team member profiles, then use the optimizer to create the project plan (task assignment).

Limitations:
- The current approach relies on a Large Language Model (LLM) to assign risk scores to tasks. However, this method has inherent limitations. Even if the same person is assigned to the same task on the same schedule, the LLM may generate different risk scores each time. This inconsistency arises because the model's output can vary independently of the explicit details provided in the prompt. Consequently, this variability can lead to unpredictable risk assessments, potentially affecting the reliability and accuracy of task management.
- Although the prompts instructs to fullfill certain criteria, it is not always guaranteed, that it will achieve it.

## References

Currently no references are added to the tutorial.

![](https://europe-west1-genai-agents-views-tracker.cloudfunctions.net/genai-agents-tracker?notebook=all-agents-tutorials--project-manager-assistant-agent)
