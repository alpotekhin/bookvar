---
title: "HR AI Assistant"
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: bd681451b254ac1a790e947b581d3997ab35013d
---

> [!note] Original source material
> This page preserves [`all_agents_tutorials/Hr_AI_Agent.ipynb`](https://github.com/NirDiamant/GenAI_Agents/blob/bd681451b254ac1a790e947b581d3997ab35013d/all_agents_tutorials/Hr_AI_Agent.ipynb) from
> *GenAI Agents* at commit `bd681451b254ac1a790e947b581d3997ab35013d`. License:
> [Custom non-commercial license](https://github.com/NirDiamant/GenAI_Agents/blob/bd681451b254ac1a790e947b581d3997ab35013d/LICENSE). Bookvar changed only
> publication markup, link paths, and characters required for safe rendering.



## In this project, i create a recruitment workflow using LangGraph, LangChain, and various APIs to automate and streamline the job posting and candidate evaluation process. The workflow consists of the following key steps:

* Requirements Gathering: The AI agent prompts the user for detailed job requirements, including the job title, company description, candidate responsibilities and qualifications, preferred location, and other relevant details.

* Job Description Generation: Once the job requirements are gathered, the agent generates a professional and compelling job description. The user can then review and approve the description, or provide feedback for the agent to refine it.

* LinkedIn Candidate Search and Outreach: If the user provides specific LinkedIn profiles of preferred candidates, the agent will directly message them about the opportunity. Alternatively, the agent can search LinkedIn for relevant candidates based on the job details and send outreach messages.

* CV Analysis: As candidates submit their CVs, the agent evaluates them against the job requirements, providing a score and recommendation (approve or reject) for each applicant. The agent then sends the appropriate response message to the candidate via LinkedIn.

* Interview Question Preparation: For approved candidates, the agent generates a set of interview questions covering technical skills, experience, and problem-solving. The user can review and approve the questions before the interviews are conducted.

## Installing required packages

```python
!pip install langchain langchain-anthropic langgraph python-dotenv pydantic langchain_community requests
```

```text
Requirement already satisfied: langchain in /usr/local/lib/python3.10/dist-packages (0.3.7)
Collecting langchain-anthropic
  Downloading langchain_anthropic-0.3.0-py3-none-any.whl.metadata (2.3 kB)
Collecting langgraph
  Downloading langgraph-0.2.50-py3-none-any.whl.metadata (15 kB)
Collecting python-dotenv
  Downloading python_dotenv-1.0.1-py3-none-any.whl.metadata (23 kB)
Requirement already satisfied: pydantic in /usr/local/lib/python3.10/dist-packages (2.9.2)
Collecting langchain_community
  Downloading langchain_community-0.3.7-py3-none-any.whl.metadata (2.9 kB)
Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (2.32.3)
Requirement already satisfied: PyYAML>=5.3 in /usr/local/lib/python3.10/dist-packages (from langchain) (6.0.2)
Requirement already satisfied: SQLAlchemy<3,>=1.4 in /usr/local/lib/python3.10/dist-packages (from langchain) (2.0.36)
Requirement already satisfied: aiohttp<4.0.0,>=3.8.3 in /usr/local/lib/python3.10/dist-packages (from langchain) (3.10.10)
Requirement already satisfied: async-timeout<5.0.0,>=4.0.0 in /usr/local/lib/python3.10/dist-packages (from langchain) (4.0.3)
Requirement already satisfied: langchain-core<0.4.0,>=0.3.15 in /usr/local/lib/python3.10/dist-packages (from langchain) (0.3.17)
Requirement already satisfied: langchain-text-splitters<0.4.0,>=0.3.0 in /usr/local/lib/python3.10/dist-packages (from langchain) (0.3.2)
Requirement already satisfied: langsmith<0.2.0,>=0.1.17 in /usr/local/lib/python3.10/dist-packages (from langchain) (0.1.142)
Requirement already satisfied: numpy<2,>=1 in /usr/local/lib/python3.10/dist-packages (from langchain) (1.26.4)
Requirement already satisfied: tenacity!=8.4.0,<10,>=8.1.0 in /usr/local/lib/python3.10/dist-packages (from langchain) (9.0.0)
Collecting anthropic<1,>=0.39.0 (from langchain-anthropic)
  Downloading anthropic-0.39.0-py3-none-any.whl.metadata (22 kB)
Requirement already satisfied: defusedxml<0.8.0,>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from langchain-anthropic) (0.7.1)
Collecting langgraph-checkpoint<3.0.0,>=2.0.4 (from langgraph)
  Downloading langgraph_checkpoint-2.0.4-py3-none-any.whl.metadata (4.6 kB)
Collecting langgraph-sdk<0.2.0,>=0.1.32 (from langgraph)
  Downloading langgraph_sdk-0.1.36-py3-none-any.whl.metadata (1.8 kB)
Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.10/dist-packages (from pydantic) (0.7.0)
Requirement already satisfied: pydantic-core==2.23.4 in /usr/local/lib/python3.10/dist-packages (from pydantic) (2.23.4)
Requirement already satisfied: typing-extensions>=4.6.1 in /usr/local/lib/python3.10/dist-packages (from pydantic) (4.12.2)
Collecting SQLAlchemy<3,>=1.4 (from langchain)
  Downloading SQLAlchemy-2.0.35-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (9.6 kB)
Collecting dataclasses-json<0.7,>=0.5.7 (from langchain_community)
  Downloading dataclasses_json-0.6.7-py3-none-any.whl.metadata (25 kB)
Collecting httpx-sse<0.5.0,>=0.4.0 (from langchain_community)
  Downloading httpx_sse-0.4.0-py3-none-any.whl.metadata (9.0 kB)
Collecting pydantic-settings<3.0.0,>=2.4.0 (from langchain_community)
  Downloading pydantic_settings-2.6.1-py3-none-any.whl.metadata (3.5 kB)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests) (3.4.0)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests) (2.2.3)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests) (2024.8.30)
Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (2.4.3)
Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.3.1)
Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (24.2.0)
Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.5.0)
Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (6.1.0)
Requirement already satisfied: yarl<2.0,>=1.12.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.17.1)
Requirement already satisfied: anyio<5,>=3.5.0 in /usr/local/lib/python3.10/dist-packages (from anthropic<1,>=0.39.0->langchain-anthropic) (3.7.1)
Requirement already satisfied: distro<2,>=1.7.0 in /usr/local/lib/python3.10/dist-packages (from anthropic<1,>=0.39.0->langchain-anthropic) (1.9.0)
Requirement already satisfied: httpx<1,>=0.23.0 in /usr/local/lib/python3.10/dist-packages (from anthropic<1,>=0.39.0->langchain-anthropic) (0.27.2)
Requirement already satisfied: jiter<1,>=0.4.0 in /usr/local/lib/python3.10/dist-packages (from anthropic<1,>=0.39.0->langchain-anthropic) (0.7.1)
Requirement already satisfied: sniffio in /usr/local/lib/python3.10/dist-packages (from anthropic<1,>=0.39.0->langchain-anthropic) (1.3.1)
Collecting marshmallow<4.0.0,>=3.18.0 (from dataclasses-json<0.7,>=0.5.7->langchain_community)
  Downloading marshmallow-3.23.1-py3-none-any.whl.metadata (7.5 kB)
Collecting typing-inspect<1,>=0.4.0 (from dataclasses-json<0.7,>=0.5.7->langchain_community)
  Downloading typing_inspect-0.9.0-py3-none-any.whl.metadata (1.5 kB)
Requirement already satisfied: jsonpatch<2.0,>=1.33 in /usr/local/lib/python3.10/dist-packages (from langchain-core<0.4.0,>=0.3.15->langchain) (1.33)
Requirement already satisfied: packaging<25,>=23.2 in /usr/local/lib/python3.10/dist-packages (from langchain-core<0.4.0,>=0.3.15->langchain) (24.2)
Requirement already satisfied: msgpack<2.0.0,>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from langgraph-checkpoint<3.0.0,>=2.0.4->langgraph) (1.1.0)
Requirement already satisfied: orjson>=3.10.1 in /usr/local/lib/python3.10/dist-packages (from langgraph-sdk<0.2.0,>=0.1.32->langgraph) (3.10.11)
Requirement already satisfied: requests-toolbelt<2.0.0,>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from langsmith<0.2.0,>=0.1.17->langchain) (1.0.0)
Requirement already satisfied: greenlet!=0.4.17 in /usr/local/lib/python3.10/dist-packages (from SQLAlchemy<3,>=1.4->langchain) (3.1.1)
Requirement already satisfied: exceptiongroup in /usr/local/lib/python3.10/dist-packages (from anyio<5,>=3.5.0->anthropic<1,>=0.39.0->langchain-anthropic) (1.2.2)
Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.10/dist-packages (from httpx<1,>=0.23.0->anthropic<1,>=0.39.0->langchain-anthropic) (1.0.6)
Requirement already satisfied: h11<0.15,>=0.13 in /usr/local/lib/python3.10/dist-packages (from httpcore==1.*->httpx<1,>=0.23.0->anthropic<1,>=0.39.0->langchain-anthropic) (0.14.0)
Requirement already satisfied: jsonpointer>=1.9 in /usr/local/lib/python3.10/dist-packages (from jsonpatch<2.0,>=1.33->langchain-core<0.4.0,>=0.3.15->langchain) (3.0.0)
Collecting mypy-extensions>=0.3.0 (from typing-inspect<1,>=0.4.0->dataclasses-json<0.7,>=0.5.7->langchain_community)
  Downloading mypy_extensions-1.0.0-py3-none-any.whl.metadata (1.1 kB)
Requirement already satisfied: propcache>=0.2.0 in /usr/local/lib/python3.10/dist-packages (from yarl<2.0,>=1.12.0->aiohttp<4.0.0,>=3.8.3->langchain) (0.2.0)
Downloading langchain_anthropic-0.3.0-py3-none-any.whl (22 kB)
Downloading langgraph-0.2.50-py3-none-any.whl (124 kB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m124.9/124.9 kB[0m [31m3.0 MB/s[0m eta [36m0:00:00[0m
[?25hDownloading python_dotenv-1.0.1-py3-none-any.whl (19 kB)
Downloading langchain_community-0.3.7-py3-none-any.whl (2.4 MB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.4/2.4 MB[0m [31m18.7 MB/s[0m eta [36m0:00:00[0m
[?25hDownloading anthropic-0.39.0-py3-none-any.whl (198 kB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m198.4/198.4 kB[0m [31m9.3 MB/s[0m eta [36m0:00:00[0m
[?25hDownloading dataclasses_json-0.6.7-py3-none-any.whl (28 kB)
Downloading httpx_sse-0.4.0-py3-none-any.whl (7.8 kB)
Downloading langgraph_checkpoint-2.0.4-py3-none-any.whl (23 kB)
Downloading langgraph_sdk-0.1.36-py3-none-any.whl (29 kB)
Downloading pydantic_settings-2.6.1-py3-none-any.whl (28 kB)
Downloading SQLAlchemy-2.0.35-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.1 MB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.1/3.1 MB[0m [31m14.3 MB/s[0m eta [36m0:00:00[0m
[?25hDownloading marshmallow-3.23.1-py3-none-any.whl (49 kB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m49.5/49.5 kB[0m [31m2.5 MB/s[0m eta [36m0:00:00[0m
[?25hDownloading typing_inspect-0.9.0-py3-none-any.whl (8.8 kB)
Downloading mypy_extensions-1.0.0-py3-none-any.whl (4.7 kB)
Installing collected packages: SQLAlchemy, python-dotenv, mypy-extensions, marshmallow, httpx-sse, typing-inspect, pydantic-settings, langgraph-sdk, dataclasses-json, anthropic, langgraph-checkpoint, langchain-anthropic, langgraph, langchain_community
  Attempting uninstall: SQLAlchemy
    Found existing installation: SQLAlchemy 2.0.36
    Uninstalling SQLAlchemy-2.0.36:
      Successfully uninstalled SQLAlchemy-2.0.36
Successfully installed SQLAlchemy-2.0.35 anthropic-0.39.0 dataclasses-json-0.6.7 httpx-sse-0.4.0 langchain-anthropic-0.3.0 langchain_community-0.3.7 langgraph-0.2.50 langgraph-checkpoint-2.0.4 langgraph-sdk-0.1.36 marshmallow-3.23.1 mypy-extensions-1.0.0 pydantic-settings-2.6.1 python-dotenv-1.0.1 typing-inspect-0.9.0
```

```python
# Core imports
import os
from typing import Dict, Any, List, Optional, Literal, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from datetime import datetime
import operator
import json
import uuid
import requests

# LangChain imports
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper

# LangGraph imports
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.errors import NodeInterrupt

# Display imports
from IPython.display import Image, display
```

```python
# API Keys
os.environ["ANTHROPIC_API_KEY"] = "anthropic_api_key"
os.environ["SERPER_API_KEY"] = "Serper_api_key"
os.environ["LINKEDIN_COOKIE"] = "linkedin_cookie"

llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)

# Initialize search utility
search = GoogleSerperAPIWrapper()
```

# Defining base models

```python
# Job requirements model
class JobRequirements(BaseModel):
    # Details about the job requirements
    title: str
    company_description: str
    job_requirements: List[str]
    candidate_responsibilities: List[str]
    candidate_qualifications: List[str]
    company_benefits: List[str]
    interview_date: datetime
    preferred_country: str
    years_experience: int
    linkedin_profiles: Optional[List[str]​] = None
    skills_required: List[str]
    salary_range: str



# Candidate profile model
class CandidateProfile(BaseModel):
    # Details about the candidate
    name: str
    linkedin_url: str
    title: str
    location: str
    cv_content: Optional[str] = None
    cv_score: Optional[float] = None
    status: str = "new"  # 'new', 'contacted', 'cv_received', 'approved', 'rejected'
    feedback: Optional[str] = None
    source: str = "direct"  # 'direct' or 'search'
    skill_matches: Optional[List[str]​] = None
    skill_gaps: Optional[List[str]​] = None
    message_sent: Optional[str] = None

# Main State Model
class RecruitmentState(BaseModel):
    # Overall state of the recruitment process
    phase: str
    messages: Annotated[List[BaseMessage], add_messages]
    job_requirements: Optional[JobRequirements]
    job_description: Optional[str]
    job_description_approved: bool
    candidates: Annotated[List[CandidateProfile], operator.add]
    linkedin_process_complete: bool
    cv_analysis_complete: bool
    interview_questions: Optional[List[str]​]
    interview_questions_approved: bool
```

# Defining tool functions

```python
# Search for LinkedIn candidates using Google Serper API
@tool
def search_linkedin_candidates(
    job_title: str,
    location: str,
    skills: List[str],
    limit: int = 5
) -> List[Dict[str, str]​]:
    """Search for candidates on LinkedIn using Google Search API"""
    search = GoogleSerperAPIWrapper()
    search_query = f"""site:linkedin.com/in/
        {job_title}
        {location}
        {' '.join(skills)}"""

    try:
        results = search.results(search_query)
        candidates = []
        for result in results.get('organic', [])[:limit]:
            if "linkedin.com/in/" in result.get("link", ""):
                candidates.append({
                    "linkedin_url": result["link"],
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", "")
                })
        return candidates
    except Exception as e:
        return [{"error": str(e)}]


# Get detailed LinkedIn profile information
@tool
def get_linkedin_profile(profile_url: str) -> Dict[str, Any]:
    """Get detailed profile information from LinkedIn URL"""
    headers = {
        'cookie': os.getenv("LINKEDIN_COOKIE"),
        'accept': 'application/json'
    }
    try:
        profile_id = profile_url.split('/in/')[-1].split('/')[0]
        api_url = f"https://www.linkedin.com/voyager/api/identity/profiles/{profile_id}/profileView"
        response = requests.get(api_url, headers=headers)
        data = response.json()

        return {
            "name": f"{data.get('firstName', '')} {data.get('lastName', '')}",
            "title": data.get('headline', ''),
            "location": data.get('locationName', ''),
            "skills": [skill.get('name', '') for skill in data.get('skills', [])]
        }
    except Exception as e:
        return {"error": str(e)}


# Send a message to a LinkedIn candidate about the job opportunity
@tool
def send_linkedin_message(profile_url: str, job_details: str) -> Dict[str, str]:
    """Send a LinkedIn message to a candidate about the job opportunity"""
    headers = {
        'cookie': os.getenv("LINKEDIN_COOKIE"),
        'accept': 'application/json'
    }
    try:
        profile_id = profile_url.split('/in/')[-1].split('/')[0]

        # Generate personalized message
        message_prompt = [
            SystemMessage(content="Generate a personalized LinkedIn outreach message for a job opportunity. Ask them to submit their CV directly through LinkedIn messages if interested."),
            HumanMessage(content=f"Job Details:\n{job_details}")
        ]
        message_content = llm.invoke(message_prompt).content

        # Send message via LinkedIn API
        message_endpoint = f"https://www.linkedin.com/voyager/api/messaging/conversations"
        payload = {
            "recipients": [profile_id],
            "message": message_content
        }
        response = requests.post(message_endpoint, headers=headers, json=payload)

        return {
            "status": "sent",
            "profile_id": profile_id,
            "message": message_content
        }
    except Exception as e:
        return {"error": str(e)}
```

# Defining workflow nodes

```python
# Requirements gathering node
def requirements_gathering(state: RecruitmentState) -> Dict[str, Any]:
    """Initialize or continue requirements gathering process"""
    if not state.messages:
        job_title = input("What is the job title? ")
        company_description = input("Provide a description of the company: ")
        job_requirements = input("List the job requirements (comma-separated): ").split(",")
        candidate_responsibilities = input("List the candidate responsibilities (comma-separated): ").split(",")
        candidate_qualifications = input("List the candidate qualifications (comma-separated): ").split(",")
        company_benefits = input("List the company benefits (comma-separated): ").split(",")
        interview_date = input("What is the interview date (YYYY-MM-DD)? ")
        preferred_country = input("What is the preferred country for the role? ")
        years_experience = int(input("What is the required years of experience? "))
        linkedin_profiles = input("Provide any LinkedIn profile URLs (comma-separated, optional): ").split(",")
        skills_required = input("List the required skills (comma-separated): ").split(",")
        salary_range = input("What is the salary range for the role? ")
        google_form_url = input("Provide the Google Form URL for CV submissions: ")

        job_requirements = JobRequirements(
            title=job_title,
            company_description=company_description,
            job_requirements=job_requirements,
            candidate_responsibilities=candidate_responsibilities,
            candidate_qualifications=candidate_qualifications,
            company_benefits=company_benefits,
            interview_date=datetime.strptime(interview_date, "%Y-%m-%d"),
            preferred_country=preferred_country,
            years_experience=years_experience,
            linkedin_profiles=[p.strip() for p in linkedin_profiles if p.strip()],
            skills_required=skills_required,
            salary_range=salary_range,
            google_form_url=google_form_url
        )

        return {
            "messages": [
                SystemMessage(content="You are an HR assistant gathering detailed job requirements."),
                HumanMessage(content="Let's begin gathering the job requirements. What is the job title?")
            ],
            "job_requirements": job_requirements,
            "phase": "requirements_gathering"
        }
    else:
        return state


# Job description generation node
def generate_job_desc(state: RecruitmentState) -> Dict[str, Any]:
    """Generate job description with human-in-the-loop review"""
    if not state.job_requirements:
        raise ValueError("Job requirements missing")

    messages = [
        SystemMessage(content="""Create a professional and compelling job description with:
        1. About the Company
        2. Role Overview
        3. Key Responsibilities
        4. Required Qualifications
        5. What We Offer (Benefits)
        6. Location and Work Mode"""),
        HumanMessage(content=str(state.job_requirements.model_dump()))
    ]

    response = llm.invoke(messages)

    # Raise NodeInterrupt for human review
    raise NodeInterrupt(
        f"Please review the generated job description:\n\n{response.content}"
    )
```

# LinkedIn process node

```python
def linkedin_process(state: RecruitmentState) -> Dict[str, Any]:
     # Implementation to handle the LinkedIn candidate search and outreach process
    """Handle LinkedIn candidate search/outreach process"""
    candidates = []
    tool_node = ToolNode([search_linkedin_candidates, get_linkedin_profile, send_linkedin_message])

    job_details = f"""
    Role: {state.job_requirements.title}
    Company: {state.job_requirements.company_description}
    Location: {state.job_requirements.preferred_country}
    Description: {state.job_description}
    """

    if state.job_requirements.linkedin_profiles:
        # Process provided profiles
        for profile_url in state.job_requirements.linkedin_profiles:
            profile = tool_node.invoke({
                "name": "get_linkedin_profile",
                "args": {"profile_url": profile_url}
            })

            if "error" not in profile:
                message_result = tool_node.invoke({
                    "name": "send_linkedin_message",
                    "args": {
                        "profile_url": profile_url,
                        "job_details": job_details
                    }
                })

                if "error" not in message_result:
                    candidates.append(
                        CandidateProfile(
                            **profile,
                            linkedin_url=profile_url,
                            status="contacted",
                            message_sent=message_result["message"]
                        )
                    )
    else:
        # Search for candidates using Serper API
        search_results = tool_node.invoke({
            "name": "search_linkedin_candidates",
            "args": {
                "job_title": state.job_requirements.title,
                "location": state.job_requirements.preferred_country,
                "skills": state.job_requirements.skills_required
            }
        })

        for result in search_results:
            if "error" not in result:
                profile = tool_node.invoke({
                    "name": "get_linkedin_profile",
                    "args": {"profile_url": result["linkedin_url"]}
                })

                if "error" not in profile:
                    message_result = tool_node.invoke({
                        "name": "send_linkedin_message",
                        "args": {
                            "profile_url": result["linkedin_url"],
                            "job_details": job_details
                        }
                    })

                    if "error" not in message_result:
                        candidates.append(
                            CandidateProfile(
                                **profile,
                                linkedin_url=result["linkedin_url"],
                                status="contacted",
                                source="search",
                                message_sent=message_result["message"]
                            )
                        )

    return {
        "candidates": candidates,
        "linkedin_process_complete": True,
        "phase": "analyze_cv"
    }
```

# CV analysis node

```python
def analyze_cv(state: RecruitmentState) -> Dict[str, Any]:
    # Implementation to analyze candidate CVs and send appropriate responses
    """Analyze CV and send appropriate response"""
    messages = [
        SystemMessage(content="""Analyze the CV against job requirements. Score from 0-10 on:
        1. Skills Match
        2. Experience Level
        3. Overall Fit

        Provide detailed feedback and clear recommendation."""),
        HumanMessage(content=f"""
        Job Requirements:
        {state.job_requirements.model_dump_json()}

        CV Content:
        {state.candidates[-1].cv_content}
        """)
    ]

    analysis = llm.invoke(messages)
    score = float(re.search(r"Overall Score:\s*(\d+\.?\d*)", analysis.content).group(1))

    tool_node = ToolNode([send_linkedin_message])

    if score >= 7.0:
        status = "approved"
        message = f"Congratulations! You've been selected for an interview on {state.job_requirements.interview_date}"
    else:
        status = "rejected"
        message = "Thank you for your application. Unfortunately..."

    # Send response via LinkedIn
    tool_node.invoke({
        "name": "send_linkedin_message",
        "args": {
            "profile_url": state.candidates[-1].linkedin_url,
            "message": message
        }
    })

    return {
        "cv_score": score,
        "status": status,
        "feedback": analysis.content,
        "phase": "prepare_interview" if status == "approved" else "complete"
    }


# Interview preparation node
def prepare_interview(state: RecruitmentState) -> Dict[str, Any]:
    """Generate interview questions with human approval"""
    messages = [
        SystemMessage(content="""Generate 10 interview questions covering:
        - Technical Skills (4)
        - Experience (3)
        - Problem Solving (3)"""),
        HumanMessage(content=f"""
        Position: {state.job_requirements.title}
        Required Skills: {', '.join(state.job_requirements.skills_required)}
        """)
    ]

    response = llm.invoke(messages)

    # Raise NodeInterrupt for human review
    raise NodeInterrupt(
        f"Please review the interview questions:\n\n{response.content}"
    )
```

# Creating the recruitment workflow

```python
def create_recruitment_workflow():
    """Create the recruitment workflow graph"""
    workflow = StateGraph(RecruitmentState)
    memory = MemorySaver()

    # Add nodes
    workflow.add_node("requirements_gathering", requirements_gathering)
    workflow.add_node("generate_job_desc", generate_job_desc)
    workflow.add_node("linkedin_process", linkedin_process)
    workflow.add_node("analyze_cv", analyze_cv)
    workflow.add_node("prepare_interview", prepare_interview)

    # Add edges with human-in-the-loop cycles
    workflow.add_edge(START, "requirements_gathering")
    workflow.add_edge("requirements_gathering", "generate_job_desc")

    def route_after_job_desc(state: RecruitmentState):
        return "linkedin_process" if state.job_description_approved else "generate_job_desc"

    def route_after_cv(state: RecruitmentState):
        return "prepare_interview" if state.status == "approved" else END

    workflow.add_conditional_edges(
        "generate_job_desc",
        route_after_job_desc,
        ["linkedin_process", "generate_job_desc"]
    )

    workflow.add_edge("linkedin_process", "analyze_cv")
    workflow.add_conditional_edges(
        "analyze_cv",
        route_after_cv,
        ["prepare_interview", END]
    )
    workflow.add_edge("prepare_interview", END)

    # Compile with breakpoints
    graph = workflow.compile(
        checkpointer=memory,
        interrupt_before=["generate_job_desc", "prepare_interview"]
    )

    # Generate and display the Mermaid visualization
    print("""graph TD
        Start --> requirements_gathering
        requirements_gathering --> generate_job_desc
        generate_job_desc -->|Approved| linkedin_process
        generate_job_desc -->|Not Approved| generate_job_desc
        linkedin_process --> analyze_cv
        analyze_cv -->|CV-Good| prepare_interview
        analyze_cv -->|CV-Bad| End
        prepare_interview -->|Approved| End
        prepare_interview -->|Not Approved| prepare_interview
    """)
    display(Image(graph.get_graph().draw_mermaid_png()))

    return graph
```

```python
def get_job_requirements():
    job_title = input("What is the job title? ")
    company_description = input("Provide a description of the company: ")
    job_requirements = input("List the job requirements (comma-separated): ").split(",")
    candidate_responsibilities = input("List the candidate responsibilities (comma-separated): ").split(",")
    candidate_qualifications = input("List the candidate qualifications (comma-separated): ").split(",")
    company_benefits = input("List the company benefits (comma-separated): ").split(",")
    interview_date = input("What is the interview date (YYYY-MM-DD)? ")
    preferred_country = input("What is the preferred country for the role? ")
    years_experience = int(input("What is the required years of experience? "))
    linkedin_profiles = input("Provide any LinkedIn profile URLs (comma-separated, optional): ").split(",")
    skills_required = input("List the required skills (comma-separated): ").split(",")
    salary_range = input("What is the salary range for the role? ")

    return JobRequirements(
        title=job_title,
        company_description=company_description,
        job_requirements=job_requirements,
        candidate_responsibilities = candidate_responsibilities,
        candidate_qualifications = candidate_qualifications,
        company_benefits = company_benefits,
        interview_date = interview_date,
        preferred_country = preferred_country,
        years_experience = years_experience,
        linkedin_profiles = linkedin_profiles,
        skills_required = skills_required,
        salary_range = salary_range,
    )
```

```python
# Create the recruitment workflow instance
recruitment_workflow = create_recruitment_workflow()

# Set the test configuration
config = {"configurable": {"thread_id": "test_1"}}

# Define the initial state of the recruitment process
initial_state = {
    "phase": "requirements_gathering",
    "messages": [],
    "job_requirements": None,
    "job_description": None,
    "job_description_approved": False,
    "candidates": [],
    "linkedin_process_complete": False,
    "cv_analysis_complete": False,
    "interview_questions": None,
    "interview_questions_approved": False
}

# Run the workflow and handle any interrupts
try:
    for event in recruitment_workflow.stream(initial_state, config, stream_mode="values"):
        print("\nNew State Update:")
        print(f"Phase: {event.get('phase')}")
        if event.get('messages'):
            print(f"Latest Message: {event['messages'][-1].content}")
except NodeInterrupt as e:
    print(f"\nHuman Review Required: {str(e)}")

    # Example of handling job description approval
    recruitment_workflow.update_state(
        config,
        {"job_description_approved": True}
    )

    # Continue the workflow execution
    for event in recruitment_workflow.stream(None, config, stream_mode="values"):
        print("\nNew State Update:")
        print(f"Phase: {event.get('phase')}")
        if event.get('messages'):
            print(f"Latest Message: {event['messages'][-1].content}")
```

```text
graph TD
        Start --> requirements_gathering
        requirements_gathering --> generate_job_desc
        generate_job_desc -->|Approved| linkedin_process
        generate_job_desc -->|Not Approved| generate_job_desc
        linkedin_process --> analyze_cv
        analyze_cv -->|CV-Good| prepare_interview
        analyze_cv -->|CV-Bad| End
        prepare_interview -->|Approved| End
        prepare_interview -->|Not Approved| prepare_interview
```

```text
<IPython.core.display.Image object>
```

![[Assets/Sources/GenAI Agents/all_agents_tutorials/Hr_AI_Agent/cell-020-output-02.png]]

```text

New State Update:
Phase: requirements_gathering
```

```python

```

![](https://europe-west1-genai-agents-views-tracker.cloudfunctions.net/genai-agents-tracker?notebook=all-agents-tutorials--hr-ai-agent)
