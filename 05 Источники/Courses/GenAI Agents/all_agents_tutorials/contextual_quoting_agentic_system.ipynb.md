---
title: "**Dynamic Quoting System for Complex Products Using Multi-Agent AI Architecture**"
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: bd681451b254ac1a790e947b581d3997ab35013d
---

> [!note] Original source material
> This page preserves [`all_agents_tutorials/contextual_quoting_agentic_system.ipynb`](https://github.com/NirDiamant/GenAI_Agents/blob/bd681451b254ac1a790e947b581d3997ab35013d/all_agents_tutorials/contextual_quoting_agentic_system.ipynb) from
> *GenAI Agents* at commit `bd681451b254ac1a790e947b581d3997ab35013d`. License:
> [Custom non-commercial license](https://github.com/NirDiamant/GenAI_Agents/blob/bd681451b254ac1a790e947b581d3997ab35013d/LICENSE). Bookvar changed only
> publication markup, link paths, and characters required for safe rendering.



## **Project Overview**
This project implements a sophisticated multi-agent system for generating personalized, context-aware quotes for complex products and services. It leverages LangChain/LangGraph to create an intelligent quoting workflow that goes beyond traditional pricing methods by incorporating:

- **Retrieval-Augmented Generation (RAG)** for context-sensitive data retrieval
- **Multi-Agent Architecture** with specialized roles:
  - **Main Assistant**: Initial information gathering
  - **Underwriting Assistant**: Risk evaluation
  - **Quote Assistant**: Premium calculation
- **Intelligent Classification** using sentiment analysis and business context
- **Dynamic Workflow Management** through a state graph system
- **Database Integration** for storing and retrieving category rates

## **Motivation**
Traditional quoting systems often struggle with complex products and services where pricing depends on multiple interrelated factors. Current solutions typically:
- Rely heavily on manual intervention
- Have limited ability to consider context
- Struggle with non-standard cases
- Lack consistency across different underwriters
- Cannot easily adapt to changing market conditions

This project addresses these limitations by creating an intelligent system that can understand context, learn from historical data, and provide consistent, accurate quotes while reducing manual effort and turnaround time.

## **General Use Cases**
This system is particularly valuable in industries such as:

1. **Insurance**
   - Commercial insurance quoting
   - Risk assessment and premium calculation
   - Policy customization based on business specifics

2. **Professional Services**
   - Consulting service pricing
   - Project cost estimation
   - Service package customization

3. **Manufacturing**
   - Custom product pricing
   - Bill of materials calculation
   - Production cost estimation

4. **Construction**
   - Project bidding
   - Material and labor cost estimation
   - Risk-adjusted pricing

## **Key Components**

1. **Core Infrastructure**
   - SQLite database for storing category rates
   - Pydantic schemas for data validation
   - State management using TypedDict
   - LangGraph for workflow orchestration

2. **Specialized Agents**
   - **Retriever Agent**: Extracts and summarizes business operations
   - **Reasoning Agent**: Determines relevant insurance categories
   - **Classification Grading Agent**: Evaluates category assignments
   - **Quote Generation Agent**: Calculates final premiums

3. **Workflow Management**
   - Dynamic routing between agents
   - State tracking across conversation flows
   - Error handling and fallback mechanisms
   - Tool management for specific actions

## **Technical Implementation**
The system uses a graph-based architecture where nodes represent different stages of the quoting process and edges define the possible transitions between these stages. The workflow is managed through:

- **State Management**: Tracking conversation context and business information
- **Routing Logic**: Determining which agent handles each step
- **Tool Integration**: Specialized functions for specific tasks
- **Memory Management**: Maintaining conversation history and context

## **Conclusion**
This multi-agent quoting system represents a significant advancement in automated pricing solutions. By combining AI-driven analysis, context awareness, and flexible workflow management, it addresses the complexities of modern pricing challenges while providing several key benefits:

### **Business Impact**
- **Reduced Manual Effort**: Automation of complex pricing decisions
- **Increased Consistency**: Standardized approach across all quotes
- **Faster Turnaround**: Reduced time from inquiry to quote
- **Better Accuracy**: Context-aware pricing decisions
- **Scalability**: Ability to handle increased quote volumes without proportional staff increases

### **Technical Achievement**
- Demonstrates the power of multi-agent architectures in solving complex business problems
- Shows how different AI components can work together in a coordinated workflow
- Provides a blueprint for building similar systems in other domains

### **Future Potential**
The architecture can be extended to include:
- Additional specialized agents for specific industries
- More sophisticated pricing models
- Integration with external data sources
- Machine learning components for continuous improvement

This system serves as a model for how AI can transform complex business processes, making them more efficient, accurate, and scalable while maintaining the flexibility to adapt to changing business needs.

# Import Necessary Libraries

```python
# ! pip install langgraph
# ! pip install langchain-core 
# ! pip install langchain-openai 
# ! pip install langchain-groq 
# ! pip install langchain-community 
# ! pip install python-dotenv 
# ! pip install pydantic 
# ! pip install typing-extensions 
# ! pip install chromadb 
# ! pip install langchain-text-splitters
```

```python
from typing import Dict, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from IPython.display import display, Image
from langchain_core.runnables.graph import MermaidDrawMethod
from dotenv import load_dotenv
import os
```

# Setup

## Environment Variables and Enable Tracing for LangGraph

```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
_set_env("LANGCHAIN_API_KEY")
_set_env("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Hackathon-Contextual-Quoting"
```

## LLM

To start with, we define the LLM that powers our contextual quoting agents. Here’s the code for initializing the LLM:

```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# llm = ChatGroq(model="llama3-groq-8b-8192-tool-use-preview", temperature=0)
```

# Create and Populate Database

In this section, we create and populate a database to store category rates for our contextual quoting agent. 

We use **SQLite**, a lightweight and efficient database, which works well for projects where simplicity and speed are key considerations.

```python
import sqlite3

# Connect to SQLite (or create the database if it doesn't exist)
conn = sqlite3.connect('data/categories.db')

# Create a cursor object to execute SQL commands
cursor = conn.cursor()

# Create the category_rates table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS category_rates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT NOT NULL,
        rate REAL NOT NULL
    )
''')

# Data to be inserted into the category_rates table
categories = [
    ('ADMINISTRATIVE', 0.61),
    ('ADVERTISING', 5.31),
    ('ARTISAN CONTRACTORS', 2.35),
    ('CLUBS/ASSOCIATIONS', 6.53),
    ('CONSULTANTS', 1.97),
    ('EDUCATION', 9.63),
    ('HEALTHCARE', 9.38),
    ('OTHER SERVICES', 6.79),
    ('PROFESSIONAL SERVICES', 8.22),
    ('RESTAURANTS/MOBILE FOOD SERVICES', 2.85),
    ('RETAIL', 5.80),
    ('TECHNOLOGY', 8.57),
    ('WHOLESALE/DISTRIBUTION', 5.61)
]

# Insert data into the category_rates table
cursor.executemany('''
    INSERT INTO category_rates (category, rate)
    VALUES (?, ?)
''', categories)

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Database and table created successfully, and data inserted.")
```

```text
Database and table created successfully, and data inserted.
```

# Define Required Pydantic Schemas

In this section, we define several schemas using Pydantic, a Python library for data validation. 

These schemas serve as structured models for initial data collection, workflow routing, classification, rationale, and accuracy assessment in our contextual quoting agent. 

They ensure data consistency and accuracy throughout the quoting process.

```python
from pydantic import BaseModel, Field
from typing import Optional, List

# Core business information model for initial data collection
class BusinessInformation(BaseModel):
    legal_name: Optional[str] = Field(default="Info not yet provided", description="The legal name of the business")
    primary_products_or_services: Optional[str] = Field(default="Info not yet provided", description="The primary products or services offered")
    secondary_or_ancillary_operations: Optional[str] = Field(default="Info not yet provided", description="Any secondary or ancillary operations")
    industries: Optional[str] = Field(default="Info not yet provided", description="In which industries does the business operate?")
    manufacturing_retail_wholesale_or_distribution: Optional[str] = Field(default="Info not yet provided", 
        description="Does the business engage in manufacturing, retail, wholesale, or distribution?")
    projected_revenue: Optional[str] = Field(default="Info not yet provided", description="The projected revenue for the upcoming year")

# Models for workflow routing between AI assistants
class ToUnderwritingAssistant(BaseModel):
    """Transfer work to the underwriting assistant"""

    request: str = Field(
        description="Any information that the underwriting assistant should know before proceeding."
     )

    class Config:
        json_schema_extra = {
            "example": {
                "request": "The business is a landscaping company that offers residential and commercial landscaping services, including lawn maintenance, garden design, tree trimming, and irrigation system installation."
            }
        }

# Models for classification and code assignment
class FinalClassificationsInfo(BaseModel):
    """Information about the final classifications and code that best describe the business."""
    
    category: str = Field(
        description="A string representing the category."
    )
    classification_description: str = Field(
        description="A string representing the category description."
    )
    code: int = Field(
        description="An integer representing the code assigned to the category."
    )

class ToQuoteAssistant(BaseModel):
    """Transfer work to the underwriting assistant"""

    request: str = Field(
        description="Any information that the underwriting assistant should know before proceeding."
     )
    
    final_classifications: list[FinalClassificationsInfo] = Field(
        description="A list of FinalClassificationsInfo objects, each containing a category, category description, and code."
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "request": "The business is a landscaping company that offers residential and commercial landscaping services, including lawn maintenance, garden design, tree trimming, and irrigation system installation.",
                "final_classifications": [
                    {"category": "CONSULTANTS", "classification_description": "Consultants--Business Services", "code": 74379},
                    {"category": "RESTAURANTS/MOBILE FOOD SERVICES", "classification_description": "Bakeries–Fast Food–Donut Shop.", "code": 10100}
                ]
            }
        }

# Workflow control models
class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the workflow to the main assistant,
    who can re-route the dialog based on the user's needs."""

    cancel: bool = True
    reason: str

    class Config:
        json_schema_extra = {
            "example": {
                "cancel": True,
                "reason": "Successfully obtained a quote.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the user's emails or calendar for more information.",
            },
        }

# Models for classification reasoning and rationale
class CategoryRationale(BaseModel):
    """Represents a category and its rationale or reasoning."""
    category: str = Field(..., description="The name of the category")
    cat_description: str = Field(..., description="The description of the category")
    code: int = Field(..., description="The unique code for the category")
    rate: float = Field(..., description="The rate for the category")
    rationalization: str = Field(..., description="A detailed rationalization for the category")

class CategoriesOutput(BaseModel):
    """Output structure for a list of categories with rationale explanations."""
    categories: List[CategoryRationale] = Field(..., description="List of categories with their rationale explanations")

# Models for classification accuracy assessment
class DescriptionGrade(BaseModel):
    """Schema to define a business description and its corresponding grade.
    
    Attributes:
        description (str): A string representing the business description.
        grade (float): A float representing the grade assigned to the business description.
    """
    category: str = Field(
        description="A string representing the category."
    )
    cat_description: str = Field(
        description="A string representing the category description."
    )
    code: int = Field(
        description="An integer representing the code assigned to the category."
    )
    grade: float = Field(
        description="A float representing the grade assigned to the business description."
    )

class DescriptionGradeSchema(BaseModel):
    """Schema to hold a list of business descriptions with their corresponding grades.
    
    Attributes:
        description_grades (list[DescriptionGrade]): A list of DescriptionGrade objects, each containing a business description and its corresponding grade.
    """
    description_grades: list[DescriptionGrade] = Field(
        description="A list of DescriptionGrade objects, each containing a business description and its corresponding grade."
    )
```

## Define State

In this section, we define the **State** classes which represent the core data structures for managing and storing various types of information during the workflow. 

These state definitions ensure that different parts of the quoting process stay synchronized and properly updated:

- **State Reducer**: Implements `update_workflow_state`, a function that manages workflow transitions, preventing duplicate entries and handling "push" or "pop" operations.
- **MainState**: Defines the main working state, which includes messages, business information, and workflow state. It maintains critical details about the workflow's progression.
- **RAGState**: Represents the state for retrieval-augmented generation (RAG), storing descriptions, documents, rationale, and graded categories to assist with generating and validating responses.
- **ExtraState**: Holds additional data, such as tool call identifiers and final classifications, to supplement the workflow process.

These states ensure data integrity across all interactions of the contextual quoting agent.

```python
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages

# State Reducer
def update_workflow_state(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the workflow state, preventing duplicates."""
    if right is None or right == []:
        return left

    # Handle 'pop' case to remove the last element
    if right == "pop":  
        return left[:-1] if left else left  # Safeguard against empty lists
    
    # Handle cases where 'right' is a list (from the tutorial)
    if isinstance(right, list):
        # Avoid adding elements already in the list
        return left + [item for item in right if item not in left]

    # If 'right' is a single string, avoid duplication
    if right not in left:
        return left + [right]

    return left 

# Main State
class MainState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    business_information: dict[str, str | float | int]  # Allow values to be str, float, or int
    workflow_state: Annotated[
        list[
            Literal[
                "main_assistant",
                "underwriting_assistant",
                "quote_assistant",
            ]
        ],
        update_workflow_state
    ]

# RAG State
class RAGState(TypedDict):
    description: str
    documents: list[str]
    rationale: str
    graded_categories: list[dict[str, float]​]

# Extra State
class ExtraState(TypedDict):
    tool_call_id: str
    final_classifications: dict
```

# Define Tools

In this section, we define **Tools** that are crucial for enabling specific functions in the contextual quoting agent. Here is a brief explanation:

- **Handle Tool Errors**: The function `handle_tool_error()` is designed to manage errors that occur during tool execution. It takes the state, identifies the error, and generates messages indicating what went wrong, allowing users to address and correct the mistakes.

- **Create Tool Node with Fallback**: The function `create_tool_node_with_fallback()` creates a tool node using **ToolNode**, and attaches a fallback mechanism using `RunnableLambda` to handle errors gracefully. This ensures that if a tool encounters an error, it defaults to an error handler that provides useful feedback.

These definitions provide robustness to the workflow by managing tool errors effectively and ensuring smooth execution through fallback mechanisms.

```python
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import ToolMessage

# Handle tool errors
def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

# Create a tool node with fallback
def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )
```

## RAG

In this section, we define the **RAG (Retrieval-Augmented Generation)** process, which is used to load and prepare the necessary data for generating responses:

- **Load Categories**: We use `TextLoader` to load the category data from a CSV file (`cat_codes_descriptions.csv`). This data contains information about different categories, which is crucial for classification tasks within the quoting agent.

- **Split Categories**: We then use `RecursiveCharacterTextSplitter` to split the loaded text into manageable chunks. This splitter divides the text into smaller, non-overlapping chunks based on line breaks, ensuring that each chunk contains a distinct line of data.

These steps prepare the data for efficient retrieval, allowing the agent to use relevant pieces of information effectively during the response generation process.

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load Categories
categories_path = "data/cat_codes_descriptions.csv"

loader = TextLoader(file_path=categories_path)

data_txt = loader.load()

# Split Categories
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,  # Smaller chunk size to ensure one line per chunk
        chunk_overlap=0,  # No overlap needed for line-by-line splitting
        separators=["\n"],  # Primary separator is line breaks
        length_function=len,  # Length function based on character count
    )

splits = text_splitter.split_documents(data_txt)
```

## VectorStore Retriever

In this section, we define the **VectorStore Retriever**, which is used for efficient information retrieval in the quoting agent:

- **VectorStore Creation**: We use **Chroma** to create a vector store from the document chunks (`splits`). The vector store acts as a database where the documents are embedded and stored. The **OpenAIEmbeddings** class is used to convert the text chunks into vector representations, making them suitable for similarity-based searches.

- **Retriever Setup**: We create a retriever from the vector store using `as_retriever()`, specifying that it should retrieve the top `k=10` most relevant results. This allows the agent to efficiently find related information when handling user queries.

This setup enhances the agent's ability to provide accurate, context-aware responses by quickly accessing relevant documents from the vector store.

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Add to vectorDB
vectorstore = Chroma.from_documents(documents=splits,
                                    collection_name="rag-chroma",
                                    embedding=OpenAIEmbeddings()
                                )
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
```

## Business Info Gathering Tool

In this section, we define the **Business Info Gathering Tool**, which is used to collect and process key business details:

- **Business Info Gathering Tool**: The `info_tool` is defined using the `@tool` decorator from **LangChain Core Tools**. This tool takes an instance of `BusinessInformation` as input and processes it to update the state (`MainState`).

This tool is essential for capturing and integrating business data into the workflow, ensuring that subsequent steps have access to accurate and complete business information.

```python
from langchain_core.tools import tool

# Business Info Gathering Tool
@tool
def info_tool(business_info: BusinessInformation) -> MainState:
    """
    A tool to process and store business information.
    """
    result = {
        "business_information": business_info.model_dump()
    }
    return result
```

## Quote Tool

In this section, we define the **Quote Tool**, which calculates the premium for given categories based on projected revenue:

- **Quote Tool Function**: The `fetch_rates` function is decorated with `@tool`, making it available as a tool in the workflow. It takes two arguments:
  - `categories`: A list of categories for which the rates are fetched.
  - `projected_revenue`: The projected revenue to be used in calculating the premiums.

- **Premium Calculation**: 
  - The function connects to the SQLite database (`categories.db`) to retrieve rates for each provided category.
  - It calculates each category's share of the projected revenue and then determines the computed premium based on the rate per $1,000 of revenue.
  - The total premium is accumulated across all categories.

This tool enables the quoting agent to quickly and accurately provide detailed insurance premium quotes for various business categories, making it a vital component of the system's quoting functionality.

```python
# Quote Tool
@tool
def fetch_rates(categories: list[str], projected_revenue: float) -> str:
    """
    Fetches the rates for given categories and calculates the computed results based on the projected revenue.

    Args:
        categories (list[str]): A list of category names for which rates need to be fetched. Example categories include:
            - CATEGORY
            - ADMINISTRATIVE
            - ADVERTISING
            - ARTISAN CONTRACTORS
            - CLUBS/ASSOCIATIONS
            - CONSULTANTS
            - EDUCATION
            - HEALTHCARE
            - OTHER SERVICES
            - PROFESSIONAL SERVICES
            - RESTAURANTS/MOBILE FOOD SERVICES
            - RETAIL
            - TECHNOLOGY
            - WHOLESALE/DISTRIBUTION
        projected_revenue (float): The projected revenue to be distributed among the categories.

    Returns:
        str: A formatted message containing the quote breakdown
    """
    # Connect to SQLite database
    conn = sqlite3.connect('data/categories.db')
    cursor = conn.cursor()

    # Prepare parts to build the formatted message
    message_parts = ["Here is the requested quote:\n", "Quote Breakdown:"]

    # Initialize total premium
    total_premium = 0

    # Fetch rate for each classification
    for classification in categories:
        cursor.execute('''
            SELECT rate FROM category_rates WHERE category = ?
        ''', (classification,))
        result = cursor.fetchone()
        if result:
            rate = result[0]
            portion_revenue = projected_revenue / len(categories)
            computed_result = (portion_revenue / 1000) * rate
            total_premium += computed_result  # Accumulate total premium

            # Add formatted message
            message_parts.append(
                f"\nCategory: {classification}"
                f"\n  - Rate: ${rate:.2f} per $1,000"
                f"\n  - Portion of Projected Revenue: ${portion_revenue:,.2f}"
                f"\n  - Computed Premium: ${computed_result:,.2f}"
            )
        else:
            message_parts.append(f"\nWarning: No rate found for category: {classification}")

    # Add total premium to the message
    message_parts.append(f"\n\nTotal Annual Premium: ${total_premium:,.2f}")

    # Close the connection
    conn.close()

    return "\n".join(message_parts)
```

# Assistants

### Helper function

In this section, we define a **Helper Function** that manages interactions with the LLM to streamline assistant behavior:

- **Assistant Class**: The `Assistant` class is a helper designed to handle calls to the LLM (Language Learning Model). It is initialized with a `Runnable` object, which represents a configured LLM or an operation that can be executed.

This helper function plays a crucial role in maintaining smooth and consistent interactions between the quoting agent and the LLM, handling any cases where the model might fail to produce a satisfactory response initially.

```python
from langchain_core.runnables import Runnable, RunnableConfig

# Helper Assistant Class for handling LLM calls
class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable.with_config()
        
    def __call__(self, state: MainState, config: RunnableConfig):
        # Create input with both messages and business_information
        input_state = {
            "messages": state["messages"],
            "business_information": state.get("business_information", None),
        }
        
        while True:
            result = self.runnable.invoke(input_state)
            # If the LLM happens to return an empty response, we will re-prompt it
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
                input_state["messages"] = messages
            else:
                break
        return {"messages": result}
```

## Main Assistant

In this section, we define the **Main Assistant** of our contextual quoting system. This assistant guides the user through the quoting process, ensuring the gathered data is complete and consistent:

- **Main Assistant Prompt**: 
  - The `main_assistant_prompt` uses `ChatPromptTemplate.from_messages()` to define the conversation flow.
  - It defines the assistant's role as a **commercial underwriter assistant**, responsible for guiding the user through the quoting process.
  - The assistant's main tasks include:
    - Using the `"info_tool"` to process and update the business information as the user provides it.
    - Evaluating the collected information with underwriters via `"ToUnderwritingAssistant"` before proceeding with the quote.
    - Providing a summary of the gathered information to the user before moving forward.
    - Ensuring only **one tool** is called at a time for better control of actions.

- **Main Assistant Tools**: The tools available to the `main_assistant` are defined in `main_assistant_tool`, starting with `[info_tool]`. This ensures the assistant can gather and update business information during the conversation.

- **Runnable**:
  - `main_assistant_runnable` binds the `main_assistant_prompt` with the LLM (`llm`) using the tools available (`main_assistant_tool + [ToUnderwritingAssistant]`).
  - This makes the assistant capable of managing user information collection, calling underwriters, and then proceeding with further steps as needed.

This setup allows the **Main Assistant** to effectively manage the quoting workflow, guiding the user while ensuring that the provided information is properly processed, evaluated, and summarized before the next step is taken.

```python
# Main Assistant Prompt
main_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
        """
You are the main commercial underwriter assistant.\n
You are responsible for guiding the user through the quoting process and ensuring that the information they provide is accurate and consistent, to later provide a quote. 
When details and/or information about the business are provided, use "info_tool" to process the information and gather or update the required data.\n\n
Once you have gathered the information, you must first evaluate the prospects information with the underwriters by calling "ToUnderwritingAssistant".\n
Before proceeding to quote, you must provide the user with a summary of the information they have provided.\n
**Only call one tool at a time.**\n
Current business information:\n
{business_information}
""",
         ),
        ("placeholder", "{messages}"),
    ]
)

# Main Assistant Tools
main_assistant_tool = [info_tool]

# Runnable
main_assistant_runnable = main_assistant_prompt | llm.bind_tools(main_assistant_tool + [ToUnderwritingAssistant])
```

## Underwriting Assistant

In this section, we define the **Underwriting Assistant**, which is tasked with evaluating the prospect's information and guiding the quoting process after the initial information collection.

- **Underwriting Assistant Prompt**:
  - The assistant's role is as a **main commercial underwriter**, where its responsibilities include:
    - Evaluating the business information provided by the user as the **principal underwriter**.
    - Ensuring accuracy in gathered information and confirming relevant business categories.
    - Once all the required information has been gathered and validated, the assistant calls `"ToQuoteAssistant"` to proceed with the quoting process.

The **Underwriting Assistant** plays a critical role in the quoting workflow, acting as a checkpoint to ensure all required information is accurate and sufficient before proceeding to the actual quote generation. This assistant ensures that no important data is missed, maintaining quality and reliability throughout the quoting process.

```python
# Underwriting Assistant Prompt
underwriting_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
        """
You are the main commercial underwriter assistant.\n
You are responsible for evaluating the prospects information as he principal underwriter.\n
**Only call one tool at a time.**\n
When all the required information and categories have been confirmed, call "ToQuoteAssistant" to proceed to quote. 
Current business information:\n
{business_information}
""",
         ),
        ("placeholder", "{messages}"),
    ]
)

# Runnable
underwriting_assistant_runnable = underwriting_assistant_prompt | llm.bind_tools([ToQuoteAssistant])
```

## Quote Assistant

In this section, we define the **Quote Assistant**, which is responsible for calculating the insurance premium for a business based on classifications and providing a clear summary to the user.

- **Quote Assistant Prompt**:
  - The role of the **Quote Assistant** is to act as an expert in underwriting and to generate the **annual premium** based on the business classifications provided.
  - Key responsibilities and guidelines for the Quote Assistant include:
    1. **Calling `fetch_rates` only once**: It fetches rates for all required categories using the projected revenue.
    2. **Finalizing with `CompleteOrEscalate`**: After receiving the rates and calculating the premium, the assistant must call `CompleteOrEscalate` to end the process.
    3. **No Recalculation unless instructed**: The assistant should not call `fetch_rates` again unless explicitly asked by the user to do so. 
    ***(This is to avoid calling the tool multiple times)***
  - The assistant also ensures that a clear and understandable **summary** of the quote is provided to the user before concluding.

- **Quote Assistant Tools**: 
  - The tools available to the quote assistant are defined in `quote_assistant_tools`, which includes `[fetch_rates]`. This enables the assistant to fetch rates for given categories and calculate the required premium.

The **Quote Assistant** plays a vital role in completing the underwriting and quoting process. By fetching rates, calculating premiums, and providing a summary to the user, it ensures that the entire workflow is concluded seamlessly with all necessary information clearly communicated to the user.

```python
# Quote Assistant Prompt
quote_prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
You are an expert commercial underwriting assistant tasked with providing a quote for the business based on the classifications.

You will use the provided classifications and their corresponding rates to calculate the annual premium.

IMPORTANT: 
1. Call fetch_rates ONLY ONCE with all required categories and the projected revenue.
2. After receiving the rates and calculating the premium, call CompleteOrEscalate to end the process.
3. Do not call fetch_rates again unless explicitly asked to recalculate.

Business Description:
{business_information}

When you have the quote results, provide a clear summary to the user and then call CompleteOrEscalate with cancel=true to complete the process.
""",
         ),
        ("placeholder", "{messages}"),
    ]
)

# Quote Assistant Tools
quote_assistant_tools = [fetch_rates]

# Runnable
quote_assistant_runnable = quote_prompt | llm.bind_tools(quote_assistant_tools + [CompleteOrEscalate])
```

# Agents

## Rewriting Agent

In this section, we define the **Rewriting Agent**, which is responsible for extracting and summarizing key operational details from business descriptions to support the insurance underwriting process.

**Different from Assistants, Agents are used to process information and return structured outputs.**
**They do not interact with the user.**

- **Retriever Assistant Prompt**:
  - The assistant's role is to analyze the given **business description** and extract important operational details relevant to insurance underwriting.
  - Key tasks include:
    1. **Analyze** the provided business description.
    2. **Identify and Summarize** key aspects like workflows, operational tasks, and products/services offered.
    3. **Highlight** specific activities that are critical for matching the business to the appropriate **code classification**.
  - The summary should be **concise** and **structured** with bullet points for easy readability, as shown in the provided example.

The **Rewriting Agent** helps ensure that business descriptions are processed into meaningful insights that can assist in determining appropriate insurance classifications. This agent makes the underwriting process more streamlined by providing clear and actionable information derived from user-provided data.

```python
from langchain_core.output_parsers import StrOutputParser

# Retriever Assistant Prompt
retriever_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert commercial underwriting assistant specialized in analyzing business descriptions to extract key operational details relevant to the insurance underwriting process.

**Your task is to:**

1. **Analyze** the provided business description.
2. **Identify and summarize** the key aspects of the business operations, focusing on workflows, operational tasks, products/services offered, and any other relevant details.
3. **Highlight** any specific activities that are crucial for matching the business to the appropriate insurance code classification.

**Please provide the summary in a clear, concise, and structured format, using bullet points for each key aspect.**

---

**Business Information:**

{business_information}

---

**Example:**

**Business Description:**

"**ABC Landscaping** offers residential and commercial landscaping services, including lawn maintenance, garden design, tree trimming, and irrigation system installation. They also provide seasonal services like snow removal. The company operates a fleet of service vehicles and heavy machinery such as lawn mowers, trimmers, and excavators."

**Summary:**

- Provides residential and commercial landscaping services.
- Services include lawn maintenance, garden design, and tree trimming.
- Offers installation of irrigation systems.
- Provides seasonal snow removal services.
- Operates a fleet of service vehicles and heavy machinery (lawn mowers, trimmers, excavators).

""",
        )
    ]
)

# Runnable
retriever_runnable = retriever_prompt | llm | StrOutputParser()
```

## Reasoning Agent

In this section, we define the **Reasoning Agent**, which is tasked with determining the relevance of different insurance categories for a business based on its description.

- **Reasoning Categories Assistant Prompt**:
  - The role of the **Reasoning Agent** is to evaluate the **business description** and determine which categories apply to the nature of the operations.
  - The agent uses **logical reasoning** to assess which categories from the given list are relevant.
  - The assistant receives both the **business description** and a list of possible **categories** and provides a reasoned analysis of which ones are applicable.
  - The response includes not just the category, but also a **rationalization**—a brief explanation of why a particular category was chosen based on the business operations. The provided example demonstrates how to present the selected categories with reasoning.

The **Reasoning Agent** is critical for ensuring that the business operations are properly categorized with a logical and justifiable approach. By providing reasoned categorization, it helps ensure accuracy and compliance in matching businesses to the right insurance classification, which is key for accurate underwriting.

```python
# Reasoning Categories Assistant Prompt
reasoning_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
        """
You are an expert commercial underwriting assistant tasked with providing reasoning for the categories of the nature of operations of a business.\n
You will be given a description of the business and a list of categories that may apply to the business.\n
Use logical reasoning to determine which categories are relevant to the business based on the provided description.\n
Business Description:\n
{description}\n
Categories:\n
{documents}\n

---

**Example:**


{{
    "categories": [
      {{
        "category": "CONSULTANTS",
        "description": "Consultants--Business Services"
        "code": "74379",
        "rate": "1.97",
        "rationalization": "GABAS, LLC provides consulting services related to AI and technology, which falls under the category of business services. This includes advising businesses on AI implementation and strategy."
      }},
      {{
        "category": "RESTAURANTS/MOBILE FOOD SERVICES",
        "description": "Bakeries–Fast Food–Donut Shop."
        "code": "10100",
        "rate": "2.85",
        "rationalization": "A bakery that offers fast food and donut services is considered part of the food service industry."
      }}
    ]
}}

""",
         )
    ]
)

# Runnable
reasoning_runnable = reasoning_prompt | llm.with_structured_output(CategoriesOutput)
```

## Classification Grading Agent

In this section, we define the **Classification Grading Agent**, which evaluates the accuracy of category assignments for a business based on its description.

- **Classification Grading Assistant Prompt**:
  - The **Classification Grading Agent** is responsible for assessing the match between the **business description** and the **assigned categories**.
  - The agent uses logical reasoning to **grade the accuracy** of the assigned categories to ensure they are the most relevant and appropriate for the business operations.
  - The prompt includes:
    - A **business description**.
    - The **category descriptions and rationalizations** (`{cat_and_rat}`) to provide context for the grading process.
  - The goal is to ensure that each classification is graded based on how well it aligns with the details of the business.

- **Runnable**:
  - `classification_grading_runnable` binds the `classification_grading_prompt` with the LLM (`llm`) and uses `.with_structured_output(DescriptionGradeSchema)` to ensure that the output is returned in a structured format.
  - The **DescriptionGradeSchema** includes details about the category, its description, and the assigned grade, helping evaluate the fit between the business operations and the assigned categories.

The **Classification Grading Agent** is an important step in the underwriting process, helping validate the quality of the category assignments. It ensures that each classification matches the actual operations of the business, which is key for accurate risk assessment and premium calculation.

```python
# Classification Grading Assistant Prompt
classification_grading_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
        """
You are an expert commercial underwriting assistant tasked with grading the accuracy between a given business description and the classifications assigned.\n
Apply logical reasoning, grade the accuracy between the business description and the categories assigned.\n
The objective is to match the business description with the most relevant categories.\n
Business Description:\n
{description}\n
Category Description and Rationalization:\n
{cat_and_rat}
""",
         )
    ]
)

# Runnable
classification_grading_runnable = classification_grading_prompt | llm.with_structured_output(DescriptionGradeSchema)
```

# Define Nodes

## Retriever Node

We define a **Retriever Node** named `reasoning`, which is responsible for providing logical reasoning to identify relevant categories for a business based on its description and supporting documents.

- **Reasoning Node Function**:
  - The function `reasoning(rag_state: RAGState) -> RAGState` takes an `RAGState` as input, which includes the business **description** and the **retrieved documents** (likely possible categories).
  - **Reasoning Process**:
    - The function uses the **Reasoning Agent** (`reasoning_runnable`) to generate a rationale based on the provided description and documents.
    - The `invoke()` method is called with a dictionary containing the **description** and **documents**, which the reasoning agent processes to provide the rationale.
  - The function then returns an updated `RAGState`, which now includes the reasoned **categories** of the business, stored in `"rationale"`.

The **Retriever Node** adds depth to the workflow by analyzing the business description alongside related documents to provide detailed reasoning behind category choices. This helps ensure that all classifications assigned to the business are thoroughly validated, contributing to a more accurate and justified underwriting process.

```python
# Retrieve Node
def retrieve(state: MainState) -> RAGState:
    """
    The function will first generate a detailed description of the business and then retrieve relevant documents.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---Rewrite Description---")
    nature_of_ops = state["business_information"]
    description = retriever_runnable.invoke({"business_information": nature_of_ops})
    print(f"Description:\n {description}")

    print("---RETRIEVE---")
    # Retrieval
    documents = retriever.invoke(description)
    print(f"Documents:\n {documents}")
    return {"description": description, "documents": documents}
```

## Reasoning Node

We define the **Reasoning Node**, which is responsible for analyzing the business description and providing a reasoned rationale for the assigned categories.

- **Reasoning Node Function**:
  - The function `reasoning(rag_state: RAGState) -> RAGState` is designed to process the **RAGState** containing:
    - The **description** of the business.
    - The **retrieved documents**, which typically represent possible categories for classification.
  - **Reasoning Process**:
    - The function calls the **Reasoning Agent** (`reasoning_runnable`) using the `invoke()` method, passing in the `description` and `documents` to generate the **rationale**.
    - This rationale explains why certain categories are relevant to the business, providing a more thorough understanding of the classification logic.
  - **Return Value**: 
    - The function returns an updated `RAGState`, including the reasoned **rationale** for the selected categories (`"rationale": rationale`).

The **Reasoning Node** is crucial for adding transparency and justification to the underwriting process. By providing a detailed explanation of why certain categories are assigned, it ensures that the classification is not only accurate but also understandable, making it easier for users or underwriters to validate and trust the results.

```python
# Reasoning Node
def reasoning(rag_state: RAGState) -> RAGState:
    """
    The function provides reasoning for the business categories based on the provided description and retrieved documents.

    Args:
        rag_state (RAGState): The current state containing the description and retrieved documents.

    Returns:
        RAGState: Updated state with reasoned categories of the business.
    """
    print("---REASONING---")
    description = rag_state["description"]
    documents = rag_state["documents"]

    # Reasoning
    rationale = reasoning_runnable.invoke({"description": description, "documents": documents})
    print(f"Rationale:\n {rationale}")
    return {"rationale": rationale}
```

## Classification Grading Node

We define the **Classification Grading Node**, which evaluates how well each business category matches the provided business description.

- **Classification Grading Node Function**:
  - The `classification_grading(rag_state: RAGState) -> ExtraState` function is designed to **grade** each of the business categories provided in the `RAGState`.
  - **Core Steps**:
    1. **Extract Core Data**:
       - Extracts the **description** of the business, **rationale**, and the **categories** from the provided `rag_state`.
       - Prints the number of categories being processed for grading.
    2. **Formatting for Grading**:
       - Formats each category's information (category name, description, code, and rationalization) into a readable string called `cat_and_rat`.
       - This formatted string is used as input to the **Classification Grading Agent** (`classification_grading_runnable`) to evaluate the categories.
    3. **Grading Categories**:
       - Calls `classification_grading_runnable.invoke()` with the business **description** and the formatted `cat_and_rat` to grade the accuracy of the assigned categories.
       - The output (`graded_categories`) includes a graded evaluation of each category.
  - **Return Value**:
    - The function returns an `ExtraState` that includes the **graded categories** (`"graded_categories": graded_categories.description_grades`).

The **Classification Grading Node** is essential for ensuring that the assigned categories accurately reflect the business operations. It helps validate and refine the classification process, providing a clear and justifiable grading that can support underwriting decisions.

```python
# Classification Grading Node
def classification_grading(rag_state: RAGState) -> ExtraState:
    """
    Grade each category based on how well it matches the business description.
    """
    print("\n=== CLASSIFICATION GRADING START ===")
    
    # Extract core data
    description = rag_state["description"]
    rationale = rag_state["rationale"]
    categories = rationale.categories
    
    print(f"Processing {len(categories)} categories for grading...")
    
    try:
        # Format categories into a single string
        cat_and_rat = "\n\n".join([
            f"Category: {cat.category}\n"
            f"Description: {cat.cat_description}\n"
            f"Code: {cat.code}\n"
            f"Rationale: {cat.rationalization}\n"
            for cat in categories
        ])
        
        graded_categories = classification_grading_runnable.invoke({
            "description": description,
            "cat_and_rat": cat_and_rat
        })
        
        print(f"\nCompleted grading {len(graded_categories.description_grades)} categories")
        
    except Exception as e:
        print(f"Error during classification grading: {str(e)}")
        raise
    
    print("=== CLASSIFICATION GRADING END ===\n")
    return {"graded_categories": graded_categories.description_grades}
```

### Routing Nodes

We define the **Routing Nodes**, which are responsible for determining the flow of conversation between different assistants within the system. These routing nodes help manage transitions across various stages of the workflow to ensure a seamless process.

1. **Route Main Assistant (`route_main_assistant`)**:
   - Determines the next step for the **main assistant**.
   - Depending on the last tool call in the state, the route directs either to continue with tools (`"main_assistant_tools"`), retrieve more information (`"retrieve"`), or end the session (`"__end__"`).
   - Routes to `"pass_tool_call_id"` if the `ToUnderwritingAssistant` is called, otherwise `"main_assistant_tools"`.

```python
from langgraph.prebuilt import tools_condition
from typing import Literal, Callable

# Function to determine the next route for the main assistant
def route_main_assistant(
    state: MainState,
) -> Literal[
    "main_assistant_tools",
    "retrieve",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToUnderwritingAssistant.__name__:
            return "pass_tool_call_id"
        else:
            return "main_assistant_tools"
    raise ValueError("Invalid route")
```

2. **Route Underwriting Assistant (`route_underwriting_assistant`)**:
   - Manages the flow for the **underwriting assistant**.
   - If the `ToQuoteAssistant` tool is called, it routes to `"update_workflow_state"`, allowing progression to the next step in the quoting process.

```python
# Function to determine the next route for the underwriting assistant
def route_underwriting_assistant(
    state: MainState,
) -> Literal[

    "update_workflow_state",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls

    if tool_calls:
        if tool_calls[0]["name"] == ToQuoteAssistant.__name__:
            return "update_workflow_state"

    raise ValueError("Invalid route")
```

3. **Route Quote Assistant (`route_quote_assistant`)**:
   - Determines the routing for the **quote assistant**.
   - If the conversation is complete and `"CompleteOrEscalate"` is called, it reroutes (`"reroute"`).
   - Otherwise, if there are tools to be used, it directs the flow to `"quote_assistant_tools"`.

```python
# Function to determine the next route for the quote assistant
def route_quote_assistant(
    state: MainState,
) -> Literal[
    "quote_assistant_tools",
    "reroute",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END

    tool_calls = state["messages"][-1].tool_calls
    
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "reroute"
    
    # Get tool names using the name property for StructuredTools
    tool_names = [t.name for t in quote_assistant_tools]
    
    if all(tc["name"] in tool_names for tc in tool_calls):
        return "quote_assistant_tools"
        
    raise ValueError(f"Invalid route. Tool call {[tc['name'] for tc in tool_calls]} not found in available tools {tool_names}")
```

4. **Create Entry Node for Quote Conversation (`create_entry_node_quote`)**:
   - Creates an **entry node** for a new quote.
   - Updates the state with the `"final_classifications"` and provides a message to initiate the quoting process with the selected assistant.

```python
# Function to create an entry node for quote conversation
def create_entry_node_quote(assistant_name: str, new_workflow_state: str) -> Callable:
    def entry_node(state: MainState):
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        final_classifications = state["messages"][-1].tool_calls[0]["args"]["final_classifications"]
        business_information = state.get("business_information", {})  # Changed from [] to {}
        
        # Create a new dictionary combining both pieces of information
        updated_business_info = {
            **business_information,
            "final_classifications": final_classifications
        }
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now {assistant_name}. You will proceed to provide a quote using the final classifications related to the business in evaluation."
                    f" Final Classifications/Categories: {final_classifications}."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ],
            "business_information": updated_business_info,
            "workflow_state": new_workflow_state
        }

    return entry_node
```

5. **Route to Workflow (`route_to_workflow`)**:
   - Determines which assistant to route to based on the current `"workflow_state"`.
   - Routes back to the **main assistant** if no specific workflow state is set.

```python
# Function to return to the workflow
def route_to_workflow(
    state: MainState,
) -> Literal[
    "main_assistant",
    "underwriting_assistant",
    "quote_assistant",
]:
    """If we are in a workflow state, route directly to the appropriate assistant."""
    workflow_state = state.get("workflow_state")
    if not workflow_state:
        return "main_assistant"
    return workflow_state
```

6. **Reroute Workflow to Main Assistant (`reroute`)**:
   - Reroutes the workflow to the **main assistant**, resuming dialog and popping the current workflow state.

```python
# Function to reroute the workflow to the main assistant
def reroute(state: MainState) -> dict:
    """Reroute the workflow to the main assistant."""
    messages = []
    if state["messages"][-1].tool_calls:
        messages.append(
            ToolMessage(
                content="Resuming dialog with the main assistant.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {
        "workflow_state": "pop",
        "messages": messages,
    }
```

7. **Update Workflow State (`update_workflow_state`)**:
   - Updates the workflow state by popping the last entry to ensure the correct assistant resumes the conversation.

```python
# Function to update the workflow state
def update_workflow_state(state: MainState) -> MainState:
    """Reroute the workflow to the main assistant."""
    return {
        "workflow_state": "pop",
    }
```

8. **Pass Tool Call ID to ExtraState (`pass_tool_call_id`)**:
   - Transfers the **tool call ID** from the state to `ExtraState` for later use.

```python
# Function to pass the tool call ID to ExtraState
def pass_tool_call_id(state: MainState) -> ExtraState:
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    return {"tool_call_id": tool_call_id}
```

9. **Pass Final Classifications (`pass_final_classifications`)**:
   - Transfers the **final classifications** from `RAGState` to `ExtraState`.

```python
# Function to pass the final classifications to ExtraState
def pass_final_classifications(rag_state: RAGState) -> ExtraState:
    final_classifications = rag_state["graded_categories"]
    return {"final_classifications": final_classifications}
```

10. **Create Tool Message (`create_tool_message`)**:
    - Generates a **ToolMessage** for the main underwriting assistant.
    - It includes information about the identified categories and instructs the assistant to confirm their relevance through follow-up questions without sharing the grades with the user.

```python
# Function to create a tool message and pass to MainState
def create_tool_message(extra_state: ExtraState) -> MainState:
    tool_call_id = extra_state["tool_call_id"]
    final_classifications = extra_state["final_classifications"]
    return {
            "messages": [
                ToolMessage(
                    content=""f"Assume the role of the Main Underwriting Assistant and reflect on the previous conversation between the host assistant and the user."
                    f" Based on the initial underwriting analysis, the following categories have been identified as most relevant to the business:"
                    f"{final_classifications}"
                    "Grades have been assigned based on the accuracy of the descriptions and the rationale provided for each category. KEEP THE GRADES PRIVATE AND DO NOT SHARE THEM WITH THE USER."
                    "To confirm their relevance to the business, please ask the user follow-up questions to finalize the validity of these categories.""",
                    tool_call_id=tool_call_id,
                )
            ],
            "workflow_state": "underwriting_assistant"
        }
```

11. **Update State (`update_state`)**:
    - Updates the main state with new business information retrieved from tool calls.
    - Handles potential JSON parsing issues to ensure that the state is updated consistently with accurate information.

```python
# Function to update the state with the new business information from tool calls
def update_state(state: MainState) -> MainState:
    """
    The function updates the state with the new business information from tool calls.
    """
    messages = state.get("messages", [])
    
    # Find the most recent tool message
    tool_messages = [msg for msg in messages if msg.type == "tool"]
    if tool_messages:
        latest_tool_message = tool_messages[-1]
        # Parse the JSON content from the tool message
        import json
        try:
            tool_content = json.loads(latest_tool_message.content)
            business_info = tool_content.get("business_information", {})
        except json.JSONDecodeError:
            business_info = state.get("business_information", {})
    else:
        business_info = state.get("business_information", {})

    return {
        "messages": messages,
        "business_information": business_info
    }
```

The **Routing Nodes** play a key role in managing the workflow's progression through different assistants and tools. They handle conditions such as tool usage, completion of processes, and transitions between assistants, ensuring each stage of the quoting and underwriting process flows smoothly and logically. These nodes make the system more modular and efficient by directing the appropriate assistant to continue based on the context and current state of the workflow.

## Define and Configure Graph

The Graph defines how different parts of the assistant workflow interconnect. Each node represents a specific step or role, and edges dictate the possible transitions between these steps. By adding conditions and sequential edges, the graph provides flexibility in how each assistant and tool works together, ensuring that the entire quoting process is both adaptable and logically structured.

This comprehensive setup allows for smooth transitions between data collection, underwriting evaluation, and premium quoting, while also handling possible errors or rerouting needs effectively.

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, END

# Graph State
graph = StateGraph(MainState)

## Add Nodes
# Main Assistant Node
graph.add_node("main_assistant", Assistant(main_assistant_runnable))
# Main Assistant Tools Node
graph.add_node("main_assistant_tools", create_tool_node_with_fallback(main_assistant_tool))
# Underwriting Assistant Node
graph.add_node("underwriting_assistant", Assistant(underwriting_assistant_runnable))
# Quote Assistant Node
graph.add_node("quote_assistant", Assistant(quote_assistant_runnable))
# Quote Assistant Tools Node
graph.add_node("quote_assistant_tools", create_tool_node_with_fallback(quote_assistant_tools))
# Entry Quote Assistant Node
graph.add_node(
    "entry_quote_assistant",
    create_entry_node_quote("Main Quote Assistant", "quote_assistant")
)
# Retrieve Node
graph.add_node("retrieve", retrieve)
# Reasoning Node
graph.add_node("reasoning", reasoning)
# Classification Grading Node
graph.add_node("classification_grading", classification_grading)
# Update State Node
graph.add_node("update_state", update_state)
# Reroute Node
graph.add_node("reroute", reroute)
# Pass Tool Call ID Node
graph.add_node("pass_tool_call_id", pass_tool_call_id)
# Pass Final Classifications Node
graph.add_node("pass_final_classifications", pass_final_classifications)
# Create Tool Message Node
graph.add_node("create_tool_message", create_tool_message)
# Update Workflow State Node
graph.add_node("update_workflow_state", update_workflow_state)

## Add Edges
# Set Conditional Entry Point
graph.set_conditional_entry_point(
    route_to_workflow,
    {
        "main_assistant": "main_assistant",
        "underwriting_assistant": "underwriting_assistant",
        "quote_assistant": "quote_assistant",
    }
)

# Conditional Edges from main assistant
graph.add_conditional_edges(
    "main_assistant",
    route_main_assistant,
    [
        "main_assistant_tools",
        "pass_tool_call_id",
        END,
    ],
)

# Conditional Edges from underwriting assistant
graph.add_conditional_edges(
    "underwriting_assistant",
    route_underwriting_assistant,
    [
        "reroute",
        "update_workflow_state",
        END,
    ],
)

# Conditional Edges from quote assistant
graph.add_conditional_edges(
    "quote_assistant",
    route_quote_assistant,
    [
        "quote_assistant_tools",
        "reroute",
        END,
    ],
)

# Edges from update workflow state to entry quote assistant
graph.add_edge("update_workflow_state", "entry_quote_assistant")
# Edges from pass tool call id to retrieve
graph.add_edge("pass_tool_call_id", "retrieve")
# Edges from retrieve to reasoning
graph.add_edge("retrieve", "reasoning")
# Edges from reasoning to classification grading
graph.add_edge("reasoning", "classification_grading")
# Edges from classification grading to pass final classifications
graph.add_edge("classification_grading", "pass_final_classifications")
# Edges from pass final classifications to create tool message  
graph.add_edge("pass_final_classifications", "create_tool_message")
# Edges from create tool message to underwriting assistant
graph.add_edge("create_tool_message", "underwriting_assistant")
# Edges from update state to main assistant
graph.add_edge("main_assistant_tools", "update_state")
# Edges from update state to main assistant
graph.add_edge("update_state", "main_assistant")
# Edges from reroute to main assistant
graph.add_edge("reroute", "main_assistant")
# Edges from entry quote assistant to quote assistant
graph.add_edge("entry_quote_assistant", "quote_assistant")
# Edges from quote assistant tools to quote assistant
graph.add_edge("quote_assistant_tools", "quote_assistant")

# Memory Saver
memory = MemorySaver()

# Compile Graph
final_graph = graph.compile(
    checkpointer=memory,
)
```

# Graph Visualization

![Final Graph](https://github.com/NirDiamant/GenAI_Agents/raw/bd681451b254ac1a790e947b581d3997ab35013d/all_agents_tutorials/../images/contextual_quoting_graph.svg)

## Test Quote Engine

```python
import uuid
from langchain.schema import AIMessage
from langchain_core.messages import ToolMessage 

# Function to print the event
def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print(f"Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "user_id": "1",
        "thread_id": thread_id,
    }
}

_printed = set()

# Function to process questions
def process_questions():

    global _printed  # Ensure the use of the global _printed variable
    _printed = set()  # Re-initialize if needed for the function's scope
    input_count = 0
    max_inputs = 20

    while input_count < max_inputs:
        # Get user input for the question
        question = input("\nEnter your question (or type '/exit' to quit): ")
        if question.lower() == '/exit':
            break

        # Process the question
        events = final_graph.stream(
            {"messages": ("user", question)}, config=config, stream_mode="values"
        )
        for event in events:
            _print_event(event, _printed)
        snapshot = final_graph.get_state(config)

        while snapshot.next:
            # We have an interrupt! The agent is trying to use a tool, and the user can approve or deny it
            user_input = input(
                "Do you approve of the above actions? Type 'y' to continue; "
                "otherwise, explain your requested changes.\n\n"
            )
            if user_input.strip().lower() == "y":
                # Just continue
                result = final_graph.invoke(
                    None,
                    config
                )
            else:
                # Satisfy the tool invocation by providing instructions on the requested changes / change of mind
                result = final_graph.invoke(
                    {
                        "messages": [
                            ToolMessage(  # Ensure ToolMessage is imported or defined
                                tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                                content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                            )
                        ]
                    },
                    config,
                )
                
            # Filter and print the last AIMessage
            aimessages = [message for message in result["messages"] if isinstance(message, AIMessage)]
            last_aimessage = aimessages[-1] if aimessages else None
            if last_aimessage is not None:
                # print(last_aimessage.content)
                _print_event(result, _printed)
                
            snapshot = final_graph.get_state(config)

        input_count += 1
        if input_count >= max_inputs:
            print("Maximum number of inputs reached. Terminating the loop.")
            break

process_questions()
```

```text
================================[1m Human Message [0m=================================

GABAS, LLC. We offer tailor made AI solutions. We also provide consulting services. We operate in the tech industry. As software providers, I believe we fall under the distribution type of business. We expect $4 million in revenues.
==================================[1m Ai Message [0m==================================
Tool Calls:
  info_tool (77aec3c3-9632-492d-8314-ff7f69f2b362)
 Call ID: 77aec3c3-9632-492d-8314-ff7f69f2b362
  Args:
    business_info: {'company_name': 'GABAS, LLC', 'description': ['tailor made AI solutions', 'consulting services'], 'industry': 'tech', 'business_type': 'distribution', 'revenue': 4000000}
=================================[1m Tool Message [0m=================================
Name: info_tool

Error: 1 validation error for info_tool
business_info
  Input should be a valid dictionary or instance of BusinessInformation [type=model_type, input_value="{'company_name': 'GABAS,...n', 'revenue': 4000000}", input_type=str]
    For further information visit https://errors.pydantic.dev/2.9/v/model_type
 Please fix your mistakes.
==================================[1m Ai Message [0m==================================
Tool Calls:
  info_tool (6fca9f47-e7f1-423a-a44c-d5bcbd6b0aa4)
 Call ID: 6fca9f47-e7f1-423a-a44c-d5bcbd6b0aa4
  Args:
    business_info: {'company_name': 'GABAS, LLC', 'description': ['tailor made AI solutions', 'consulting services'], 'industry': 'tech', 'business_type': 'software_provider', 'revenue': 4000000}
=================================[1m Tool Message [0m=================================
Name: info_tool

Error: 1 validation error for info_tool
business_info
  Input should be a valid dictionary or instance of BusinessInformation [type=model_type, input_value="{'company_name': 'GABAS,...r', 'revenue': 4000000}", input_type=str]
    For further information visit https://errors.pydantic.dev/2.9/v/model_type
 Please fix your mistakes.
==================================[1m Ai Message [0m==================================
Tool Calls:
  info_tool (ed7edf8b-e21c-4dcf-bc53-b9b7f1ce444c)
 Call ID: ed7edf8b-e21c-4dcf-bc53-b9b7f1ce444c
  Args:
    business_info: {'company_name': 'GABAS, LLC', 'description': ['tailor made AI solutions', 'consulting services'], 'industry': 'tech', 'business_type': 'software_provider', 'revenue': 4000000}
=================================[1m Tool Message [0m=================================
Name: info_tool

Error: 1 validation error for info_tool
business_info
  Input should be a valid dictionary or instance of BusinessInformation [type=model_type, input_value="{'company_name': 'GABAS,...r', 'revenue': 4000000}", input_type=str]
    For further information visit https://errors.pydantic.dev/2.9/v/model_type
 Please fix your mistakes.
==================================[1m Ai Message [0m==================================
Tool Calls:
  info_tool (48910bb5-f882-4cd9-ba6d-4cab7a11da7e)
 Call ID: 48910bb5-f882-4cd9-ba6d-4cab7a11da7e
  Args:
    business_info: {'company_name': 'GABAS, LLC', 'description': ['tailor made AI solutions', 'consulting services'], 'industry': 'tech', 'business_type': 'software_provider', 'revenue': 4000000}
=================================[1m Tool Message [0m=================================
Name: info_tool

Error: 1 validation error for info_tool
business_info
  Input should be a valid dictionary or instance of BusinessInformation [type=model_type, input_value="{'company_name': 'GABAS,...r', 'revenue': 4000000}", input_type=str]
    For further information visit https://errors.pydantic.dev/2.9/v/model_type
 Please fix your mistakes.
==================================[1m Ai Message [0m==================================

The business information provided is:

* Company Name: GABAS, LLC
* Description: Tailor made AI solutions and consulting services
* Industry: Tech
* Business Type: Software Provider
* Revenue: $4 million

Please wait while I evaluate this information with the underwriters... 

**ToUnderwritingAssistant**

Please provide a response to the business information provided.
================================[1m Human Message [0m=================================

Looks good. 
==================================[1m Ai Message [0m==================================
Tool Calls:
  info_tool (8a8ff5b5-0a70-41e6-91e0-d5bce6f4cb96)
 Call ID: 8a8ff5b5-0a70-41e6-91e0-d5bce6f4cb96
  Args:
    business_info: {'company_name': 'GABAS, LLC', 'description': ['tailor made AI solutions', 'consulting services'], 'industry': 'tech', 'business_type': 'software_provider', 'revenue': 4000000}
=================================[1m Tool Message [0m=================================
Name: info_tool

Error: 1 validation error for info_tool
business_info
  Input should be a valid dictionary or instance of BusinessInformation [type=model_type, input_value="{'company_name': 'GABAS,...r', 'revenue': 4000000}", input_type=str]
    For further information visit https://errors.pydantic.dev/2.9/v/model_type
 Please fix your mistakes.
==================================[1m Ai Message [0m==================================

Based on the business information provided, I have gathered the following details:

* Company Name: GABAS, LLC
* Description: Tailor made AI solutions and consulting services
* Industry: Tech
* Business Type: Software Provider
* Revenue: $4 million

Before proceeding to quote, would you like me to provide a summary of the information provided?
```

```python

```

![](https://europe-west1-genai-agents-views-tracker.cloudfunctions.net/genai-agents-tracker?notebook=all-agents-tutorials--contextual-quoting-agentic-system)
