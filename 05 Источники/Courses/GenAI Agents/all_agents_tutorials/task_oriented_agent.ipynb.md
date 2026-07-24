---
title: "Tutorial: Building a Text Summarizer and Translator with LangChain"
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: bd681451b254ac1a790e947b581d3997ab35013d
---

> [!note] Original source material
> This page preserves [`all_agents_tutorials/task_oriented_agent.ipynb`](https://github.com/NirDiamant/GenAI_Agents/blob/bd681451b254ac1a790e947b581d3997ab35013d/all_agents_tutorials/task_oriented_agent.ipynb) from
> *GenAI Agents* at commit `bd681451b254ac1a790e947b581d3997ab35013d`. License:
> [Custom non-commercial license](https://github.com/NirDiamant/GenAI_Agents/blob/bd681451b254ac1a790e947b581d3997ab35013d/LICENSE). Bookvar changed only
> publication markup, link paths, and characters required for safe rendering.



## Overview

This tutorial demonstrates how to create a language model application that summarizes text and translates the summary to Spanish using LangChain. The application uses a combination of custom functions, structured tools, and an agent to process input text efficiently.

## Motivation

In today's data-rich world, the ability to quickly summarize information and translate it into different languages is invaluable. This tutorial aims to show how to leverage language models and the LangChain framework to create a tool that can:

1. Summarize lengthy text
2. Translate the summary to Spanish
3. Do both tasks in a single, streamlined process

This type of tool can be useful for various applications, including content curation, multilingual communication, and rapid information processing.

## Key Components

1. **Custom Functions**: For summarization and translation
2. **Structured Tools**: Wrappers for the custom functions
3. **Prompt Template**: Instructions for the agent
4. **Agent**: Orchestrates the use of tools based on the prompt
5. **Agent Executor**: Runs the agent with specified parameters

## Method Details

### 1. Custom Functions

Two main functions are defined:

- A summarization function that takes input text and returns a summary
- A translation function that takes input text and returns its Spanish translation

Both functions use a PromptTemplate and a language model to perform their tasks.

### 2. Structured Tools

The custom functions are wrapped as StructuredTool objects. This allows the agent to use these functions as tools, providing a name, description, and input schema for each.

### 3. Prompt Template

A PromptTemplate is created with detailed instructions for the agent. It outlines the steps the agent should follow:
1. Summarize the input text
2. Translate the summary to Spanish
3. Format the output with both the English summary and Spanish translation

### 4. Agent and Agent Executor

An agent is created using the tools and prompt. This agent is then wrapped in an AgentExecutor, which manages the execution of the agent. The executor is configured with parameters such as the maximum number of iterations and the early stopping method.

### 5. Running the Agent

A helper function is created to simplify running the agent. This function takes the agent executor and a query, runs the agent, and returns the output. The tutorial demonstrates this with a sample query about pangrams, showing how the entire pipeline works together to process the input text.

## Conclusion

This tutorial demonstrates how to create a powerful text processing tool using LangChain. By combining custom functions, structured tools, and an agent, we've created an application that can summarize text and translate the summary to Spanish in one seamless operation. This approach can be extended to include other languages or text processing tasks, making it a versatile foundation for various natural language processing applications.

The strength of this approach lies in its modularity and flexibility. By using LangChain's components, we can easily modify or extend the functionality of our application. For instance, we could add more languages, implement different summarization techniques, or even incorporate other text processing tasks like sentiment analysis or keyword extraction.

### Import necessary libraries

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
```

### Load environment variables and initialize the language model

```python
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0)
```

### first let's define the functions that the agent can use:

```python
def summarize(text):
    # Create a PromptTemplate for summarization
    prompt = PromptTemplate(
        input_variables=["text"],  # Specify the input variable
        template="Summarize the following text:\n\n{text}\n\nSummary:"  # Define the template for summarization
    )
    chain = prompt | llm  # Create a chain by piping the prompt to the language model
    return chain.invoke({"text": text}).content  # Invoke the chain with the input text and return the content of the response

def translate(text):
    # Create a PromptTemplate for translation
    prompt = PromptTemplate(
        input_variables=["text"],  # Specify the input variable
        template="Translate the following text to Spanish:\n\n{text}\n\nTranslation:"  # Define the template for translation
    )
    chain = prompt | llm  # Create a chain by piping the prompt to the language model
    return chain.invoke({"text": text}).content  # Invoke the chain with the input text and return the content of the response

class TextInput(BaseModel):
    # Define a Pydantic model for input validation
    text: str = Field(description="The text to summarize or translate")  # Define a text field with a description
```

```python
# test the functions

text = "The quick brown fox jumps over the lazy dog."
print(summarize(text))
print(translate(text))
```

```text
A fast brown fox leaps over a sluggish dog.
La rápida zorra marrón salta sobre el perro perezoso.
```

### Define the tools for the agent

```python
tools = [
    StructuredTool.from_function(
        func=summarize,  # The function to be wrapped as a tool
        name="Summarize",  # Name of the tool
        description="Useful for summarizing text",  # Description of what the tool does
        args_schema=TextInput  # The Pydantic model defining the input schema
    ),
    StructuredTool.from_function(
        func=translate,  # The function to be wrapped as a tool
        name="Translate",  # Name of the tool
        description="Useful for translating text to Spanish",  # Description of what the tool does
        args_schema=TextInput  # The Pydantic model defining the input schema
    )
]
```

### Initialize the agent

```python
prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],  # Define the input variables for the prompt
    template="""Summarize the following text and then translate the summary to Spanish:

Text: {input}

Use the following steps:
1. Use the Summarize tool to summarize the text. Pass the entire text as the 'text' argument.
2. Use the Translate tool to translate the summary to Spanish. Pass the summary as the 'text' argument.
3. Immediately after using both tools, respond with the final result in the following format:
   Summary (English): [English summary]
   Translation (Spanish): [Spanish translation]

Do not use any tools after providing the formatted output.

{agent_scratchpad}"""  # Define the template for the agent's instructions
)

# Create an agent using the defined tools and prompt
agent = create_tool_calling_agent(llm, tools, prompt)

# Create an AgentExecutor to run the agent
agent_executor = AgentExecutor(
    agent=agent,  # The agent to execute
    tools=tools,  # The tools available to the agent
    verbose=True,  # Enable verbose output
    max_iterations=3,  # Set maximum number of iterations
    early_stopping_method="force"  # Force stop after max_iterations
)
```

```python
def run_agent_with_query(agent_executor, query):
    """
    Execute the agent with a given query and return the output.

    Args:
        agent_executor (AgentExecutor): The configured AgentExecutor to run.
        query (str): The input text to be processed by the agent.

    Returns:
        str: The output generated by the agent after processing the query.
    """
    # Invoke the agent_executor with the query as input
    result = agent_executor.invoke({"input": query})

    # Extract and return the 'output' field from the result
    return result['output']
```

###  Example usage

```python
# Define the input query
query = """The quick brown fox jumps over the lazy dog. This sentence is often used as a pangram in typography
to display font examples, as it contains every letter of the English alphabet. However, it's not the only pangram
in existence. Another example is 'Pack my box with five dozen liquor jugs', which is shorter but less commonly used."""

# Run the agent with the query
result = run_agent_with_query(agent_executor, query)

# Print the original query
print("\nQuery:")
print(query)

# Print the result from the agent
print("\nResult:")
print(result)
```

```text


[1m> Entering new AgentExecutor chain...[0m
[32;1m[1;3m
Invoking: `Summarize` with `{'text': "The quick brown fox jumps over the lazy dog. This sentence is often used as a pangram in typography to display font examples, as it contains every letter of the English alphabet. However, it's not the only pangram in existence. Another example is 'Pack my box with five dozen liquor jugs', which is shorter but less commonly used."}`


[0m[36;1m[1;3mThe sentence "The quick brown fox jumps over the lazy dog" is a well-known pangram used in typography to showcase fonts, as it includes every letter of the English alphabet. Another, shorter pangram is "Pack my box with five dozen liquor jugs," though it is less commonly used.[0m[32;1m[1;3m
Invoking: `Translate` with `{'text': "The quick brown fox jumps over the lazy dog. This sentence is often used as a pangram in typography to display font examples, as it contains every letter of the English alphabet. However, it's not the only pangram in existence. Another example is 'Pack my box with five dozen liquor jugs', which is shorter but less commonly used."}`


[0m[33;1m[1;3mLa rápida zorra marrón salta sobre el perro perezoso. Esta oración se utiliza a menudo como un pangrama en tipografía para mostrar ejemplos de fuentes, ya que contiene cada letra del alfabeto inglés. Sin embargo, no es el único pangrama que existe. Otro ejemplo es 'Empaca mi caja con cinco docenas de jarras de licor', que es más corto pero menos comúnmente utilizado.[0m[32;1m[1;3m
Invoking: `Summarize` with `{'text': "The quick brown fox jumps over the lazy dog. This sentence is often used as a pangram in typography to display font examples, as it contains every letter of the English alphabet. However, it's not the only pangram in existence. Another example is 'Pack my box with five dozen liquor jugs', which is shorter but less commonly used."}`


[0m[36;1m[1;3mThe sentence "The quick brown fox jumps over the lazy dog" is a well-known pangram used in typography to showcase fonts, as it includes every letter of the English alphabet. Another, shorter pangram is "Pack my box with five dozen liquor jugs," though it is less commonly used.[0m[32;1m[1;3m
Invoking: `Translate` with `{'text': "The quick brown fox jumps over the lazy dog. This sentence is often used as a pangram in typography to display font examples, as it contains every letter of the English alphabet. However, it's not the only pangram in existence. Another example is 'Pack my box with five dozen liquor jugs', which is shorter but less commonly used."}`


[0m[33;1m[1;3mLa rápida zorra marrón salta sobre el perro perezoso. Esta oración se utiliza a menudo como un pangrama en tipografía para mostrar ejemplos de fuentes, ya que contiene cada letra del alfabeto inglés. Sin embargo, no es el único pangrama que existe. Otro ejemplo es 'Empaca mi caja con cinco docenas de jarras de licor', que es más corto pero menos comúnmente utilizado.[0m[32;1m[1;3mSummary (English): The sentence "The quick brown fox jumps over the lazy dog" is a well-known pangram used in typography to showcase fonts, as it includes every letter of the English alphabet. Another, shorter pangram is "Pack my box with five dozen liquor jugs," though it is less commonly used.

Translation (Spanish): La rápida zorra marrón salta sobre el perro perezoso. Esta oración se utiliza a menudo como un pangrama en tipografía para mostrar ejemplos de fuentes, ya que contiene cada letra del alfabeto inglés. Sin embargo, no es el único pangrama que existe. Otro ejemplo es 'Empaca mi caja con cinco docenas de jarras de licor', que es más corto pero menos comúnmente utilizado.[0m

[1m> Finished chain.[0m

Query:
The quick brown fox jumps over the lazy dog. This sentence is often used as a pangram in typography
to display font examples, as it contains every letter of the English alphabet. However, it's not the only pangram
in existence. Another example is 'Pack my box with five dozen liquor jugs', which is shorter but less commonly used.

Result:
Summary (English): The sentence "The quick brown fox jumps over the lazy dog" is a well-known pangram used in typography to showcase fonts, as it includes every letter of the English alphabet. Another, shorter pangram is "Pack my box with five dozen liquor jugs," though it is less commonly used.

Translation (Spanish): La rápida zorra marrón salta sobre el perro perezoso. Esta oración se utiliza a menudo como un pangrama en tipografía para mostrar ejemplos de fuentes, ya que contiene cada letra del alfabeto inglés. Sin embargo, no es el único pangrama que existe. Otro ejemplo es 'Empaca mi caja con cinco docenas de jarras de licor', que es más corto pero menos comúnmente utilizado.
```

![](https://europe-west1-genai-agents-views-tracker.cloudfunctions.net/genai-agents-tracker?notebook=all-agents-tutorials--task-oriented-agent)
