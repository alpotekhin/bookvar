---
title: "Data Analysis Simple Agent with PydanticAI"
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: bd681451b254ac1a790e947b581d3997ab35013d
---

> [!note] Original source material
> This page preserves [`all_agents_tutorials/simple_data_analysis_agent_notebook-pydanticai.ipynb`](https://github.com/NirDiamant/GenAI_Agents/blob/bd681451b254ac1a790e947b581d3997ab35013d/all_agents_tutorials/simple_data_analysis_agent_notebook-pydanticai.ipynb) from
> *GenAI Agents* at commit `bd681451b254ac1a790e947b581d3997ab35013d`. License:
> [Custom non-commercial license](https://github.com/NirDiamant/GenAI_Agents/blob/bd681451b254ac1a790e947b581d3997ab35013d/LICENSE). Bookvar changed only
> publication markup, link paths, and characters required for safe rendering.



**This tutorial is based on the LangChain tutorial: `Data Analysis Simple Agent`. It demonstrates the same concept using PydanticAI as the agent framework.**

**You don’t need to be familiar with the LangChain notebook to follow along—this tutorial stands on its own and explains everything you need to know.** For more information about PydanticAI, visit their [official website](https://ai.pydantic.dev/), or check out the PydanticAI Overview in [this notebook](https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/simple_conversational_agent-pydanticai.ipynb).

In this version of the notebook, we replicate the [Data Analysis Simple Agent](https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/simple_data_analysis_agent_notebook.ipynb) workflow using **PydanticAI**. The primary difference is that LangChain includes a built-in agent designed to handle Pandas DataFrames and perform actions on them directly. PydanticAI, being a newer framework, does not yet include such a built-in tool. As a result, we’ll create the tool ourselves, providing an opportunity to explore how to build custom tools and implement retry logic with PydanticAI.

## Overview
This tutorial guides you through creating an AI-powered data analysis agent that can interpret and answer questions about a dataset using natural language. It combines language models with data manipulation tools to enable intuitive data exploration.

## Motivation
Data analysis often requires specialized knowledge, limiting access to insights for non-technical users. By creating an AI agent that understands natural language queries, we can democratize data analysis, allowing anyone to extract valuable information from complex datasets without needing to know programming or statistical tools.

## Key Components
1. Language Model: Processes natural language queries and generates human-like responses
2. Data Manipulation Framework: Handles dataset operations and analysis
3. Agent Framework: Connects the language model with data manipulation tools
4. Synthetic Dataset: Represents real-world data for demonstration purposes

## Method
1. Create a synthetic dataset representing car sales data
2. Construct an agent that combines the language model with data analysis capabilities
3. Create a tool that the agent can use to query our dataset.
4. Implement a query processing function to handle natural language questions
5. Demonstrate the agent's abilities with example queries

## Conclusion
This approach to data analysis offers significant benefits:
- Accessibility for non-technical users
- Flexibility in handling various query types
- Efficient ad-hoc data exploration

By making data insights more accessible, this method has the potential to transform how organizations leverage their data for decision-making across various fields and industries.

## Import libraries and set environment variables

```python
# %pip install -q pydantic-ai
```

```python
import os
import pandas as pd
import numpy as np

from dataclasses import dataclass
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Any

from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.messages import Message, MessagesTypeAdapter
from pydantic_ai.result import RunResult

# Set a random seed for reproducibility
np.random.seed(42)
```

```python
# Apply `nest_asyncio` to avoid errors when running asyncio code in a Jupyter notebook.
# This prevents `event loop is already running` errors by allowing nested event loops.

import nest_asyncio
nest_asyncio.apply()
```

```python
# Load environment
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LOGFIRE_IGNORE_NO_CONFIG'] = '1'
```

## Generate Sample Data

In this section, we create a sample dataset of car sales. This includes generating dates, car makes, models, colors, and other relevant information.

```python
# Generate sample data
n_rows = 1000

# Generate dates
start_date = datetime(2022, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(n_rows)]

# Define data categories
makes = ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'Nissan', 'BMW', 'Mercedes', 'Audi', 'Hyundai', 'Kia']
models = ['Sedan', 'SUV', 'Truck', 'Hatchback', 'Coupe', 'Van']
colors = ['Red', 'Blue', 'Black', 'White', 'Silver', 'Gray', 'Green']

# Create the dataset
data = {
    'Date': dates,
    'Make': np.random.choice(makes, n_rows),
    'Model': np.random.choice(models, n_rows),
    'Color': np.random.choice(colors, n_rows),
    'Year': np.random.randint(2015, 2023, n_rows),
    'Price': np.random.uniform(20000, 80000, n_rows).round(2),
    'Mileage': np.random.uniform(0, 100000, n_rows).round(0),
    'EngineSize': np.random.choice([1.6, 2.0, 2.5, 3.0, 3.5, 4.0], n_rows),
    'FuelEfficiency': np.random.uniform(20, 40, n_rows).round(1),
    'SalesPerson': np.random.choice(['Alice', 'Bob', 'Charlie', 'David', 'Eva'], n_rows)
}

# Create DataFrame and sort by date
df = pd.DataFrame(data).sort_values('Date')

# Display sample data and statistics
print("\nFirst few rows of the generated data:")
print(df.head())

print("\nDataFrame info:")
df.info()

print("\nSummary statistics:")
print(df.describe())
```

```text

First few rows of the generated data:
        Date       Make      Model  Color  Year     Price  Mileage  \
0 2022-01-01   Mercedes      Sedan  Green  2022  57952.65   5522.0   
1 2022-01-02  Chevrolet  Hatchback    Red  2021  58668.22  94238.0   
2 2022-01-03       Audi      Truck  White  2019  69187.87   7482.0   
3 2022-01-04     Nissan  Hatchback  Black  2016  40004.44  43846.0   
4 2022-01-05   Mercedes  Hatchback    Red  2016  63983.07  52988.0   

   EngineSize  FuelEfficiency SalesPerson  
0         2.0            24.7       Alice  
1         1.6            26.2         Bob  
2         2.0            28.0       David  
3         3.5            24.8       David  
4         2.5            24.1       Alice  

DataFrame info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 10 columns):
 #   Column          Non-Null Count  Dtype         
---  ------          --------------  -----         
 0   Date            1000 non-null   datetime64[ns]
 1   Make            1000 non-null   object        
 2   Model           1000 non-null   object        
 3   Color           1000 non-null   object        
 4   Year            1000 non-null   int64         
 5   Price           1000 non-null   float64       
 6   Mileage         1000 non-null   float64       
 7   EngineSize      1000 non-null   float64       
 8   FuelEfficiency  1000 non-null   float64       
 9   SalesPerson     1000 non-null   object        
dtypes: datetime64[ns](1), float64(4), int64(1), object(4)
memory usage: 78.3+ KB

Summary statistics:
                      Date         Year         Price       Mileage  \
count                 1000  1000.000000   1000.000000   1000.000000   
mean   2023-05-15 12:00:00  2018.445000  51145.360800  48484.643000   
min    2022-01-01 00:00:00  2015.000000  20026.570000     19.000000   
25%    2022-09-07 18:00:00  2017.000000  36859.940000  23191.500000   
50%    2023-05-15 12:00:00  2018.000000  52215.155000  47506.000000   
75%    2024-01-20 06:00:00  2020.000000  65741.147500  73880.250000   
max    2024-09-26 00:00:00  2022.000000  79972.640000  99762.000000   
std                    NaN     2.256117  17041.610861  29103.404593   

        EngineSize  FuelEfficiency  
count  1000.000000     1000.000000  
mean      2.744500       29.688500  
min       1.600000       20.000000  
25%       2.000000       24.500000  
50%       2.500000       29.700000  
75%       3.500000       34.700000  
max       4.000000       40.000000  
std       0.839389        5.896316
```

## Create Data Analysis Agent

Unlike the [LangChain notebook](https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/simple_data_analysis_agent_notebook.ipynb) on which this example is based, PydanticAI does not (yet?) have a built-in tool for processing Pandas DataFrames.

To address this, we’ll create a custom tool that implements the required functionality for our example.

We’ll begin by defining the agent itself, along with the dependencies that the tool will need. The tool implementation will follow in the next section.

#### Dependencies

PydanticAI uses a [dependency injection system](https://ai.pydantic.dev/dependencies/) to provide data and services to an agent’s system prompts, tools, and result validators.

We’ll use this system to define the DataFrame as a dependency, allowing us to reference it inside the tool.

```python
@dataclass
class Deps:
    """The only dependency we need is the DataFrame we'll be working with."""
    df: pd.DataFrame
```

```python
agent = Agent(
    model='openai:gpt-4o-mini',
    system_prompt="""You are an AI assistant that helps extract information from a pandas DataFrame.
    If asked about columns, be sure to check the column names first.
    Be concise in your answers.""",
    deps_type=Deps,

    # Allow the agent to make mistakes and correct itself. Details will be covered in the tool definition.
    retries=10,
)
```

## Create a Tool to Query the DataFrame

Our tool is straightforward. Unlike the LangChain function `create_pandas_dataframe_agent`, which you can see in the [LangChain notebook](https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/simple_data_analysis_agent_notebook.ipynb) and uses the Python REPL and can be dangerous, we run our queries using `pd.eval`.

`pd.eval` allows execution of only a subset of Pandas commands, limiting the potential for malicious code execution.

The downside is that this approach supports only a subset of the regular Pandas syntax, which means the agent may occasionally make mistakes by using unsupported syntax. To handle such cases, we enable retries. During the agent’s definition, we set the number of allowed retries to 10. If an error occurs during tool execution, we raise a `ModelRetry` exception to prompt the agent to retry its query.

```python
@agent.tool
async def df_query(ctx: RunContext[Deps], query: str) -> str:
    """A tool for running queries on the `pandas.DataFrame`. Use this tool to interact with the DataFrame.

    `query` will be executed using `pd.eval(query, target=df)`, so it must contain syntax compatible with
    `pandas.eval`.
    """

    # Print the query for debugging purposes and fun :)
    print(f'Running query: `{query}`')
    try:
        # Execute the query using `pd.eval` and return the result as a string (must be serializable).
        return str(pd.eval(query, target=ctx.deps.df))
    except Exception as e:
        #  On error, raise a `ModelRetry` exception with feedback for the agent.
        raise ModelRetry(f'query: `{query}` is not a valid query. Reason: `{e}`') from e
```

## Define Question-Asking Function

This function allows us to easily ask questions to our data analysis agent and display the results.

```python
def ask_agent(question):
    """Function to ask questions to the agent and display the response"""
    deps = Deps(df=df)
    print(f"Question: {question}")
    response = agent.run_sync(question, deps=deps)
    print(f"Answer: {response.new_messages()[-1].content}")
    print("---")
```

## Example Questions

Here are some example questions you can ask the data analysis agent. You can modify these or add your own questions to analyze the dataset.

```python
# Example questions
ask_agent("What are the column names in this dataset?")
ask_agent("How many rows are in this dataset?")
ask_agent("What is the average price of cars sold?")
```

```text
Question: What are the column names in this dataset?
Running query: `df.columns.tolist()`
Answer: The column names in the dataset are: 

- Date
- Make
- Model
- Color
- Year
- Price
- Mileage
- EngineSize
- FuelEfficiency
- SalesPerson
---
Question: How many rows are in this dataset?
Running query: `len(df)`
Running query: `df.shape[0]`
Answer: The dataset contains 1,000 rows.
---
Question: What is the average price of cars sold?
Running query: `cars['price'].mean()`
Running query: `df['price'].mean()`
Running query: `df.columns`
Running query: `df['Price'].mean()`
Answer: The average price of cars sold is approximately $51,145.36.
---
```

#### Analysis of Examples

As you can see in the above example, the agent got the column names right away but needed to retry a few times before arriving at the correct syntax to query the number of rows and the average price.

The primary issue was that the `Price` column name starts with a capital `P`, which caused some retries when querying the average price. We could improve the agent’s performance by including additional context in the prompt, such as column names, types, or usage examples, to help the agent arrive at correct answers more efficiently.

```python

```

![](https://europe-west1-genai-agents-views-tracker.cloudfunctions.net/genai-agents-tracker?notebook=all-agents-tutorials--simple-data-analysis-agent-notebook-pydanticai)
