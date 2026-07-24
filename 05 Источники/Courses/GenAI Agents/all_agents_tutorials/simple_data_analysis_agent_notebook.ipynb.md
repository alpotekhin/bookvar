---
title: "Data Analysis Simple Agent"
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: bd681451b254ac1a790e947b581d3997ab35013d
---

> [!note] Original source material
> This page preserves [`all_agents_tutorials/simple_data_analysis_agent_notebook.ipynb`](https://github.com/NirDiamant/GenAI_Agents/blob/bd681451b254ac1a790e947b581d3997ab35013d/all_agents_tutorials/simple_data_analysis_agent_notebook.ipynb) from
> *GenAI Agents* at commit `bd681451b254ac1a790e947b581d3997ab35013d`. License:
> [Custom non-commercial license](https://github.com/NirDiamant/GenAI_Agents/blob/bd681451b254ac1a790e947b581d3997ab35013d/LICENSE). Bookvar changed only
> publication markup, link paths, and characters required for safe rendering.



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
3. Implement a query processing function to handle natural language questions
4. Demonstrate the agent's abilities with example queries

## Conclusion
This approach to data analysis offers significant benefits:
- Accessibility for non-technical users
- Flexibility in handling various query types
- Efficient ad-hoc data exploration

By making data insights more accessible, this method has the potential to transform how organizations leverage their data for decision-making across various fields and industries.

## Import libraries and set environment variables

```python
import os
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load environment variables
from dotenv import load_dotenv
import os

# Load environment variables and set OpenAI API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Set a random seed for reproducibility
np.random.seed(42)
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
 4   Year            1000 non-null   int32         
 5   Price           1000 non-null   float64       
 6   Mileage         1000 non-null   float64       
 7   EngineSize      1000 non-null   float64       
 8   FuelEfficiency  1000 non-null   float64       
 9   SalesPerson     1000 non-null   object        
dtypes: datetime64[ns](1), float64(4), int32(1), object(4)
memory usage: 74.3+ KB

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

Here, we create a Pandas DataFrame agent using LangChain. This agent will be capable of analyzing our dataset and answering questions about it.

```python
# Create the Pandas DataFrame agent
agent = create_pandas_dataframe_agent(
    ChatOpenAI(model="gpt-4o", temperature=0),
    df,
    verbose=True,
    allow_dangerous_code=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)
print("Data Analysis Agent is ready. You can now ask questions about the data.")
```

```text
Data Analysis Agent is ready. You can now ask questions about the data.
```

## Define Question-Asking Function

This function allows us to easily ask questions to our data analysis agent and display the results.

```python
def ask_agent(question):
    """Function to ask questions to the agent and display the response"""
    response = agent.invoke({
        "input": question,
        "agent_scratchpad": f"Human: {question}\nAI: To answer this question, I need to use Python to analyze the dataframe. I'll use the python_repl_ast tool.\n\nAction: python_repl_ast\nAction Input: ",
    })
    print(f"Question: {question}")
    print(f"Answer: {response}")
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
C:\Users\N7\AppData\Local\Temp\ipykernel_16872\610968568.py:3: LangChainDeprecationWarning: The method `Chain.run` was deprecated in langchain 0.1.0 and will be removed in 1.0. Use invoke instead.
  response = agent.invoke({
```

```text


[1m> Entering new AgentExecutor chain...[0m
[32;1m[1;3m
Invoking: `python_repl_ast` with `{'query': 'df.columns.tolist()'}`


[0m[36;1m[1;3m['Date', 'Make', 'Model', 'Color', 'Year', 'Price', 'Mileage', 'EngineSize', 'FuelEfficiency', 'SalesPerson'][0m[32;1m[1;3mThe column names in the dataset are:
1. Date
2. Make
3. Model
4. Color
5. Year
6. Price
7. Mileage
8. EngineSize
9. FuelEfficiency
10. SalesPerson[0m

[1m> Finished chain.[0m
Question: What are the column names in this dataset?
Answer: The column names in the dataset are:
1. Date
2. Make
3. Model
4. Color
5. Year
6. Price
7. Mileage
8. EngineSize
9. FuelEfficiency
10. SalesPerson
---


[1m> Entering new AgentExecutor chain...[0m
[32;1m[1;3m
Invoking: `python_repl_ast` with `{'query': 'df.shape[0]'}`


[0m[36;1m[1;3m1000[0m[32;1m[1;3mThe dataset contains 1000 rows.[0m

[1m> Finished chain.[0m
Question: How many rows are in this dataset?
Answer: The dataset contains 1000 rows.
---


[1m> Entering new AgentExecutor chain...[0m
[32;1m[1;3m
Invoking: `python_repl_ast` with `{'query': "df['Price'].mean()"}`


[0m[36;1m[1;3m51145.360799999995[0m[32;1m[1;3mThe average price of cars sold is approximately $51,145.36.[0m

[1m> Finished chain.[0m
Question: What is the average price of cars sold?
Answer: The average price of cars sold is approximately $51,145.36.
---
```

![](https://europe-west1-genai-agents-views-tracker.cloudfunctions.net/genai-agents-tracker?notebook=all-agents-tutorials--simple-data-analysis-agent-notebook)
