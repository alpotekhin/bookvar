---
title: "**EU Green deal compliance FAQ Bot**"
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: bd681451b254ac1a790e947b581d3997ab35013d
---

> [!note] Original source material
> This page preserves [`all_agents_tutorials/EU_Green_Compliance_FAQ_Bot.ipynb`](https://github.com/NirDiamant/GenAI_Agents/blob/bd681451b254ac1a790e947b581d3997ab35013d/all_agents_tutorials/EU_Green_Compliance_FAQ_Bot.ipynb) from
> *GenAI Agents* at commit `bd681451b254ac1a790e947b581d3997ab35013d`. License:
> [Custom non-commercial license](https://github.com/NirDiamant/GenAI_Agents/blob/bd681451b254ac1a790e947b581d3997ab35013d/LICENSE). Bookvar changed only
> publication markup, link paths, and characters required for safe rendering.



<a href="https://colab.research.google.com/github/Avtr99/GenAI_Agents/blob/main/EU_Green_Compliance_FAQ_Bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

A RAG based AI agent that helps SMEs/ businesses quickly find answers to common questions about EU green deal policies. This bot will focus on responding to frequently asked questions (FAQs) related to the most relevant regulations, providing short and clear answers to help businesses understand and meet compliance standards.

**Functionality:** The bot answers basic questions about key EU environmental regulations, focusing on common requirements like waste management, carbon footprint reporting, and renewable energy.

# **Motivation**

Navigating EU green compliance can be overwhelming for businesses, especially smaller ones without dedicated resources. The project aims to simplify this process by creating a smart, accessible FAQ bot that provides instant, accurate answers to common questions about the EU Green Deal, emissions reporting, and waste management. By helping businesses understand and meet green regulations, compliance easier—it will contribute to a more sustainable future for everyone.

# **Method Details**

### **Document Storage and Embedding:**
Large documents are preprocessed into manageable chunks using a LLM for semantic chunking and stored in a vectorstore.
### **Query Processing:**
User queries are first rephrased to improve clarity and intent matching. The rephrased queries are then embedded using the same model. Using vector similarity and semantic relevance, the system retrieves the most relevant document chunks from the FAISS vectorstore.

### **Summarization:**
Context-aware and concise response are generated from the retrieved chunks using an LLM. This summarization step emphasizes clarity and ensures the answer directly aligns with the user’s query, distilling only the most relevant information.
### **Evaluation:**
Generated answers are evaluated against a gold Q&A dataset for factual accuracy and contextual relevance. The evaluation process includes metrics such as cosine similarity, F1 score, and semantic match.
### **Key Agents:**
Retriever Agent:
Retrieves the most semantically relevant chunks from the FAISS vectorstore based on the processed and embedded user query

Summarizer Agent:
Generate a coherent, concise response based on retrieved content.

Evaluation Agent:
Evaluates the quality of the generated response using gold-standard answers and similarity metrics.

# **Benefits of the Approach**

### **Accuracy and Fact-Checking:**
Reduces hallucination by grounding answers in external knowledge.

### **Modularity:**
The system's components (retriever, summarizer, evaluator) are independently - designed, allowing seamless improvements or replacements as needed.

### **Better evaluation:**
Combines advanced metrics like cosine similarity and F1 scores with gold q&a benchmark.

### **Flexibility:**
Adaptable across various domains and use cases with minimal pipeline changes, accommodating tailored retriever and summarizer configurations.

### **Context-Aware Responses:**
Incorporates context from both the query and the retrieved information.

# **Setup**

Import the required libraries

```python
!pip install langchain langchain-openai python-dotenv openai
pip install langchain-experimental
pip install faiss-cpu
```

```text
Requirement already satisfied: langchain in /usr/local/lib/python3.10/dist-packages (0.3.7)
Collecting langchain-openai
  Downloading langchain_openai-0.2.8-py3-none-any.whl.metadata (2.6 kB)
Collecting python-dotenv
  Downloading python_dotenv-1.0.1-py3-none-any.whl.metadata (23 kB)
Requirement already satisfied: openai in /usr/local/lib/python3.10/dist-packages (1.54.4)
Requirement already satisfied: PyYAML>=5.3 in /usr/local/lib/python3.10/dist-packages (from langchain) (6.0.2)
Requirement already satisfied: SQLAlchemy<3,>=1.4 in /usr/local/lib/python3.10/dist-packages (from langchain) (2.0.36)
Requirement already satisfied: aiohttp<4.0.0,>=3.8.3 in /usr/local/lib/python3.10/dist-packages (from langchain) (3.10.10)
Requirement already satisfied: async-timeout<5.0.0,>=4.0.0 in /usr/local/lib/python3.10/dist-packages (from langchain) (4.0.3)
Requirement already satisfied: langchain-core<0.4.0,>=0.3.15 in /usr/local/lib/python3.10/dist-packages (from langchain) (0.3.17)
Requirement already satisfied: langchain-text-splitters<0.4.0,>=0.3.0 in /usr/local/lib/python3.10/dist-packages (from langchain) (0.3.2)
Requirement already satisfied: langsmith<0.2.0,>=0.1.17 in /usr/local/lib/python3.10/dist-packages (from langchain) (0.1.142)
Requirement already satisfied: numpy<2,>=1 in /usr/local/lib/python3.10/dist-packages (from langchain) (1.26.4)
Requirement already satisfied: pydantic<3.0.0,>=2.7.4 in /usr/local/lib/python3.10/dist-packages (from langchain) (2.9.2)
Requirement already satisfied: requests<3,>=2 in /usr/local/lib/python3.10/dist-packages (from langchain) (2.32.3)
Requirement already satisfied: tenacity!=8.4.0,<10,>=8.1.0 in /usr/local/lib/python3.10/dist-packages (from langchain) (9.0.0)
Collecting tiktoken<1,>=0.7 (from langchain-openai)
  Downloading tiktoken-0.8.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.6 kB)
Requirement already satisfied: anyio<5,>=3.5.0 in /usr/local/lib/python3.10/dist-packages (from openai) (3.7.1)
Requirement already satisfied: distro<2,>=1.7.0 in /usr/local/lib/python3.10/dist-packages (from openai) (1.9.0)
Requirement already satisfied: httpx<1,>=0.23.0 in /usr/local/lib/python3.10/dist-packages (from openai) (0.27.2)
Requirement already satisfied: jiter<1,>=0.4.0 in /usr/local/lib/python3.10/dist-packages (from openai) (0.7.1)
Requirement already satisfied: sniffio in /usr/local/lib/python3.10/dist-packages (from openai) (1.3.1)
Requirement already satisfied: tqdm>4 in /usr/local/lib/python3.10/dist-packages (from openai) (4.66.6)
Requirement already satisfied: typing-extensions<5,>=4.11 in /usr/local/lib/python3.10/dist-packages (from openai) (4.12.2)
Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (2.4.3)
Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.3.1)
Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (24.2.0)
Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.5.0)
Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (6.1.0)
Requirement already satisfied: yarl<2.0,>=1.12.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.17.1)
Requirement already satisfied: idna>=2.8 in /usr/local/lib/python3.10/dist-packages (from anyio<5,>=3.5.0->openai) (3.10)
Requirement already satisfied: exceptiongroup in /usr/local/lib/python3.10/dist-packages (from anyio<5,>=3.5.0->openai) (1.2.2)
Requirement already satisfied: certifi in /usr/local/lib/python3.10/dist-packages (from httpx<1,>=0.23.0->openai) (2024.8.30)
Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.10/dist-packages (from httpx<1,>=0.23.0->openai) (1.0.6)
Requirement already satisfied: h11<0.15,>=0.13 in /usr/local/lib/python3.10/dist-packages (from httpcore==1.*->httpx<1,>=0.23.0->openai) (0.14.0)
Requirement already satisfied: jsonpatch<2.0,>=1.33 in /usr/local/lib/python3.10/dist-packages (from langchain-core<0.4.0,>=0.3.15->langchain) (1.33)
Requirement already satisfied: packaging<25,>=23.2 in /usr/local/lib/python3.10/dist-packages (from langchain-core<0.4.0,>=0.3.15->langchain) (24.2)
Requirement already satisfied: orjson<4.0.0,>=3.9.14 in /usr/local/lib/python3.10/dist-packages (from langsmith<0.2.0,>=0.1.17->langchain) (3.10.11)
Requirement already satisfied: requests-toolbelt<2.0.0,>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from langsmith<0.2.0,>=0.1.17->langchain) (1.0.0)
Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.10/dist-packages (from pydantic<3.0.0,>=2.7.4->langchain) (0.7.0)
Requirement already satisfied: pydantic-core==2.23.4 in /usr/local/lib/python3.10/dist-packages (from pydantic<3.0.0,>=2.7.4->langchain) (2.23.4)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2->langchain) (3.4.0)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2->langchain) (2.2.3)
Requirement already satisfied: greenlet!=0.4.17 in /usr/local/lib/python3.10/dist-packages (from SQLAlchemy<3,>=1.4->langchain) (3.1.1)
Requirement already satisfied: regex>=2022.1.18 in /usr/local/lib/python3.10/dist-packages (from tiktoken<1,>=0.7->langchain-openai) (2024.9.11)
Requirement already satisfied: jsonpointer>=1.9 in /usr/local/lib/python3.10/dist-packages (from jsonpatch<2.0,>=1.33->langchain-core<0.4.0,>=0.3.15->langchain) (3.0.0)
Requirement already satisfied: propcache>=0.2.0 in /usr/local/lib/python3.10/dist-packages (from yarl<2.0,>=1.12.0->aiohttp<4.0.0,>=3.8.3->langchain) (0.2.0)
Downloading langchain_openai-0.2.8-py3-none-any.whl (50 kB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m50.4/50.4 kB[0m [31m3.6 MB/s[0m eta [36m0:00:00[0m
[?25hDownloading python_dotenv-1.0.1-py3-none-any.whl (19 kB)
Downloading tiktoken-0.8.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.2 MB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.2/1.2 MB[0m [31m40.9 MB/s[0m eta [36m0:00:00[0m
[?25hInstalling collected packages: python-dotenv, tiktoken, langchain-openai
Successfully installed langchain-openai-0.2.8 python-dotenv-1.0.1 tiktoken-0.8.0
Collecting rank_bm25
  Downloading rank_bm25-0.2.2-py3-none-any.whl.metadata (3.2 kB)
Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from rank_bm25) (1.26.4)
Downloading rank_bm25-0.2.2-py3-none-any.whl (8.6 kB)
Installing collected packages: rank_bm25
Successfully installed rank_bm25-0.2.2
```

```python

# Then import necessary modules
import os  # Add this import first
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import List, Dict
from dotenv import load_dotenv

# Set your API key
os.environ["OPENAI_API_KEY"] = "ADD your key here" #set an openAI key
```

initialize language model

```python
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0.7)
```

# **Graph**

```python
from IPython.display import Image, display

def render_mermaid(graph_definition: str, width: int = 800, height: int = 600):
    """
    Render a mermaid graph as an image using mermaid.ink and scale it.

    Args:
        graph_definition (str): The mermaid graph definition in string format.
        width (int): Desired width of the graph.
        height (int): Desired height of the graph.
    """
    import base64
    graph_bytes = graph_definition.encode("utf-8")
    base64_bytes = base64.urlsafe_b64encode(graph_bytes)
    base64_string = base64_bytes.decode("ascii")
    image_url = f"https://mermaid.ink/img/{base64_string}"
    display(Image(url=image_url, width=width, height=height))

# Modified Mermaid Graph
mermaid_graph = """
graph TD
    subgraph User_Query
        U[User Input Query] -->|Initiates Process| E[Rephrased Query]
    end
    subgraph Knowledge_Base_Processing
        A[EU Compliance Documents] -->|Text Splitter| B[Document Chunks]
        B -->|OpenAI Embedding| C[Vector Embeddings]
        C -->|Embeddings to Retriever| F[Retriever Agent]
    end
    subgraph Retriever_Agent
        E -->|Query Rephrasing| F[Processed Query]
        F -->|Vector Similarity Search| H[Retriever Search]
        H -->|Top-K Relevant Chunks| J[Retrieved Chunks]
    end
    subgraph Summarizer_Agent
        J -->|Contextual Summary| K[Context-Aware Summary]
        K -->|OpenAI LLM| L[Generated Summary]
        L -->|Summary for User| M[Final Summary]
    end
    subgraph Evaluation_Agent
        L -->|Evaluate Answer| N{Evaluation Metrics}
        P[(Gold Q&A Dictionary)] -->|Benchmark for Evaluation| N
        N -->|Cosine Similarity, F1 Score| O{Score Evaluation}
        N -->|Precision@1, Semantic Match| O
        O -->|Displayed Answer| M
    end
    M -->|Final Answer| T[User]
"""
render_mermaid(mermaid_graph, width=1200, height=1600)
```

```text
<IPython.core.display.Image object>
```

# Chunking the documents and Vector store

Semantic chunker using LLM and storing in a vectorstore

```python
import os
import sys
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Set the folder path containing the documents
folder_path = "/content/data"  # Path to the folder containing documents

# Step 2: Read and combine content from all documents in the folder
def load_documents(folder_path):
    """
    Load and combine content from all text documents in the specified folder.

    Args:
        folder_path (str): Path to the folder containing documents.

    Returns:
        str: Combined content of all documents.
    """
    combined_content = ""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith((".txt", ".md", ".docx")):  # Adjust extensions as needed
            with open(file_path, 'r', encoding='utf-8') as file:
                combined_content += file.read() + "\n"
    return combined_content

content = load_documents(folder_path)
if not content:
    raise ValueError("No valid documents found in the folder.")

# Step 3: Initialize SemanticChunker with the custom embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")  # Specify the desired embedding model
text_splitter = SemanticChunker(
    embeddings=embedding_model,  # Use the custom embedding model here
    breakpoint_threshold_type='percentile',  # Use percentile-based semantic shifts for splitting
    breakpoint_threshold_amount=90  # Define the threshold value (90th percentile)
)

# Step 4: Create semantic chunks from the combined document content
docs = text_splitter.create_documents([content])  # Semantic chunks as documents
print(f"Generated {len(docs)} semantic chunks.")

# Step 5: Embed and store chunks in FAISS vectorstore using the custom embedding model
vectorstore = FAISS.from_documents(docs, embedding_model)

# Step 6: Configure a retriever for the chunks
chunks_query_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Retrieve top-3 relevant chunks

# Step 7: Example Query
query = "What are the goals of the European Green Deal?"
retrieved_chunks = chunks_query_retriever.invoke(query)

# Output the retrieved chunks for the query
print("Retrieved Chunks for the Query:")
for idx, chunk in enumerate(retrieved_chunks, start=1):
    print(f"Chunk {idx}: {chunk.page_content}")
```

```text
Generated 1531 semantic chunks.
Retrieved Chunks for the Query:
Chunk 1: What is the European Green Deal?
Chunk 2: Forests and oceans are being polluted and destroyed 1 . The European Green Deal is a response to these challenges. It is a new growth strategy that aims to transform the EU into a fair and prosperous society, with a modern, resource-efficient and competitive economy where there are no net emissions of greenhouse gases in 2050 and where economic growth is decoupled from resource use. It also aims to protect, conserve and enhance the EU's natural capital, and protect the health and well-being of citizens from environment-related risks and impacts. At the same time, this transition must be just and inclusive.
Chunk 3: The policy response must be bold and comprehensive and seek to maximise benefits for health, quality of life, resilience and competitiveness. It will require intense coordination to exploit the available synergies across all policy areas 2 . The Green Deal is an integral part of this Commission’s strategy to implement the United Nation’s 2030 Agenda and the sustainable development goals 3 , and the other priorities announced in President von der Leyen’s political guidelines 4 . As part of the Green Deal, the Commission will refocus the European Semester process of macroeconomic coordination to integrate the United Nations’ sustainable development goals, to put sustainability and the well-being of citizens at the centre of economic policy, and the sustainable development goals at the heart of the EU’s policymaking and action. The figure below illustrates the various elements of the Green Deal. Figure 1: The European Green Deal

2.Transforming the EU’s economy for a sustainable future

2.1.Designing a set of deeply transformative policies

To deliver the European Green Deal, there is a need to rethink policies for clean energy supply across the economy, industry, production and consumption, large-scale infrastructure, transport, food and agriculture, construction, taxation and social benefits. To achieve these aims, it is essential to increase the value given to protecting and restoring natural ecosystems, to the sustainable use of resources and to improving human health. This is where transformational change is most needed and potentially most beneficial for the EU economy, society and natural environment. The EU should also promote and invest in the necessary digital transformation and tools as these are essential enablers of the changes. While all of these areas for action are strongly interlinked and mutually reinforcing, careful attention will have to be paid when there are potential trade-offs between economic, environmental and social objectives. The Green Deal will make consistent use of all policy levers: regulation and standardisation, investment and innovation, national reforms, dialogue with social partners and international cooperation. The European Pillar of Social Rights will guide action in ensuring that no one is left behind. New measures on their own will not be enough to achieve the European Green Deal’s objectives.
```

# **Define the different functions for the collaboration system**

Next, the retriever agent should retreive the relevant chunks. Using both vector similarity and LLM-based grading.

## **Retriever agent**

```python
from typing import List, Dict
from openai import OpenAI
import json
import numpy as np
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.agents import initialize_agent
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor
from langchain_openai import OpenAI
import requests

class RetrieverAgent:
    def __init__(self, vectorstore, model="gpt-4o-mini", temperature=0.0):
        """
        Initialize the Retriever Agent with a FAISS vectorstore and OpenAI model.

        Args:
            vectorstore: FAISS vectorstore containing document chunks and their embeddings
            model (str): OpenAI model to use for relevance scoring (default: gpt-4o-mini)
        """
        self.vectorstore = vectorstore
        self.model = model
        self.temperature = temperature
        openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure the OpenAI API key is set from environment variable

        # Initialize the LLM client for grading
        self.llm = OpenAI(model=self.model, temperature=self.temperature)

        # Define the system prompt for grading
        self.system = """You are a grader assessing relevance of a retrieved document to a user question.
                         If the document contains keyword(s) or semantic meaning related to the user question,
                         grade it as relevant.
                         It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
                         Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    def _get_relevance_score(self, query: str, chunk_text: str) -> str:
        """
        Use the LLM with function call to grade the relevance of the chunk.

        Args:
            query (str): User query
            chunk_text (str): Text content of the chunk

        Returns:
            str: 'yes' or 'no' indicating whether the chunk is relevant or not
        """
        prompt = f"""Query: {query}
                    Chunk: {chunk_text}
                    Grade the relevance of this chunk to the query. Respond only with 'yes' or 'no'."""

        try:
            # Use LLM to grade the chunk relevance
            response = self.llm.generate([prompt])  # Assuming llm has a generate method
            grade = response['choices'][0]['text'].strip()
            return grade.lower()

        except Exception as e:
            print(f"Error in grading: {e}")
            return "no"  # Default to no if there's an error

    def retrieve_relevant_chunks(self, query: str, top_k: int = 3, rerank: bool = True) -> List[Dict]:
      """
      Retrieve and optionally rerank the most relevant chunks using both vector similarity
      and LLM-based grading.

      Args:
          query (str): User query
          top_k (int): Number of top relevant chunks to return
          rerank (bool): Whether to rerank results using LLM grading

      Returns:
          list: List of dictionaries containing similarity scores and chunk text
      """
      # First, get candidates using vector similarity
      retrieved_docs = self.vectorstore.similarity_search_with_score(
          query,
          k=top_k * (2 if rerank else 1)  # Get more candidates if reranking
      )

      # Debugging: Print the raw retrieved_docs to check its structure
      print("Retrieved Docs (Raw):", retrieved_docs)

      relevant_chunks = []

      for doc, vector_score in retrieved_docs:
          # Use vector_score directly for similarity, and 1 - vector_score for ranking
          chunk_info = {
              "vector_similarity": float(vector_score),  # Vector similarity score
              "chunk_text": doc.page_content,
              "metadata": doc.metadata
          }

          if rerank:
              # Get LLM-based relevance grade ('yes' or 'no')
              relevance_grade = self._get_relevance_score(query, doc.page_content)

              # Only add chunks that are graded as relevant ('yes')
              if relevance_grade == "yes":
                  chunk_info["relevance_grade"] = relevance_grade
                  chunk_info["combined_score"] = 1 - vector_score  # Adjust this as necessary
                  relevant_chunks.append(chunk_info)
          else:
              # If reranking is disabled, just use vector similarity
              chunk_info["combined_score"] = 1 - vector_score  # Adjust this as necessary
              relevant_chunks.append(chunk_info)

      # Sort by combined score and take top_k
      relevant_chunks.sort(key=lambda x: x["combined_score"], reverse=True)
      return relevant_chunks[:top_k]


    def batch_retrieve(self, queries: List[str], top_k: int = 3, rerank: bool = True) -> Dict[str, List[Dict]​]:
        """
        Batch process multiple queries.

        Args:
            queries (List[str]): List of queries to process
            top_k (int): Number of top relevant chunks to return per query
            rerank (bool): Whether to rerank results using LLM grading

        Returns:
            Dict[str, List[Dict]​]: Dictionary mapping queries to their relevant chunks
        """
        results = {}
        for query in queries:
            results[query] = self.retrieve_relevant_chunks(query, top_k, rerank)
        return results

def create_retriever_agent(vectorstore, model="gpt-4o-mini", temperature=0.0):
    """
    Factory function to create a RetrieverAgent instance.

    Args:
        vectorstore: FAISS vectorstore containing document chunks
        model (str): OpenAI model to use for scoring (default: gpt-4o-mini)

    Returns:
        RetrieverAgent: Initialized retriever agent
    """
    return RetrieverAgent(vectorstore, model, temperature)
```

## **Summarizer Agent**

Context aware summarization using LLM

```python
import openai
import os
import requests
from typing import List, Dict

class SummarizerAgent:
    def __init__(self, model="gpt-4o-mini"):  # Default model can be adjusted
        """
        Initialize the Summarizer Agent with OpenAI model.

        Args:
            model (str): OpenAI model to use for summarization (default: gpt-4o-mini)
        """
        self.model = model
        openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure the OpenAI API key is set from environment variable

    def summarize_text(self, query: str, text: str) -> str:
        """
        Summarize the given text in the context of the query, focusing on concise and clear details within two sentences.

        Args:
            query (str): User query.
            text (str): Text content to summarize.

        Returns:
            str: Concise and clear summary relevant to the query.
        """
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"  # Ensure the OpenAI API key is set
        }

        prompt = f"""Summarize the following text based on the query. Focus on extracting the most relevant details in a clear and concise manner, ensuring the summary is no more than two sentences.

        Query: {query}

        Text to summarize: {text}

        Please make sure the summary is brief, clear, and focuses on the key information, avoiding unnecessary details and providing a direct answer to the query.
        """

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a summarization assistant. Your task is to summarize text into two sentences, focusing on the key points and ensuring clarity and conciseness."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,  # Low temperature for more focused responses
            "max_tokens": 150  # Ensure a concise summary
        }

        try:
            # Make the request to OpenAI's API
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()  # Raise an exception if the request fails

            # Extract the summarized content from the response
            result = response.json()
            summarized_text = result['choices'][0]['message']['content'].strip()
            return summarized_text

        except requests.exceptions.RequestException as e:
            print(f"Error in summarization: {e}")
            return "Sorry, I could not generate the summary at the moment."

    def batch_summarize(self, queries: List[str], texts: List[str]) -> Dict[str, str]:
        """
        Batch process multiple queries and summarize corresponding texts.

        Args:
            queries (List[str]): List of queries to process.
            texts (List[str]): List of texts to summarize.

        Returns:
            Dict[str, str]: Dictionary mapping each query to its summarized text.
        """
        summaries = {}
        for query, text in zip(queries, texts):
            summaries[query] = self.summarize_text(query, text)
        return summaries

# Example usage of the SummarizerAgent
summarizer = SummarizerAgent(model="gpt-4o-mini")  # Use the same model or another available model

query = "What is the European Green Deal?"
text = """The European Green Deal is a set of policy initiatives by the European Commission to address climate change, promote sustainability, and reduce carbon emissions by 2030. The Deal includes measures to promote clean energy, sustainable agriculture, and investments in green technologies. It aims to make Europe the first carbon-neutral continent by 2050."""

summary = summarizer.summarize_text(query, text)
print(f"Summary: {summary}")
```

```text
Summary: The European Green Deal is a comprehensive set of policy initiatives by the European Commission aimed at combating climate change and achieving carbon neutrality by 2050. It includes measures to promote clean energy, sustainable agriculture, and investments in green technologies, with a target to reduce carbon emissions by 2030.
```

# **Evaluation Agent**

Gold Q&A: List of curated question and answers that will be used to evaluted the answer

```python
gold_qa_dict = [
    {"query": "What is the European Green Deal (EGD)?", "answer": "The EGD is the EU’s strategy to reach net zero greenhouse gas emissions by 2050 while achieving sustainable economic growth. It covers policies across sectors like agriculture, energy, and manufacturing to ensure products meet higher sustainability standards."},
    {"query": "What is the Farm to Fork (F2F) Strategy?", "answer": "The F2F strategy is part of the EGD, focusing on making the EU’s food system fair, healthy, and environmentally friendly. It targets reducing pesticide use, nutrient loss, and promoting organic farming."},
    {"query": "What is the Circular Economy Action Plan (CEAP)?", "answer": "CEAP aims to eliminate waste by promoting the reuse, repair, and recycling of materials. It emphasizes creating sustainable products and reducing waste generation in industries like packaging, textiles, and electronics."},
    {"query": "What is the EU Green Deal Industrial Plan?", "answer": "The Plan aims to enhance Europe’s net-zero industrial base by simplifying regulations, increasing funding, developing skills, and fostering trade. It focuses on manufacturing key technologies like batteries, hydrogen systems, and wind turbines to achieve climate neutrality by 2050."},
    {"query": "What is the Net-Zero Industry Act (NZIA)?", "answer": "The NZIA aims to boost the EU's manufacturing capacity for net-zero technologies, such as solar panels, batteries, and electrolysers. It sets goals like manufacturing at least 40% of strategic net-zero technologies domestically by 2030."},
    {"query": "What is the EU Biodiversity Strategy for 2030?", "answer": "A key part of the Green Deal, it focuses on reversing biodiversity loss by restoring degraded ecosystems, reducing pesticide use by 50%, and ensuring 25% of farmland is organic by 2030."},
    {"query": "What is the Carbon Border Adjustment Mechanism (CBAM)?", "answer": "CBAM is a policy tool designed to prevent carbon leakage by imposing carbon costs on imports of certain goods from countries with less stringent climate policies. It ensures that imported products are priced similarly to EU-manufactured goods under the EU's carbon pricing system."},
    {"query": "Which sectors does CBAM initially cover?", "answer": "CBAM applies to high-emission sectors such as cement, iron and steel, fertilizers, electricity, and aluminum. Additional sectors may be included in the future."},
    {"query": "How does CBAM impact SMEs exporting to the EU?", "answer": "SMEs exporting CBAM-regulated goods must report the carbon emissions embedded in their products and potentially pay a carbon price. This may require investment in cleaner technologies and better transparency in production processes."},
    {"query": "When will CBAM come into effect?", "answer": "CBAM will be implemented in stages, starting with a reporting phase in 2023 and transitioning to full operation with financial obligations by 2026."},
    {"query": "How can exporters mitigate CBAM costs?", "answer": "Exporters can invest in low-carbon production methods or provide evidence of carbon taxes already paid in their home countries to reduce or eliminate CBAM charges."},
    {"query": "What sustainability standards must SMEs exporting to the EU meet?", "answer": "SMEs must meet standards for reduced waste, traceable production, eco-friendly packaging, and compliance with the new Ecodesign for Sustainable Products Regulation."},
    {"query": "What are the traceability requirements for exporters?", "answer": "Exporters must provide detailed information on product life cycles, including manufacturing, materials used, and compliance with sustainability criteria."},
    {"query": "How does the Carbon Border Adjustment Mechanism (CBAM) affect imports?", "answer": "CBAM imposes carbon taxes on imported goods with high greenhouse gas footprints, ensuring imports align with EU environmental standards."},
    {"query": "What is required under the new EU organic regulations?", "answer": "Imported organic products must display control body codes, follow strict organic certification rules, and meet labeling requirements."},
    {"query": "How does the Green Deal Industrial Plan simplify regulations for SMEs?", "answer": "The Plan introduces streamlined permitting processes and 'one-stop shops' to reduce red tape for projects related to renewable technologies."},
    {"query": "What is the Digital Product Passport (DPP)?", "answer": "The DPP provides detailed information about a product’s lifecycle, ensuring traceability and compliance with sustainability standards. It helps SMEs align with EU buyers' expectations."},
    {"query": "What are the biodiversity-related commitments for agricultural land?", "answer": "By 2030, 10% of farmland must feature biodiversity-friendly measures, and pesticide use must be cut by 50%."},
    {"query": "What challenges might SMEs face due to the EGD?", "answer": "SMEs may encounter higher production costs, complex sustainability reporting requirements, and the need to adapt to new eco-friendly technologies."},
    {"query": "What are the compliance deadlines for key regulations?", "answer": "Major regulations like the revision of pesticide use directives and the CBAM will be implemented in stages, with some taking effect by 2024."},
    {"query": "How does the EU support skill development for the green transition?", "answer": "The EU is establishing Net-Zero Industry Academies to train workers in net-zero technologies, with funding for reskilling and upskilling programs."},
    {"query": "What is the timeline for major Green Deal initiatives?", "answer": "Key initiatives like the NZIA and biodiversity commitments have milestones up to 2030, with significant mid-term reviews and funding disbursements expected between 2023 and 2026."},
    {"query": "What funding mechanisms are available for SMEs under the Green Deal?", "answer": "SMEs can access funding through programs like the Innovation Fund, InvestEU, and the European Sovereignty Fund. These mechanisms support green technology projects and offer tax breaks."},
    {"query": "What is the European Hydrogen Bank?", "answer": "It is a financial instrument to support renewable hydrogen production and imports. The Bank offers subsidies to bridge the cost gap between renewable and fossil hydrogen."},
    {"query": "What trade opportunities does the Green Deal provide?", "answer": "The Plan promotes open and fair trade through partnerships, free trade agreements, and initiatives like the Critical Raw Materials Club to ensure supply chain resilience."},
    {"query": "How can SMEs benefit from the EU Green Deal?", "answer": "SMEs can capitalize on increased demand for sustainable products, gain partnerships with EU companies, and access new markets driven by sustainability goals."},
    {"query": "What support is available for SMEs transitioning to sustainable practices?", "answer": "EU-based programs provide subsidies, technical support, and resources like the Digital Product Passport to help SMEs adapt."},
    {"query": "What opportunities do CEAP and F2F provide?", "answer": "These initiatives create markets for sustainable products, such as organic food and recycled textiles, enhancing SME competitiveness."},
    {"query": "What is the role of the EU Digital Product Passport?", "answer": "This tool standardizes and simplifies compliance, providing detailed product information to buyers while promoting transparency."},
    {"query": "What are Net-Zero Strategic Projects?", "answer": "These are priority projects essential for the EU's energy transition, such as large-scale solar or battery manufacturing plants. They benefit from accelerated permitting and funding."},
    {"query": "How does the EU address biodiversity in urban planning?", "answer": "Through the Green City Accord, urban planning integrates green spaces and biodiversity-focused infrastructure."},
    {"query": "What role does hydrogen play in the EU's climate strategy?", "answer": "Hydrogen is a cornerstone for reducing industrial emissions, with a target of producing 10 million tonnes of renewable hydrogen in the EU and importing an additional 10 million tonnes by 2030."},
    {"query": "What are the packaging requirements under the EGD?", "answer": "All packaging must be reusable or recyclable by 2024, with reduced material complexity and increased recycled content."},
    {"query": "How does the EU Biodiversity Strategy impact exporters?", "answer": "Exporters must ensure their products do not contribute to deforestation or biodiversity loss and comply with due diligence laws."}
]
```

**Evaluation Agent:** Evaluates the generated answer

```python
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score

class EvaluationAgent:
    def __init__(self, gold_qa_dict, similarity_threshold=0.85):
        """
        Initialize the Evaluation Agent with a cosine similarity-based approach.

        Args:
            gold_qa_dict (list): A list of dictionaries containing gold Q&A where each
                                  dictionary has keys "query" and "answer".
            similarity_threshold (float): Minimum cosine similarity score to accept an answer
                                           without human review (default is 0.85).
        """
        self.gold_qa_dict = gold_qa_dict
        self.similarity_threshold = similarity_threshold

    def _tokenize_text(self, text):
        """
        Tokenize the text by splitting it into words and converting to lowercase.

        Args:
            text (str): The text to tokenize.

        Returns:
            list: List of tokens (words).
        """
        return text.lower().split()

    def _vectorize_text(self, text):
        """
        Convert tokenized text into a term frequency (TF) vector.

        Args:
            text (str): The text to vectorize.

        Returns:
            dict: Term frequency (TF) vector.
        """
        tokens = self._tokenize_text(text)
        return Counter(tokens)

    def _cosine_similarity(self, vec1, vec2):
        """
        Calculate cosine similarity between two term frequency vectors.

        Args:
            vec1 (dict): Term frequency vector of the first text.
            vec2 (dict): Term frequency vector of the second text.

        Returns:
            float: Cosine similarity score between 0 and 1.
        """
        # Convert term frequency vectors to sorted lists of word counts
        all_tokens = set(vec1.keys()).union(set(vec2.keys()))
        vec1_list = [vec1.get(token, 0) for token in all_tokens]
        vec2_list = [vec2.get(token, 0) for token in all_tokens]

        # Compute cosine similarity
        return cosine_similarity([vec1_list], [vec2_list])[0][0]

    def _calculate_f1_score(self, generated_answer, gold_answer):
        """
        Calculate F1 score based on token overlap between generated and gold answers.

        Args:
            generated_answer (str): The answer generated by the system.
            gold_answer (str): The gold standard answer.

        Returns:
            float: F1 score based on token overlap.
        """
        gen_tokens = set(self._tokenize_text(generated_answer))
        gold_tokens = set(self._tokenize_text(gold_answer))

        # Calculate Precision and Recall
        precision = len(gen_tokens & gold_tokens) / len(gen_tokens) if len(gen_tokens) > 0 else 0
        recall = len(gen_tokens & gold_tokens) / len(gold_tokens) if len(gold_tokens) > 0 else 0

        # Calculate F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1

    def evaluate_answer(self, generated_answer, query):
        """
        Evaluate the generated answer using multiple metrics including F1 score, Precision@1, and cosine similarity.

        Args:
            generated_answer (str): The answer generated by the system.
            query (str): The user query to evaluate.

        Returns:
            dict: Evaluation results with various metrics.
        """
        # Normalize query to lowercase and strip extra spaces
        normalized_query = query.strip().lower()

        # Check if the normalized query exists in the gold QA list
        gold_answer = None
        for qa in self.gold_qa_dict:
            gold_query = qa["query"].strip().lower()
            if normalized_query == gold_query:
                gold_answer = qa["answer"]
                break

        if not gold_answer:
            return {"error": "No Gold Standard: The query is not in the gold Q&A dictionary."}

        # Vectorize both the generated answer and the gold standard answer
        gen_vec = self._vectorize_text(generated_answer)
        gold_vec = self._vectorize_text(gold_answer)

        # Calculate cosine similarity between the vectors
        cosine_sim = self._cosine_similarity(gen_vec, gold_vec)

        # Calculate F1 Score (overlap) based on tokenized text
        f1 = self._calculate_f1_score(generated_answer, gold_answer)

        # Evaluate based on the similarity score
        semantic_match = cosine_sim >= self.similarity_threshold
        precision_at_1 = 1 if semantic_match else 0

        # Human review only if the similarity score is below the threshold
        human_review_needed = cosine_sim < self.similarity_threshold

        # Return a dictionary with the evaluation results
        return {
            "cosine_similarity": cosine_sim,
            "f1_score": f1,
            "precision_at_1": precision_at_1,
            "semantic_match": semantic_match,
            "human_review_needed": human_review_needed,
            "generated_answer": generated_answer,
            "gold_answer": gold_answer
        }


# Example Usage

# Define the gold Q&A dictionary as a list of dictionaries
gold_qa_dict = [
    {"query": "What is the European Green Deal (EGD)?", "answer":
     "The EGD is the EU’s strategy to reach net zero greenhouse gas emissions by 2050 while achieving sustainable economic growth. It covers policies across sectors like agriculture, energy, and manufacturing to ensure products meet higher sustainability standards."},
    {"query": "What is the Farm to Fork strategy (F2F)?", "answer":
     "The F2F strategy is part of the European Green Deal, focusing on making the EU’s food system fair, healthy, and environmentally friendly. It targets reducing pesticide use, nutrient loss, and promoting organic farming."}
]

# Initialize the evaluation agent with the gold Q&A dictionary
evaluation_agent = EvaluationAgent(gold_qa_dict, similarity_threshold=0.85)

# Assume `generated_answer` is the answer from the system and `user_question` is the query
generated_answer = "The F2F strategy is part of the EGD, focusing on making the EU’s food system fair, healthy, and environmentally friendly. It targets reducing pesticide use, nutrient loss, and promoting organic farming."
user_question = "What is the Farm to Fork strategy (F2F)?"

# Evaluate the generated answer
evaluation_result = evaluation_agent.evaluate_answer(generated_answer, user_question)

# Print the evaluation result
print(f"Cosine Similarity: {evaluation_result['cosine_similarity']:.2f}")
print(f"F1 Score (Overlap): {evaluation_result['f1_score']:.2f}")
print(f"Precision@1: {evaluation_result['precision_at_1']}")
print(f"Semantic Match: {evaluation_result['semantic_match']}")
print(f"Human Review Needed: {evaluation_result['human_review_needed']}")
print(f"Generated Answer: {evaluation_result['generated_answer']}")
print(f"Gold Answer: {evaluation_result['gold_answer']}")
```

```text
Cosine Similarity: 0.95
F1 Score (Overlap): 0.93
Precision@1: 1
Semantic Match: True
Human Review Needed: False
Generated Answer: The F2F strategy is part of the EGD, focusing on making the EU’s food system fair, healthy, and environmentally friendly. It targets reducing pesticide use, nutrient loss, and promoting organic farming.
Gold Answer: The F2F strategy is part of the European Green Deal, focusing on making the EU’s food system fair, healthy, and environmentally friendly. It targets reducing pesticide use, nutrient loss, and promoting organic farming.
```

# **RelevanceSummarySystem Class**

Brings together all the agents. There is also a rephraser function, that rephrases the user query for better retrieval accuracy

```python
import os
import requests

class RelevanceSummarizationSystem:
    def __init__(self, retriever_agent, summarizer_agent, evaluation_agent, relevance_threshold=0.6, openai_api_key=None):
        """
        Initialize the Relevance Summarization System.
        """
        self.retriever_agent = retriever_agent
        self.summarizer_agent = summarizer_agent
        self.evaluation_agent = evaluation_agent
        self.relevance_threshold = relevance_threshold
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required for rephrasing queries.")

    def _send_openai_request(self, prompt: str, model="gpt-4o-mini", temperature=0.7, max_tokens=150):
        """
        Helper function to send a request to OpenAI's API and handle the response.
        """
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except requests.exceptions.RequestException as e:
            print(f"❌ Error during API request: {e}")
            return None

    def rephrase_query(self, query: str) -> str:
        """
        Rephrase the query using OpenAI's API to improve retrieval accuracy.
        """
        prompt = f"You are a rephrasing expert. Rephrase the following question to make it clearer and more likely to retrieve relevant information: {query}"
        rephrased_query = self._send_openai_request(prompt, model="gpt-4o-mini", max_tokens=60)

        if rephrased_query:
            print(f"🔄 Rephrased query: {rephrased_query}")
            return rephrased_query
        return query  # Fallback to the original query if rephrasing fails

    def process_query(self, query: str, top_k: int = 3):
        """
        Process a user query by retrieving relevant chunks and summarizing them.
        """
        print(f"🔍 Processing query: {query}\n")

        # Step 1: Rephrase the query
        rephrased_query = self.rephrase_query(query)

        # Step 2: Retrieve relevant chunks for both original and rephrased query
        try:
            original_chunks = self.retriever_agent.retrieve_relevant_chunks(query, top_k=top_k)
            rephrased_chunks = self.retriever_agent.retrieve_relevant_chunks(rephrased_query, top_k=top_k)
        except Exception as e:
            print(f"❌ Error during retrieval: {e}")
            return "An error occurred while processing your query. Please try again later."

        # Merge both sets of retrieved chunks
        all_chunks = sorted(original_chunks + rephrased_chunks, key=lambda x: x["combined_score"], reverse=True)

        if not all_chunks:
            print("⚠️ No relevant chunks found.\n")
            return "I don't know the answer to this question. Can you try rephrasing your question and try again?"

        # Step 3: Check relevance of the top chunk
        top_relevance = all_chunks[0]["combined_score"]
        print(f"📊 Top relevance score: {top_relevance:.2f}")

        if top_relevance < self.relevance_threshold:
            print(f"⚠️ Relevance score too low (Score: {top_relevance:.2f}).\n")
            return "I don't know the answer to this question. Can you try rephrasing your question and try again?"

        # Step 4: Summarize the retrieved chunks
        try:
            summary = self.summarizer_agent.summarize_retrieved_chunks(all_chunks, query)
        except Exception as e:
            print(f"❌ Error during summarization: {e}")
            return "An error occurred while summarizing the information. Please try again later."

        # Step 5: Evaluate the answer
        evaluation_result = self.evaluation_agent.evaluate_answer(summary, query)

        # Print the concise output
        print(f"📝 Evaluation Results: {evaluation_result}\n")

        # Return only the final summary and evaluation results as output
        return summary.strip(), evaluation_result
```

# **Example Usage**

Try executing the code to type your question

```python
# Example Usage:

# Initialize the Evaluation Agent
evaluation_agent = EvaluationAgent(gold_qa_dict)

# Initialize the RelevanceSummarizationSystem with the retriever, summarizer, and evaluation agent
relevance_system = RelevanceSummarizationSystem(
    retriever_agent=retriever_agent,  # Assuming this is already defined
    summarizer_agent=summarizer_agent,  # Assuming this is already defined
    evaluation_agent=evaluation_agent,
    relevance_threshold=0.6
)

# Take user input (query)
user_question = input("Enter your question: ")  # User-provided query

# Process the user query and get the response
final_summary, evaluation_results = relevance_system.process_query(user_question, top_k=3)

# Print the result
print("\nResponse:")
print(final_summary)  # Clean and concise summary
print("\nEvaluation Results:")
print(evaluation_results)  # Evaluation metrics
```

```text
Enter your question: What is the Farm to Fork (F2F) Strategy?
🔍 Processing query: What is the Farm to Fork (F2F) Strategy?

🔄 Rephrased query: What does the Farm to Fork (F2F) Strategy entail?
📊 Top relevance score: 0.95
📝 Summarizer Agent: Summarizing retrieved content with context...
✅ Context-aware summary generated:
The Farm to Fork (F2F) Strategy is an initiative launched by the European Commission on May 20, 2020, aimed at making Europe’s food systems more sustainable and climate-neutral by 2050. It addresses the climate crisis by promoting a fair, healthy, and environmentally-friendly food system for all Europeans, with specific actions to reduce the environmental and climate footprint of food production and reverse biodiversity loss.

F2F outlines five key targets to be achieved by 2030:

1. Reduce the use and risk of chemical pesticides by 50%.
2. Reduce nutrient losses by at least 50%.
3. Reduce fertilizer use by at least 20%.
4. Cut sales of antibiotics for farm animals by 50%.
5. Increase organic farming area to at least 25% of total arable land.

The strategy emphasizes the importance of sustainable practices such as precision agriculture and aims to ensure fair prices for food producers while promoting affordable, healthy food for all citizens. It also includes initiatives to reduce food waste, enhance consumer information regarding food sources and environmental impacts, and protect ecosystems. Overall, the F2F Strategy is a comprehensive approach to transforming the European food system in alignment with sustainability goals.

📝 Evaluation Results: {'error': 'No Gold Standard: The query is not in the gold Q&A dictionary.'}


Response:
The Farm to Fork (F2F) Strategy is an initiative launched by the European Commission on May 20, 2020, aimed at making Europe’s food systems more sustainable and climate-neutral by 2050. It addresses the climate crisis by promoting a fair, healthy, and environmentally-friendly food system for all Europeans, with specific actions to reduce the environmental and climate footprint of food production and reverse biodiversity loss.

F2F outlines five key targets to be achieved by 2030:

1. Reduce the use and risk of chemical pesticides by 50%.
2. Reduce nutrient losses by at least 50%.
3. Reduce fertilizer use by at least 20%.
4. Cut sales of antibiotics for farm animals by 50%.
5. Increase organic farming area to at least 25% of total arable land.

The strategy emphasizes the importance of sustainable practices such as precision agriculture and aims to ensure fair prices for food producers while promoting affordable, healthy food for all citizens. It also includes initiatives to reduce food waste, enhance consumer information regarding food sources and environmental impacts, and protect ecosystems. Overall, the F2F Strategy is a comprehensive approach to transforming the European food system in alignment with sustainability goals.

Evaluation Results:
{'error': 'No Gold Standard: The query is not in the gold Q&A dictionary.'}
```

![](https://europe-west1-genai-agents-views-tracker.cloudfunctions.net/genai-agents-tracker?notebook=all-agents-tutorials--eu-green-compliance-faq-bot)
