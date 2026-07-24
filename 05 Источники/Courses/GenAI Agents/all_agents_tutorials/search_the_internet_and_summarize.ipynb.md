---
title: "Search and Summarize: AI-Powered Web Research Tool"
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: bd681451b254ac1a790e947b581d3997ab35013d
---

> [!note] Original source material
> This page preserves [`all_agents_tutorials/search_the_internet_and_summarize.ipynb`](https://github.com/NirDiamant/GenAI_Agents/blob/bd681451b254ac1a790e947b581d3997ab35013d/all_agents_tutorials/search_the_internet_and_summarize.ipynb) from
> *GenAI Agents* at commit `bd681451b254ac1a790e947b581d3997ab35013d`. License:
> [Custom non-commercial license](https://github.com/NirDiamant/GenAI_Agents/blob/bd681451b254ac1a790e947b581d3997ab35013d/LICENSE). Bookvar changed only
> publication markup, link paths, and characters required for safe rendering.



## Overview
This Jupyter notebook implements an intelligent web research assistant that combines web search capabilities with AI-powered summarization. It automates the process of gathering information from the internet and distilling it into concise, relevant summaries, enhancing the efficiency of online research tasks.

## Motivation
In the age of information overload, efficiently extracting relevant knowledge from the vast expanse of the internet is increasingly challenging. This tool addresses several key pain points:

1. Time consumption in manual web searches
2. Information overload from multiple sources
3. Difficulty in quickly grasping key points from lengthy articles
4. Need for focused research on specific websites

By automating the search and summarization process, this tool aims to significantly reduce the time and cognitive load associated with web research, allowing users to quickly gain insights on any topic.

## Key Components
The notebook consists of several integral components:

1. **Web Search Module**: Utilizes DuckDuckGo's search API to fetch relevant web pages based on user queries.
2. **Result Parser**: Processes raw search results into a structured format for further analysis.
3. **Text Summarization Engine**: Leverages OpenAI's language models to generate concise summaries of web content.
4. **Integration Layer**: Combines the search and summarization functionalities into a seamless workflow.

## Method Details

### Web Search Process
1. The user provides a search query and optionally specifies a target website.
2. If a specific site is given, the tool performs two searches:
   a. A site-specific search within the specified domain
   b. A general search excluding the specified site
3. Without a specific site, it conducts a general web search.
4. Search results are parsed to extract snippets, titles, and links.

### Summarization Approach
1. For each search result, the tool extracts the relevant text content.
2. The extracted text is sent to the AI model with a prompt requesting a concise summary.
3. The AI generates a summary in the form of 1-2 bullet points, capturing the key information.
4. Summaries are compiled along with their sources (title and link).

### Integration and Output
1. The tool combines the web search and summarization processes into a single function call.
2. It returns a formatted output containing summaries from multiple sources, each clearly attributed.
3. The output is designed to provide a quick overview of the topic, with links to full sources for further reading.

## Conclusion
This notebook demonstrates the power of combining web search capabilities with AI-driven summarization. It offers a practical solution for rapid information gathering and synthesis, applicable in various domains such as research, journalism, business intelligence, and general knowledge acquisition. By automating the tedious aspects of web research, it allows users to focus on higher-level analysis and decision-making based on quickly acquired, relevant information.

The modular design of this tool also allows for future enhancements, such as integration with different search engines, implementation of more advanced summarization techniques, or adaptation to specific domain knowledge requirements.

## Import Dependencies

This cell imports all necessary libraries and sets up the environment.

```python
import os
import requests
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Tuple, Optional
import re
import nltk
from dotenv import load_dotenv

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load environment variables
load_dotenv()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
```

## Initialize DuckDuckGo Search

This cell initializes the DuckDuckGo search tool.

```python
search = DuckDuckGoSearchResults()
```

## Define Data Models

This cell defines the data model for text summarization.

```python
class SummarizeText(BaseModel):
    """Model for text to be summarized."""
    text: str = Field(..., title="Text to summarize", description="The text to be summarized")
```

## Helper Functions

This section contains helper functions for parsing search results and performing web searches.

```python
def parse_search_results(results_string: str) -> List[dict]:
    """Parse a string representation of search results into a list of dictionaries."""
    results = []
    entries = results_string.split(', snippet: ')
    for entry in entries[1:]:  # Skip the first split as it's empty
        parts = entry.split(', title: ')
        if len(parts) == 2:
            snippet = parts[0]
            title_link = parts[1].split(', link: ')
            if len(title_link) == 2:
                title, link = title_link
                results.append({
                    'snippet': snippet,
                    'title': title,
                    'link': link
                })
    return results


def fetch_article_content(url: str, fallback_snippet: str = '', max_chars: int = 3000) -> str:
    """Fetch the actual article content from a URL.

    Falls back to the search result snippet if the URL cannot be fetched
    (e.g. paywalled content, network errors, or non-HTML responses).
    """
    if not url:
        return fallback_snippet
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; research-bot/1.0)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        # Skip non-HTML responses (PDFs, images, binaries) since BeautifulSoup
        # would either yield garbage or a large blob that wastes LLM tokens.
        content_type = response.headers.get('Content-Type', '').lower()
        if 'html' not in content_type:
            print(f"Skipping non-HTML content from {url} (Content-Type: {content_type or 'unknown'}).")
            return fallback_snippet
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove non-content elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        if text:
            return text[:max_chars]
        return fallback_snippet
    except Exception as e:
        print(f"Could not fetch content from {url}: {e}. Using snippet instead.")
        return fallback_snippet


def perform_web_search(query: str, specific_site: Optional[str] = None) -> Tuple[List[str], List[Tuple[str, str]​]]:
    """Perform a web search based on a query, optionally including a specific website."""
    try:
        if specific_site:
            specific_query = f"site:{specific_site} {query}"
            print(f"Searching for: {specific_query}")
            specific_results = search.invoke(specific_query)
            print(f"Specific search results: {specific_results}")
            specific_parsed = parse_search_results(specific_results)

            general_query = f"-site:{specific_site} {query}"
            print(f"Searching for: {general_query}")
            general_results = search.invoke(general_query)
            print(f"General search results: {general_results}")
            general_parsed = parse_search_results(general_results)

            combined_results = (specific_parsed + general_parsed)[:3]
        else:
            print(f"Searching for: {query}")
            web_results = search.invoke(query)
            print(f"Web results: {web_results}")
            combined_results = parse_search_results(web_results)[:3]

        # Fetch actual article content from each URL; fall back to snippet if unavailable
        web_knowledge = [
            fetch_article_content(result.get('link', ''), fallback_snippet=result.get('snippet', ''))
            for result in combined_results
        ]
        sources = [(result.get('title', 'Untitled'), result.get('link', '')) for result in combined_results]

        print(f"Processed sources: {sources}")
        return web_knowledge, sources
    except Exception as e:
        print(f"Error in perform_web_search: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], []
```

## Text Summarization Function

This cell defines the function to summarize text using OpenAI's language model.

```python
def summarize_text(text: str, source: Tuple[str, str]) -> str:
    """Summarize the given text using OpenAI's language model."""
    try:
        llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")
        prompt_template = "Please summarize the following text in 1-2 bullet points:\n\n{text}\n\nSummary:"
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["text"],
        )
        summary_chain = prompt | llm
        input_data = {"text": text}
        summary = summary_chain.invoke(input_data)

        summary_content = summary.content if hasattr(summary, 'content') else str(summary)

        formatted_summary = f"Source: {source[0]} ({source[1]})\n{summary_content.strip()}\n"
        return formatted_summary
    except Exception as e:
        print(f"Error in summarize_text: {str(e)}")
        return ""
```

## Main Search and Summarize Function

This cell defines the main function that combines web search and text summarization.

```python
def search_summarize(query: str, specific_site: Optional[str] = None) -> str:
    """Perform a web search and summarize the results."""
    web_knowledge, sources = perform_web_search(query, specific_site)

    if not web_knowledge or not sources:
        print("No web knowledge or sources found.")
        return ""

    summaries = [summarize_text(knowledge, source) for knowledge, source in zip(web_knowledge, sources) if summarize_text(knowledge, source)]

    combined_summary = "\n".join(summaries)
    return combined_summary
```

## Example Usage

This cell demonstrates how to use the search_summarize function.

```python
query = "What are the latest advancements in artificial intelligence?"
specific_site = "https://www.nature.com"  # Optional: specify a site or set to None
result = search_summarize(query, specific_site)
print(f"Summary of latest advancements in AI (including information from {specific_site if specific_site else 'various sources'}):")
print(result)
```

```text
Searching for: site:https://www.nature.com What are the latest advancements in artificial intelligence?
Specific search results: snippet: The latest advancements in generative artificial intelligence (AI) models have enabled the creation of realistic representations learned from vast amounts of data., title: Place identity: a generative AI's perspective, link: https://www.nature.com/articles/s41599-024-03645-7, snippet: Powered by a large language model (LLM) and trained on much of the text published on the Internet, the artificial intelligence (AI) chatbot, created by OpenAI in San Francisco, California, makes ..., title: Chatbots in science: What can ChatGPT do for you? - Nature, link: https://www.nature.com/articles/d41586-024-02630-z, snippet: Artificial-intelligence tools are transforming data-driven science — better ethical standards and more robust data curation are needed to fuel the boom and prevent a bust. NEWS FEATURE, title: Science and the new age of AI - Nature, link: https://www.nature.com/immersive/d41586-023-03017-2/index.html, snippet: Artificial intelligence (AI) systems, such as the chatbot ChatGPT, have become so advanced that they now very nearly match or exceed human performance in tasks including reading comprehension ..., title: AI now beats humans at basic tasks — new benchmarks are ... - Nature, link: https://www.nature.com/articles/d41586-024-01087-4
Searching for: -site:https://www.nature.com What are the latest advancements in artificial intelligence?
General search results: snippet: The web page does not mention any recent technological breakthrough in artificial intelligence, but it predicts four hot trends for 2024, such as customized chatbots, generative video, and AI ..., title: What's next for AI in 2024 | MIT Technology Review, link: https://www.technologyreview.com/2024/01/04/1086046/whats-next-for-ai-in-2024/, snippet: How generative AI tools like ChatGPT changed the internet and our daily interactions with it. Learn about the latest developments and challenges of AI in 2024, from chatbots to image-making models., title: AI for everything: 10 Breakthrough Technologies 2024, link: https://www.technologyreview.com/2024/01/08/1085096/artificial-intelligence-generative-ai-chatgpt-open-ai-breakthrough-technologies, snippet: As we approach 2024, experts foresee a blossoming interest in AI ethics education and a heightened prioritization of ethical considerations within AI research and development realms. 2. Augmented ..., title: The 5 Biggest Artificial Intelligence Trends For 2024 - Forbes, link: https://www.forbes.com/sites/bernardmarr/2023/11/01/the-top-5-artificial-intelligence-trends-for-2024/, snippet: Learn how generative AI will evolve in the next year, from more realistic expectations and multimodal models to smaller language models and open source advancements. Discover the challenges and opportunities for enterprises and users in the AI landscape., title: The Top Artificial Intelligence Trends | IBM, link: https://www.ibm.com/think/insights/artificial-intelligence-trends
Processed web_knowledge: ['Powered by a large language model (LLM) and trained on much of the text published on the Internet, the artificial intelligence (AI) chatbot, created by OpenAI in San Francisco, California, makes ...', 'Artificial-intelligence tools are transforming data-driven science — better ethical standards and more robust data curation are needed to fuel the boom and prevent a bust. NEWS FEATURE', 'Artificial intelligence (AI) systems, such as the chatbot ChatGPT, have become so advanced that they now very nearly match or exceed human performance in tasks including reading comprehension ...']
Processed sources: [('Chatbots in science: What can ChatGPT do for you? - Nature', 'https://www.nature.com/articles/d41586-024-02630-z'), ('Science and the new age of AI - Nature', 'https://www.nature.com/immersive/d41586-023-03017-2/index.html'), ('AI now beats humans at basic tasks — new benchmarks are ... - Nature', 'https://www.nature.com/articles/d41586-024-01087-4')]
Summary of latest advancements in AI (including information from https://www.nature.com):
Source: Chatbots in science: What can ChatGPT do for you? - Nature (https://www.nature.com/articles/d41586-024-02630-z)
- OpenAI's AI chatbot, developed in San Francisco, utilizes a large language model trained on extensive internet text.
- The chatbot is designed to generate human-like responses and assist users in various tasks.

Source: Science and the new age of AI - Nature (https://www.nature.com/immersive/d41586-023-03017-2/index.html)
- Artificial intelligence is revolutionizing data-driven science, but there is a need for improved ethical standards and enhanced data curation to support sustainable growth in the field.
- Ensuring robust ethical practices and data management is essential to prevent potential setbacks in the advancement of AI in scientific research.

Source: AI now beats humans at basic tasks — new benchmarks are ... - Nature (https://www.nature.com/articles/d41586-024-01087-4)
- AI systems like ChatGPT are approaching or surpassing human performance in various tasks, including reading comprehension.
- The advancements in AI technology highlight their growing capabilities and effectiveness in processing and understanding information.
```

![](https://europe-west1-genai-agents-views-tracker.cloudfunctions.net/genai-agents-tracker?notebook=all-agents-tutorials--search-the-internet-and-summarize)
