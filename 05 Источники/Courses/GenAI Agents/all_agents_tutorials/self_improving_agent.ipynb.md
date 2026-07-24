---
title: "Self-Improving Agent Tutorial"
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: bd681451b254ac1a790e947b581d3997ab35013d
---

> [!note] Original source material
> This page preserves [`all_agents_tutorials/self_improving_agent.ipynb`](https://github.com/NirDiamant/GenAI_Agents/blob/bd681451b254ac1a790e947b581d3997ab35013d/all_agents_tutorials/self_improving_agent.ipynb) from
> *GenAI Agents* at commit `bd681451b254ac1a790e947b581d3997ab35013d`. License:
> [Custom non-commercial license](https://github.com/NirDiamant/GenAI_Agents/blob/bd681451b254ac1a790e947b581d3997ab35013d/LICENSE). Bookvar changed only
> publication markup, link paths, and characters required for safe rendering.



## Overview
This tutorial demonstrates the implementation of a Self-Improving Agent using LangChain, a framework for developing applications powered by language models. The agent is designed to engage in conversations, learn from its interactions, and continuously improve its performance over time.

## Motivation
As AI systems become more integrated into our daily lives, there's a growing need for agents that can adapt and improve based on their interactions. This self-improving agent serves as a practical example of how we can create AI systems that don't just rely on their initial training, but continue to evolve and enhance their capabilities through ongoing interactions.

## Key Components

1. **Language Model**: The core of the agent, responsible for generating responses and processing information.
2. **Chat History Management**: Keeps track of conversations for context and learning.
3. **Response Generation**: Produces relevant replies to user inputs.
4. **Reflection Mechanism**: Analyzes past interactions to identify areas for improvement.
5. **Learning System**: Incorporates insights from reflection to enhance future performance.

## Method Details

### Initialization
The agent is initialized with a language model, a conversation store, and a system for managing prompts and chains. This setup allows the agent to maintain context across multiple interactions and sessions.

### Response Generation
When the agent receives input, it considers the current conversation history and any recent insights gained from learning. This context-aware approach allows for more coherent and improving responses over time.

### Reflection Process
After a series of interactions, the agent reflects on its performance. It analyzes the conversation history to identify patterns, potential improvements, and areas where it could have provided better responses.

### Learning Mechanism
Based on the reflections, the agent generates learning points. These are concise summaries of how it can improve, which are then incorporated into its knowledge base and decision-making process for future interactions.

### Continuous Improvement Loop
The cycle of interaction, reflection, and learning creates a feedback loop that allows the agent to continuously refine its responses and adapt to different conversation styles and topics.

## Conclusion
This Self-Improving Agent demonstrates a practical implementation of an AI system that can learn and adapt from its interactions. By combining the power of large language models with mechanisms for reflection and learning, we create an agent that not only provides responses but also improves its capabilities over time.

This approach opens up exciting possibilities for creating more dynamic and adaptable AI assistants, chatbots, and other conversational AI applications. As we continue to refine these techniques, we move closer to AI systems that can truly learn and grow from their experiences, much like humans do.

## Imports and Setup

First, we'll import the necessary libraries and load our environment variables.

```python
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
```

```text
True
```

## Helper Functions

We'll define helper functions for each capability of our agent.

### Chat History Management

```python
def get_chat_history(store, session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
```

### Response Generation

```python
def generate_response(chain_with_history, human_input: str, session_id: str, insights: str):
    response = chain_with_history.invoke(
        {"input": human_input, "insights": insights},
        config={"configurable": {"session_id": session_id}}
    )
    return response.content
```

### Reflection

```python
def reflect(llm, store, session_id: str):
    reflection_prompt = ChatPromptTemplate.from_messages([
        ("system", "Based on the following conversation history, provide insights on how to improve responses:"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Generate insights for improvement:")
    ])
    reflection_chain = reflection_prompt | llm
    history = get_chat_history(store, session_id)
    reflection_response = reflection_chain.invoke({"history": history.messages})
    return reflection_response.content
```

### Learning

```python
def learn(llm, store, session_id: str, insights: str):
    learning_prompt = ChatPromptTemplate.from_messages([
        ("system", "Based on these insights, update the agent's knowledge and behavior:"),
        ("human", "{insights}"),
        ("human", "Summarize the key points to remember:")
    ])
    learning_chain = learning_prompt | llm
    learned_points = learning_chain.invoke({"insights": insights}).content
    get_chat_history(store, session_id).add_ai_message(f"[SYSTEM] Agent learned: {learned_points}")
    return learned_points
```

## Self-Improving Agent Class

Now we'll define our `SelfImprovingAgent` class that uses these functions.

```python
class SelfImprovingAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0.7)
        self.store = {}
        self.insights = ""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a self-improving AI assistant. Learn from your interactions and improve your performance over time."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
            ("system", "Recent insights for improvement: {insights}")
        ])

        self.chain = self.prompt | self.llm
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            lambda session_id: get_chat_history(self.store, session_id),
            input_messages_key="input",
            history_messages_key="history"
        )

    def respond(self, human_input: str, session_id: str):
        return generate_response(self.chain_with_history, human_input, session_id, self.insights)

    def reflect(self, session_id: str):
        self.insights = reflect(self.llm, self.store, session_id)
        return self.insights

    def learn(self, session_id: str):
        self.reflect(session_id)
        return learn(self.llm, self.store, session_id, self.insights)
```

## Example Usage

Let's create an instance of our agent and interact with it to demonstrate its self-improving capabilities.

```python
agent = SelfImprovingAgent()
session_id = "user_123"

# Interaction 1
print("AI:", agent.respond("What's the capital of France?", session_id))

# Interaction 2
print("AI:", agent.respond("Can you tell me more about its history?", session_id))

# Learn and improve
print("\nReflecting and learning...")
learned = agent.learn(session_id)
print("Learned:", learned)

# Interaction 3 (potentially improved based on learning)
print("\nAI:", agent.respond("What's a famous landmark in this city?", session_id))

# Interaction 4 (to demonstrate continued improvement)
print("AI:", agent.respond("What's another interesting fact about this city?", session_id))
```

```text
AI: The capital of France is Paris.
AI: Paris has a rich and complex history that spans thousands of years. Its story begins around the 3rd century BC with the Parisii, a Gallic tribe that settled on the banks of the River Seine. The Romans conquered the Parisii in 52 BC, establishing a town known as Lutetia, which flourished as a Roman city.

In the Middle Ages, Paris emerged as a center of learning and the arts, home to one of the first universities in Europe, the University of Paris (founded around 1150). The city expanded its influence and infrastructure, including the construction of the iconic Notre-Dame Cathedral, which began in the 12th century.

The Renaissance era brought further development, but it was also a period of religious strife, highlighted by the St. Bartholomew's Day Massacre in 1572, where thousands of Huguenots (French Protestants) were killed. The 17th and 18th centuries were marked by the opulence of the Sun King, Louis XIV, and the construction of architectural marvels such as the Palace of Versailles.

However, wealth disparities led to unrest, culminating in the French Revolution in 1789. Paris was the revolution's heart, witnessing significant events like the storming of the Bastille and the Reign of Terror. The 19th century saw the city transform under Napoleon Bonaparte and later, under Napoleon III and Baron Haussmann, who redesigned Paris with its wide boulevards, parks, and the iconic landmarks we recognize today.

The 20th century was tumultuous, with Paris enduring two world wars, including the Nazi occupation during World War II. The post-war years were a time of reconstruction and cultural renaissance, solidifying Paris's status as a global hub of art, fashion, and gastronomy.

Today, Paris is celebrated for its vibrant culture, historical monuments, and significant contributions to art, science, and philosophy. It remains a symbol of beauty, resilience, and the enduring spirit of enlightenment and innovation.

Reflecting and learning...
Learned: To enhance responses, especially for historical overviews, keep these key points in mind:

1. **Prioritize Key Information**: Focus on the most significant events or aspects to keep the narrative engaging and digestible.

2. **Incorporate Storytelling Elements**: Use storytelling to create compelling narratives, setting scenes, and highlighting the human aspects of history.

3. **Use Clear, Vivid Language**: Employ descriptive language to paint a vivid picture of historical events and transformations.

4. **Interactive Elements**: Suggest additional resources like further reading or virtual tours to provide a more in-depth understanding, where applicable.

5. **Personalize Responses**: Tailor responses to the user's interests to increase relevance and engagement.

6. **Inclusion of Lesser-Known Facts**: Intersperse lesser-known facts or anecdotes to pique interest and provide a unique perspective.

7. **Clarity and Structure**: Organize the response clearly, possibly with subheadings or a chronological approach, to enhance understandability.

8. **Encourage Interaction**: Conclude with an invitation for further questions, fostering ongoing engagement and tailored information sharing.

By applying these principles, responses can be made more engaging, informative, and satisfying for the reader.

AI: One of the most famous landmarks in Paris is the Eiffel Tower. Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially criticized by some of France's leading artists and intellectuals for its design but has since become a global cultural icon of France and one of the most recognizable structures in the world.

Standing at approximately 324 meters (1,063 feet) tall, the Eiffel Tower was the tallest man-made structure in the world until the completion of the Chrysler Building in New York in 1930. It was designed by the French engineer Gustave Eiffel's company. The tower is made of iron and weighs about 10,000 tonnes. Despite its initial intended temporary presence, it was saved from dismantling due to its value as a radiotelegraph station and now attracts millions of visitors each year.

The Eiffel Tower has three levels for visitors. Tickets can be purchased to ascend by stairs or elevators to the first and second levels. The journey to the top level offers a breathtaking panoramic view of Paris, making it one of the most visited monuments in the world.

Beyond its architectural and engineering significance, the Eiffel Tower has become a symbol of French creativity and ingenuity, representing the spirit of progress and the beauty of Paris to the world. Whether seen up close or from one of the many beautiful vantage points in the city, the Eiffel Tower continues to awe visitors with its imposing structure and the story of its creation.

For those interested in exploring more, virtual tours or a visit to the official Eiffel Tower website can offer deeper insights into its history, construction, and the experience of visiting this iconic monument.
AI: Another interesting fact about Paris is its nickname, "The City of Light" ("La Ville Lumière"). This name originated in the 17th century, during the reign of King Louis XIV, when the city began to replace its dark, narrow streets with wide boulevards and installed thousands of gas lamps to illuminate them. This transformation made Paris one of the first major European cities to use street lighting extensively, enhancing its beauty and safety at night.

However, the nickname also refers to Paris's leading role during the Age of Enlightenment, a period in the 18th century characterized by intellectual, cultural, and scientific advancements. Paris became a center for education, ideas, and philosophical thought, attracting scholars, artists, and writers from all over Europe. It was during this time that the city truly became a beacon of light, symbolizing hope, progress, and innovation.

Today, the moniker "City of Light" aptly reflects both the literal illumination of Paris's streets and monuments, as well as its ongoing influence as a center for culture, art, fashion, and gastronomy. The sparkling lights of the Eiffel Tower at night and the illuminated bridges over the Seine River continue to enchant visitors, embodying the city's enduring charm and its historical significance as a beacon of enlightenment and progress.
```

![](https://europe-west1-genai-agents-views-tracker.cloudfunctions.net/genai-agents-tracker?notebook=all-agents-tutorials--self-improving-agent)
