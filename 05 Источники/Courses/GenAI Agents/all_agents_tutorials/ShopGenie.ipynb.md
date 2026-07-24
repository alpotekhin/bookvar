---
title: "**ShopGenie**"
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: bd681451b254ac1a790e947b581d3997ab35013d
---

> [!note] Original source material
> This page preserves [`all_agents_tutorials/ShopGenie.ipynb`](https://github.com/NirDiamant/GenAI_Agents/blob/bd681451b254ac1a790e947b581d3997ab35013d/all_agents_tutorials/ShopGenie.ipynb) from
> *GenAI Agents* at commit `bd681451b254ac1a790e947b581d3997ab35013d`. License:
> [Custom non-commercial license](https://github.com/NirDiamant/GenAI_Agents/blob/bd681451b254ac1a790e947b581d3997ab35013d/LICENSE). Bookvar changed only
> publication markup, link paths, and characters required for safe rendering.



Redifining the concept of Online shopping customer experience with agentic AI.


---

# Overview
This AI agent , **ShopGenie** , is built on langGraph and aims to provide decisive shopping experience to all people using the power of LLM.It uses tavily for web search , and llama-3.1-70B-versatile model through groq. This ai agent is made using totally open source technologies.



---

# Motivation
This AI agent is made to assist a customer to get best desired product specifically tailoured for his needs and wants. Eventhough if do not has any expertise in that particular field of whose product he wants to buy, still using the power of **ShopGenie** he could land for the best suited product for him.

---

#Key Features:
- **Tavily** for web search.
- **llama-3.1-70B** for arranging the data into specific schema and comparing products.
- Tells the best product among the searched ones.
- **Youtube API** for providing the review link of the best product for self-satisfaction.
- **SMTP** for sending mail about the best product and its review to the user.

# Important packages
following are the required packeages for this agent to function.

```python
%pip install langchain-groq langgraph tavily-python google-api-python-client langchain-community  beautifulsoup4  > /dev/null 2>&1
```

# Import necessary modules

```python
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from tavily import TavilyClient
from langchain_community.tools import TavilySearchResults
from typing import List, Optional, Dict
from typing_extensions import TypedDict
from googleapiclient.discovery import build
from google.colab import userdata
from IPython.display import Image, display
import getpass
import os
import json
import bs4
from langchain_community.document_loaders import WebBaseLoader
from pydantic import BaseModel, HttpUrl, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import time
```

# Setting environment variables
These are the totally open source technologies used and environment variables which can be changed according to one's needs and availability

```python
groq_api_key = userdata.get('GROQ_API_KEY')
tavily_api_key = userdata.get('TAVILY_API_KEY')
youtube_api_key = userdata.get('YOUTUBE_API_KEY')

#LLM being used in this notebook
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    api_key=groq_api_key,
    temperature=0.5,
)

#Tavily for web search
tavily_client = TavilyClient(api_key=tavily_api_key)

#Youtube api for video search
youtube = build('youtube', 'v3', developerKey=youtube_api_key)
```

# Other Structures

```python


class SpecsComparison(BaseModel):
    processor: str = Field(..., description="Processor type and model, e.g., 'Snapdragon 888'")
    battery: str = Field(..., description="Battery capacity and type, e.g., '4500mAh'")
    camera: str = Field(..., description="Camera specs, e.g., '108MP primary'")
    display: str = Field(..., description="Display type, size, refresh rate, e.g., '6.5 inch OLED, 120Hz'")
    storage: str = Field(..., description="Storage options and expandability, e.g., '128GB, expandable'")

class RatingsComparison(BaseModel):
    overall_rating: float = Field(..., description="Overall rating out of 5, e.g., 4.5")
    performance: float = Field(..., description="Rating for performance out of 5, e.g., 4.7")
    battery_life: float = Field(..., description="Rating for battery life out of 5, e.g., 4.3")
    camera_quality: float = Field(..., description="Rating for camera quality out of 5, e.g., 4.6")
    display_quality: float = Field(..., description="Rating for display quality out of 5, e.g., 4.8")

class Comparison(BaseModel):
    product_name: str = Field(..., description="Name of the product")
    specs_comparison: SpecsComparison
    ratings_comparison: RatingsComparison
    reviews_summary: str = Field(..., description="Summary of key points from user reviews about this product")

class BestProduct(BaseModel):
    product_name: str = Field(..., description="Name of the best product")
    justification: str = Field(..., description="Explanation of why this product is the best choice")

class ProductComparison(BaseModel):
    comparisons: List[Comparison]
    best_product: BestProduct

class Highlights(BaseModel):
    Camera: Optional[str] = None
    Performance: Optional[str] = None
    Display: Optional[str] = None
    Fast_Charging: Optional[str] = None

class SmartphoneReview(BaseModel):
    """A review of a smartphone."""
    title: str = Field(..., description="The title of the smartphone review")
    url: Optional[str] = Field(None, description="The URL of the smartphone review")
    content: Optional[str] = Field(None, description="The main content of the smartphone review")
    pros: Optional[List[str]​] = Field(None, description="The pros of the smartphone")
    cons: Optional[List[str]​] = Field(None, description="The cons of the smartphone")
    highlights: Optional[dict] = Field(None, description="The highlights of the smartphone")
    score: Optional[float] = Field(None, description="The score of the smartphone")

class ListOfSmartphoneReviews(BaseModel):
    """A list of smartphone reviews."""
    reviews: List[SmartphoneReview] = Field(..., description="List of individual smartphone reviews")

class EmailRecommendation(BaseModel):
    subject: str = Field(..., description="The email subject line, designed to capture the recipient's attention.")
    heading: str = Field(..., description="The main heading of the email, introducing the recommended product.")
    justification_line: str = Field(..., description="A concise explanation of why the product is being recommended.")
```

# Main State

```python
class State(TypedDict):
    query: str
    email: str
    products: list[dict]
    product_schema: list[SmartphoneReview]
    blogs_content: Optional[List[dict]​]
    best_product: dict
    comparison: list
    youtube_link: str
```

# Sending Email
This is a complete function of sending mail which takes ShopGenie to next level and speaks of its potential

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google.colab import userdata

def send_email(recipient_email, subject, body):
    """Send an email dynamically using SMTP."""
    # SMTP server configuration
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = userdata.get("GMAIL_USER")
    sender_password = userdata.get("GMAIL_PASS")

    try:
        # Create email content
        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = recipient_email
        message['Subject'] = subject

        # Add the email body
        message.attach(MIMEText(body, 'html'))

        # Connect to the SMTP server
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Start TLS encryption
            server.login(sender_email, sender_password)  # Login to the server
            server.send_message(message)  # Send the email
            print(f"Email sent successfully to {recipient_email}.")

    except Exception as e:
        print(f"Failed to send email: {e}")
```

```python
#Email prompt template
email_template_prompt = """
You are an expert email content writer.

Generate an email recommendation based on the following inputs:
- Product Name: {product_name}
- Justification Line: {justification_line}
- User Query: "{user_query}" (a general idea of the user's interest, such as "a smartphone for photography" or "a premium gaming laptop").

Return your output in the following JSON format:
{format_instructions}

### Input Example:
Product Name: Google Pixel 8 Pro
Justification Line: Praised for its exceptional camera, advanced AI capabilities, and vibrant display.
User Query: a phone with an amazing camera

### Example Output:
{{
  "subject": "Capture Every Moment with Google Pixel 8 Pro",
  "heading": "Discover the Power of the Ultimate Photography Smartphone",
  "justification_line": "Known for its exceptional camera quality, cutting-edge AI features, and vibrant display, the Google Pixel 8 Pro is perfect for photography enthusiasts."
}}

Now generate the email recommendation based on the inputs provided.
"""
```

```python
#email html template
email_html_template = """
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f4f4f4;
            }}
            .email-container {{
                max-width: 600px;
                margin: 20px auto;
                background-color: #ffffff;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                overflow: hidden;
            }}
            .header {{
                background-color: #007BFF;
                color: #ffffff;
                padding: 20px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 24px;
            }}
            .content {{
                padding: 20px;
            }}
            .content h2 {{
                color: #333333;
                font-size: 20px;
                margin-bottom: 10px;
            }}
            .content p {{
                color: #555555;
                font-size: 16px;
                line-height: 1.5;
            }}
            .button {{
                display: inline-block;
                margin-top: 20px;
                background-color: #007BFF;
                color: #ffffff;
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 5px;
            }}
            .footer {{
                text-align: center;
                font-size: 14px;
                color: #999999;
                padding: 10px 20px;
            }}
            .footer a {{
                color: #007BFF;
                text-decoration: none;
            }}
        </style>
    </head>
    <body>
        <div class="email-container">
            <div class="header">
                <h1>{heading}</h1>
            </div>
            <div class="content">
                <h2>Our Top Pick: {product_name}</h2>
                <p>{justification}</p>
                <p>Watch our in-depth review to explore why this phone is the best choice for you:</p>
                <a href="{youtube_link}" class="button" target="_blank">Watch the Review</a>
            </div>
            <div class="footer">
                <p>
                    Want to learn more? Visit our website or follow us for more recommendations.
                    <a href="https://www.youtube.com">Explore Now</a>
                </p>
                <p>&copy; 2024 Smartphone Recommendations, All Rights Reserved.</p>
            </div>
        </div>
    </body>
    </html>
    """
```

# Loading web content
This is the complete where the tavily loading all the content available on the web for that particular search

```python
# Function to load content from a specific URL
def load_blog_content(page_url):
    try:
        # Initialize WebBaseLoader with the URL
        loader = WebBaseLoader(web_paths=[page_url], bs_get_text_kwargs={"separator": " ", "strip": True})
        loaded_content = loader.load()

        # Extract full text from loaded content
        blog_content = " ".join([doc.page_content for doc in loaded_content])  # Assuming the loader returns a list of docs

        # print("Loaded Blog Content:", blog_content)
        return blog_content

    except Exception as e:
        print(f"Error loading blog content from URL {page_url}: {e}")
        return ""
```

# Nodes
These are node which are performing the functiond in the ShopGenie and provides it the base.

```python
#  Node function to search with Tavily and store content
def tavily_search_node(state):
    try:
        # Use the user-provided query from the state
        query = state.get('query', state['query'])
        # Perform the search with Tavily to retrieve multiple blog links
        response = tavily_client.search(query=query, max_results=1)
        if "results" not in response or not response["results"]:
            raise ValueError("No results found for the given query.")
        # Initialize an empty list to store each blog's content
        blogs_content = []
        # Iterate over the search results
        for blog in response['results']:
            blog_url = blog.get("url", "")
            if blog_url:
                # Load and store content from each URL using WebBaseLoader
                content = load_blog_content(blog_url)
                if content:
                  # Append blog details to blogs_content
                  blogs_content.append({
                      "title": blog.get("title", ""),
                      "url": blog_url,
                      "content": content,  # Use loaded content
                      "score": blog.get("score", "")
                  })

        # Store all blog contents in the state
        if len(blogs_content) > 0:

            print("Extracted Blogs Content:", blogs_content)

            return {"blogs_content":blogs_content}
        else:
            raise ValueError("No blogs content found.")

    except tavily.InvalidAPIKeyError:
        print("Error: Invalid Tavily API key. Please verify your key.")
        return {"blogs_content": []}
    except tavily.UsageLimitExceededError:
        print("Error: Tavily usage limit exceeded. Check your plan or limits.")
        return {"blogs_content": []}
    except Exception as e:
        print(f"Error with Tavily API call: {e}")
        return {"blogs_content": []}
#mapping values extarcted from web search
def schema_mapping_node(state: State):
  max_retries = 2  # Maximum number of retries
  wait_time = 60   # Wait time in seconds between retries (1 minute)
  try:
    # Check if "blogs_content" exists in the state and is not empty
    if "blogs_content" in state and state["blogs_content"]:
            # Extract blog content from the state
            blogs_content = state["blogs_content"]
            # Define the prompt
            prompt_template = """
You are a professional assistant tasked with extracting structured information from a blogs.

### Instructions:

1. **Product Details**: For each product mentioned in the blog post, populate the `products` array with structured data for each item, including:
   - `title`: The product name.
   - `url`: Link to the blog post or relevant page.
   - `content`: A concise summary of the product's main features or purpose.
   - `pros`: A list of positive aspects or advantages of the product.if available other wise extract blog content.
   - `cons`: A list of negative aspects or disadvantages.if available other wise extract blog content.
   - `highlights`: A dictionary containing notable features or specifications.if available other wise extract blog content.
   - `score`: A numerical rating score if available; otherwise, use `0.0`.

### Blogs Contents: {blogs_content}

After extracting all information, just return the response in the JSON structure given below. Do not add any extracted information. The JSON should be in a valid structure with no extra characters inside, like Python’s \n.


"""
            # Set up a parser and inject instructions into the prompt template.
            parser = JsonOutputParser(pydantic_object=ListOfSmartphoneReviews)
            # Format the prompt with the full blogs content
            prompt = PromptTemplate(
                template = prompt_template,
                input_variables = ["blogs_content"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            # Retry mechanism to invoke LLM and parse the response
            for attempt in range(1, max_retries + 1):
                # try:
                    # Use LLM to process the prompt and return structured smartphone details
                    chain = prompt | llm | parser  # Invokes LLM with the prepared prompt
                    response = chain.invoke({"blogs_content": blogs_content})

                    # Check if the response contains more than one product in the schema
                    if response.get('products') and len(response['products']) > 1:
                        # If valid, store the structured schema in the state
                        return {"product_schema": response['products']}
                    else:
                        print(f"Attempt {attempt} failed: Product schema has one or fewer products.")

                    # Wait for 1 minute before retrying if not successful and retry limit not reached
                    if attempt < max_retries:
                        time.sleep(wait_time)

                # except Exception as retry_exception:
                #     print(f"Retry {attempt} error: {retry_exception}")
                #     if attempt < max_retries:
                #         time.sleep(wait_time)

            # Return an empty schema if all retries fail
            print("All retry attempts failed to create a valid product schema with more than one product.")
            return {"product_schema": []}
    else:
      # If "blogs_content" is not present or is empty, log and return state unmodified
      print("No blog content available or content is empty; schema extraction skipped.")
      return {"product_schema":[]}

  except Exception as e:
        # Error handling to catch any unexpected issues and log the error message
        print(f"Error occurred during schema extraction: {e}")
        return state
#comparing the products
def product_comparison_node(state: State):
    try:
      # Check if "product_schema" is present in the state and is not empty
      if "product_schema" in state and state["product_schema"]:
        product_schema = state["product_schema"]


        prompt_template = """
1. **List of Products for Comparison (`comparisons`):**
   - Each product should include:
     - **Product Name**: The name of the product (e.g., "Smartphone A").
     - **Specs Comparison**:
       - **Processor**: Type and model of the processor (e.g., "Snapdragon 888").
       - **Battery**: Battery capacity and type (e.g., "4500mAh").
       - **Camera**: Camera specifications (e.g., "108MP primary").
       - **Display**: Display type, size, and refresh rate (e.g., "6.5 inch OLED, 120Hz").
       - **Storage**: Storage options and whether it is expandable (e.g., "128GB, expandable").
     - **Ratings Comparison**:
       - **Overall Rating**: Overall rating out of 5 (e.g., 4.5).
       - **Performance**: Rating for performance out of 5 (e.g., 4.7).
       - **Battery Life**: Rating for battery life out of 5 (e.g., 4.3).
       - **Camera Quality**: Rating for camera quality out of 5 (e.g., 4.6).
       - **Display Quality**: Rating for display quality out of 5 (e.g., 4.8).
     - **Reviews Summary**: Summary of key points from user reviews that highlight the strengths and weaknesses of this product.

2. **Best Product Selection (`best_product`):**
   - **Product Name**: Select the best product among the compared items.
   - **Justification**: Provide a brief explanation of why this product is considered the best choice. This should be based on factors such as balanced performance, high user ratings, advanced specifications, or unique features.

---

### Example Output:

```json
{{
  "comparisons": [
    {{
      "product_name": "Smartphone A",
      "specs_comparison": {{
        "processor": "Snapdragon 888",
        "battery": "4500mAh",
        "camera": "108MP primary",
        "display": "6.5 inch OLED, 120Hz",
        "storage": "128GB, expandable"
      }},
      "ratings_comparison": {{
        "overall_rating": 4.5,
        "performance": 4.7,
        "battery_life": 4.3,
        "camera_quality": 4.6,
        "display_quality": 4.8
      }},
      "reviews_summary": "Highly rated for display quality and camera performance, with a strong processor. Battery life is good but may drain faster with heavy use."
    }},
    {{
      "product_name": "Smartphone B",
      "specs_comparison": {{
        "processor": "Apple A15 Bionic",
        "battery": "4000mAh",
        "camera": "12MP Dual",
        "display": "6.1 inch Super Retina XDR, 60Hz",
        "storage": "256GB, non-expandable"
      }},
      "ratings_comparison": {{
        "overall_rating": 4.6,
        "performance": 4.8,
        "battery_life": 4.1,
        "camera_quality": 4.5,
        "display_quality": 4.7
      }},
      "reviews_summary": "Smooth user experience with excellent performance and display. The battery is slightly smaller but generally sufficient for moderate use."
    }}
  ],
  "best_product": {{
    "product_name": "Smartphone A",
    "justification": "Chosen for its high-quality display, strong camera, and balanced performance that meets most user needs."
  }}
}}

```
Here is the product data to analyze:\n
{product_data}

"""
        parser = JsonOutputParser(pydantic_object=ProductComparison)
      # Format the prompt with the full blogs content
        prompt = PromptTemplate(
        template = prompt_template,
        input_variables = ["product_data"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
        # prompt = prompt_template.format(product_data=json.dumps(state['product_schema']))

        # Use LLM to process the prompt and return structured smartphone details
        chain = prompt | llm | parser  # Invokes LLM with the prepared prompt
        # display(response.content)
        response = chain.invoke({"product_data": json.dumps(state['product_schema'])})

        # print(response['products'])
        display(response)

        return {"comparison": response['comparisons'],"best_product":response['best_product']}


      else:
          # If "product_schema" is missing or empty, log and skip comparison logic
          print("No product schema available; product comparison skipped.")
          return state


    except Exception as e:
        print(f"Error during product comparison: {e}")
        return {"best_product": {}, "comparison_report": "Comparison failed"}
#youtube review
def youtube_review_node(state: State):
    best_product_name = state.get("best_product", {}).get("product_name")

    if not best_product_name:
        print("Skipping YouTube search: No best product found.")
        return {"youtube_link": None}

    try:
        search_response = youtube.search().list(
            q=f"{best_product_name} review",
            part="snippet",
            type="video",
            maxResults=1
        ).execute()

        video_items = search_response.get("items", [])
        if not video_items:
            print("No YouTube videos found for the best product.")
            return {"youtube_link": None}

        video_id = video_items[0]["id"]["videoId"]
        youtube_link = f"https://www.youtube.com/watch?v={video_id}"
        return {"youtube_link": youtube_link}

    except Exception as e:
        print(f"Error during YouTube search: {e}")
        return {"youtube_link": None}
#final display on the UI
def display_node(state: State):
  if "comparison" in state and state['comparison']:


      return {
        "products": state["product_schema"],
        "best_product": state["best_product"],
        "comparison": state["comparison"],
        "youtube_link": state["youtube_link"]
    }
  else:
    print("comparison not available")
#sending email
def send_email_node(state:State):
  if "best_product" in state and state['best_product']:
    user_query = state["query"]
    best_product_name = state["best_product"]["product_name"]
    justification = state["best_product"]["justification"]
    youtube_link = state["youtube_link"]
    recipient_email=state['email']
    parser = JsonOutputParser(pydantic_object=EmailRecommendation)
    prompt = PromptTemplate(
    template=email_template_prompt,
    input_variables=["product_name", "justification_line", "user_query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    response = chain.invoke({"product_name": best_product_name, "justification_line": justification, "user_query": user_query})
    html_content = email_html_template.format(product_name=best_product_name, justification=response["justification_line"], youtube_link=youtube_link,heading=response['heading'])
    send_email(recipient_email,subject=response['subject'],body=html_content)
```

# Workflow
Compiling the LangGraph workflow

```python
# Build the LangGraph
builder = StateGraph(State)
builder.add_node("tavily_search", tavily_search_node)
builder.add_node("schema_mapping", schema_mapping_node)
builder.add_node("product_comparison", product_comparison_node)
builder.add_node("youtube_review", youtube_review_node)
builder.add_node("display", display_node)
builder.add_node("send_email", send_email_node)
# Define edges to control flow between nodes
builder.add_edge(START, "tavily_search")
builder.add_edge("tavily_search", "schema_mapping")
builder.add_edge("schema_mapping", "product_comparison")
builder.add_edge("product_comparison", "youtube_review")
builder.add_edge("youtube_review", "display")
builder.add_edge("display", END)
builder.add_edge("youtube_review","send_email")
builder.add_edge("send_email",END)
```

```text
<langgraph.graph.state.StateGraph at 0x7f9d45b039d0>
```

# Diagram

```python
# Compile and display graph as Mermaid diagram
graph = builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))
```

```text
<IPython.core.display.Image object>
```

![[Assets/Sources/GenAI Agents/all_agents_tutorials/ShopGenie/cell-023-output-01.png]]

# Running the Graph

```python
# Run the LangGraph workflow
initial_state = {"query": "Best smartphones under $1000","email":"asadsher2324@gmail.com"}

for event in graph.stream(input=initial_state,stream_mode="updates"):
  print(event)
```

```text
Extracted Blogs Content: [{'title': 'The Best Phones Under $1000 to Buy Today - NextPit', 'url': 'https://www.nextpit.com/best-smartphones-under-1000', 'content': "The Best Phones Under $1000 to Buy Today Buying Guide Buying Guide Smartphones The best smartphones Best under $400 Best under $200 Wearables The best Bluetooth headphones to buy in 2024 Which Garmin smartwatch is the best for me? The best Apple and Android smartwatches of 2024 Deals Samsung Galaxy S22: Should you buy it now? Buying the iPhone 13? Best OnePlus 11 offer: Where to buy it! Our Favorite Bluetti AC200L Power Station is $1000 Off Apple's M2 MacBook Air with 16 GB RAM for $749 is a Must-Have Laptop Apple's Thinner Watch Series 10 Falls to a New Low for 13% Off Reviews Reviews Smartphone Samsung Galaxy S23 Ultra review Samsung Galaxy S23 review Samsung Galaxy S23+ review Apple iPhone 14 Plus review Wearables OnePlus Buds Pro 2 review Sony WH-1000XM5 review Apple Watch Ultra review Nuki Smart Lock Pro 4.0 Review: Bid Your Keys Goodbye How This Withings Smart Scale Transformed My Understanding of My Body Samsung Galaxy Tab S10+: The Productivity Powerhouse You Pay For News News Apple iPhone Comparison Best iPad Samsung Galaxy S23 Ultra vs S22 Ultra Which Samsung phones will receive Android 13? Smartphone Android 13 iOS 16.4 Beta Wearables Apps Hidden iOS 18 Trick: iPhone Reboots Itself to Boost Anti-Theft Security Top 5 Apps of the Week: Carrion, Pokémon TCG Pocket, and More! One of the Most Famous Video Game Classics is Free This Week How To How To Smarthome How to check the battery status Samsung One UI Top 10 Android 13 gestures Apps Contacts not showing in WhatsApp? Get fast and easy translations on your Android Activate the incognito mode on YouTube How to Use Your Phone as a Wi-Fi Extender How to Install Xiaomi's Super Wallpapers on Compatible Android Smartphones Possible Fix for YouTube's New Update Glitches Topics Topics Smartphones Wearables Apps eMobility Smart Home More Forum Forum Latest forum posts Whatsapp non officiel avec mon numero de telephone The transition from Dalvik to Android Runtime (ART) has significantly enhanced app performance via JIT and AOT compilation. How do these techniques di Using AI to create Images Hello from a Newbie! Waze is not showing the map! Additional Macro Lens for Moto G Fast Hello everyone Gimbooks Pay Unanswered The transition from Dalvik to Android Runtime (ART) has significantly enhanced app performance via JIT and AOT compilation. How do these techniques di Gimbooks Pay [APP] Unique Zipper Lock Screen - Zip Lock New Member Introduction [App] 3D Parallax Wallpaper & Wallpaper 4K PHOTOS OUT OF ORDER ON WHATSAPP Kids Match Game: Play & Learn WhatsApp is pausing the audio when it’s on my ear Quick Links Recent posts Ask a question Post new thread Our forum rules Mods + Admins Hall of Fame Community Guide Search Login Hot topics Android 15 iOS 18 iPhone 16 iPhone 16 Pro Home Smartphone Hardware The Best Sub-$1,000 Smartphones You Can Buy in 2024 7 min read 7 min No comments 0 Nov 11, 2024, 10:46 AM © nextpit Rubens Eishima Writer Which high-end smartphone should you buy for under $1,000 in 2024? To help you choose the most powerful smartphone, the best camera smartphone, or simply a compact option, we have selected for you the best affordable flagships of the moment. Such as the Galaxy S24, the iPhone 16, and the Google Pixel 8 Pro. Table of Contents: The best sub-$1,000 smartphones in 2024 The best Android sub-$1000: Samsung Galaxy S24 The best sub-$1,000 iPhone: Apple iPhone 16 The best camera alternative under $1,000: Googl e Pixel 8\xa0Pro The best sub-$1,000 foldable: Galaxy Z Flip 5 Why we select these\xa0sub-$1,000 phones The best sub-$1,000 smartphones in 2024 Editor's choice The best iPhone Camera alternative Compact option Product Samsung Galaxy S24 Apple iPhone 16 Google Pixel 8 Pro Samsung Galaxy Z Flip 5 Picture Review Review: Samsung Galaxy S24 Review: Apple iPhone 16 Review: Google Pixel 8 Pro Review: Samsung Galaxy Z Flip 5 Performance Snapdragon 8 Gen 3 (US) Exynos 2400 (global) 8 GB LPDDR5X RAM 128 GB UFS 3.1 storage 256 GB UFS 4.0 storage No storage expansion Apple A18 8 GB RAM 128 / 256 / 512 GB storage No storage expansion Google Tensor G3 12 GB LPDDR5x RAM 128 / 256 / 512 / 1024 GB UFS 3.1 storage No storage expansion Snapdragon 8 Gen 2 8 GB RAM 256 / 512 GB UFS 4.0 storage No storage expansion Camera Wide: 50 MP, f/1.8, OIS Ultra-wide: 12 MP, f/2.2 3x telephoto: 10 MP, f/2.4, OIS Selfie: 12 MP, f/2.2 Main: 48 MP, f/1.6, OIS Ultra-wide: 12 MP, f/2.2 - Selfie: 12 MP, f/1.9 Main: 50 MP, f/1.68, OIS Ultra-wide: 48 MP, f/1.95 5x telephoto: 48 MP, f/2.8 Selfie: 10.5 MP, f/2.2 Main: 12 MP, f/1.8, OIS Ultra-wide: 12 MP, f/2.2 - Selfie: 10 MP, f/2.2 Offer* Check offer $ 799 . 99 (128GB -  new) * Check offer (Samsung) * Free w/ trade-in (T-Mobile) * Check offer $ 0 . 01 (128 GB -  new) * Check offer (BestBuy) * Find on eBay (eBay) * Check offer $ 709 . 99 (128 GB -  new) * Check offer (Google) * Check offer (BestBuy) * Check offer $ 667 . 99 (128GB -  new) * Check offer (Samsung) * Free w/ trade-in (T-Mobile) * The best sub-$1000 Android: Samsung Galaxy S24 The S24 beautiful back can also delight smartphone customers. / © nextpit Shortly after its release, the Samsung Galaxy S24 quickly became our top pick for the best smartphones under $1,000. With this new model, Samsung has outdone itself once again, launching an impressive high-end device that not only features an exceptional 6.2-inch display but also delivers powerful performance with the Snapdragon 8 Gen 3 processor. Additionally, users can expect up to seven years of updates and innovative AI functions . Our review of the Galaxy S24 dives into what you can expect from these features. Also read: Best Samsung smartphones to buy in 2024 Samsung has remained true to its dimensions, and you can expect a compact semi-flagship with a good feel. Unfortunately, there are no innovations in the camera area, which is not necessarily a bad thing as the camera setup is still one of the best on the market. The battery also lasts a long time, but you are missing a modern quick-charging feature. Summary Buy Samsung Galaxy S24 Good Powerful AI functions Outstanding display Compact and good feel Commendable update policy Performance is absolutely okay Bad No camera upgrade 128 GB UFS 3.1 memory Larger battery, shorter runtime Charging not up to date Check offer $ 799 . 99 (128GB -  new) * Check offer (Samsung) * Free w/ trade-in (T-Mobile) * Go to review Samsung Galaxy S24 Check offer $ 799 . 99 (128GB -  new) Check offer (Samsung) Free w/ trade-in (T-Mobile) The best sub-$1,000 iPhone: Apple iPhone 16 The new aligned camera arrangement makes it easy to spot the new model. / © nextpit With a streamlined selection of phones and with the discontinuation of the previous generation Pro model, the vanilla iPhone is the usual suggestion in this price category. For 2024, the demands of the AI trend dictated two discreet upgrades on the base model: Expanded RAM and a new A18 processor ready to power all the Apple Intelligence the phone can get and bring better energy efficiency to boot. Additionally, the iPhone 16 has not one but two new buttons, the Action button which debuted on the iPhone 15 Pro family, and the Camera Control, a capacitive and dual-stage button that can be used as a shutter button, shortcut, and camera settings selector. All these upgrades make the iPhone a more versatile camera for both stills and video. Summary Buy Apple iPhone 16 Good New shortcuts with the Action Button and Camera Control Major hardware upgrade thanks to A18 SoC and 8 GB RAM Image quality can be customized in many ways Outstanding battery life Bad Lags behind the competition without AI integration Only 60 Hz refresh rate for the display Camera Control is only really practical in landscape mode Check offer $ 0 . 01 (128 GB -  new) * Check offer (BestBuy) * Find on eBay (eBay) * Go to review Apple iPhone 16 $829.99 Check offer $ 0 . 01 (128 GB -  new) Check offer (BestBuy) Find on eBay (eBay) The best camera phone under $1,000: Google Pixel 8\xa0Pro The Pixel 8 Pro is the king of smartphone photography. / © nextpit The Google Pixel 8 Pro is a pricier option than before, starting at $999, and it comes in cool colors like light blue, black, and beige. It offers 12 GB of RAM and 128 GB of storage as a base model, but you can choose versions with more storage (256 GB or 512 GB). It's important to note that you can't expand storage with a microSD card. The phone's display is exceptional, with super bright settings that make it easy to use outdoors. It has a solid processor for everyday tasks, but it might not handle really demanding games as well as some competitors. When it comes to photos, Google's software and artificial intelligence make the Pixel 8 Pro stand out. The battery life is decent for a day of use, but it could be better. The main downside is the higher price compared to previous generations, even though Google promises seven years of updates . Read also: Best camera phones to buy in\xa02024 Some people might compare it to iPhones, Samsung Galaxy phones, or Xiaomi phones, which also cost a lot. Those phones may have faster processors, but the Pixel 8 Pro shines in display quality and camera performance. However, it charges slowly, doesn't come with a power adapter, and some promised features aren't available right away. We'll have to wait and see if Google can keep its promise of long-term updates. Summary Buy Google Pixel 8 Pro Good A smartphone camera at its best Merciless update promise Better haptics than the predecessor Sufficient everyday performance Great AI functions 1-120 Hz display Bad G3 is not a flagship processor Price hike No charger included Some promised features are still missing Check offer $ 709 . 99 (128 GB -  new) * Check offer (Google) * Check offer (BestBuy) * Go to review Google Pixel 8 Pro $999.00 Check offer $ 709 . 99 (128 GB -  new) Check offer (Google) Check offer (BestBuy) The best sub-$1,000 compact/foldable: Galaxy Z Flip 5 Bigger and more functional: The cover screen offers many more possibilities in 2024. / © nextpit With the discontinuation of compact phones such as the iPhone mini and the Asus Zenfone, flip phones are the de facto compact smartphones nowadays. The Galaxy Z Flip 5 may not be the newest of those, but it offers almost the same features and performance as its successor, with a lower price (and more frequent deals). The external screen was expanded to display selected apps ( but there are workarounds here ) so you don't need to open the phone all the time. And the Z Flip 5 got a couple of Galaxy AI features since its launch , with more to come. There are a few compromises in the Flip experience though: The camera is not as versatile, and battery life is shorter than our other selections. Summary Buy Samsung Galaxy Z Flip 5 Good Truly useful cover display Improved hinge mechanics Balanced display image quality Fluid software experience Above-average camera quality Bad Slightly larger crease in the display Only average battery life Charging time exceeds one hour No charger included in the box Check offer $ 667 . 99 (128GB -  new) * Check offer (Samsung) * Free w/ trade-in (T-Mobile) * Go to review Samsung Galaxy Z Flip 5 $999.99 Check offer $ 667 . 99 (128GB -  new) Check offer (Samsung) Free w/ trade-in (T-Mobile) Why are\xa0sub-$1,000 smartphones not real\xa0flagships anymore? With phones long past the $1000 mark, we will inevitably deal with trade-offs when looking for an option under that price. A few features like 5G, eSIM, NFC, and wireless charging are still standard in this category, but in other categories, we still find some differentiation. So for this selection we concentrated on the following specs: Our selection criteria Display: The screen characteristics influence not only how sharp (resolution) or smooth (refresh rate) content is displayed, they also indicate how big or small the phone is. We chose options that range from the pocket-friendly Galaxy Flip all the way to the big 6.7-inch camera alternative. Performance: Although all phones above should perform pretty well in both apps and games with their flagship SoCs.\xa0The amount of RAM will determine how fluid will be the multitasking performance, especially with the memory demands of AI features. Also, be careful to avoid 128 GB storage models if you like to have a lot of apps, photos, and videos stored on your device. Camera: The feature that separates these phones from those in the cheaper selections is mainly the cameras: Better image quality with bigger sensors, models with telephoto lenses for zoomed shots, and better image processing for night images and filters. If you like to photograph big vistas, make sure the ultra-wide lens has enough resolution for your photos, on the other hand, prioritize a telephoto lens if you usually struggle to get a good enough zoom on your subjects. Buying advice What to expect from a $1,000 smartphone in 2024 Smartphones that cost less than $1,000 have become less premium in nature, but they are still considered high-end smartphones. When buying a smartphone close to the $1,000 mark, it is clear that compromises will have to be made, although not to the extent of a $400 smartphone. To remain relevant at the $1,000 price point, these smartphones offer everything you need to have an almost flawless user experience. You will benefit from an excellent update policy with at least five years of security updates. The finish and workmanship should be impeccable with IP68 certification and a glass back to boot. Battery life is not to be sneezed at, thanks to the huge battery capacities that lie between 4,000 and 5,000 mAh. When it comes to the camera, you can expect very good image quality and even a telephoto lens. Compromises made in a sub-$1,000 smartphones As mentioned earlier, there are compromises made in a smartphone that falls within this price range that will not make it a crippling experience. The user experience is still pleasant enough, and you can do almost anything you want with your smartphone. However, just like the cameras help separate $1000 phones from $600 models, true flagship phones have even more advanced cameras. Manufacturers also differentiate their high-end smartphones with hardware elements such as a less impressive primary lens, an older SoC, or by using older connectivity and fast charging standards. That's it for our buying guide of the best sub-$1,000 smartphones. Depending on what you are looking for, we hope you found your next flagship! Upcoming sales events Black Week 25 to 29 November 2024 Black Friday 29 November 2024 Cyber Monday 2 December 2024 Amazon Prime Day tbc What do you think of the fact that smartphones under $1,000 are not the 'real' flagships anymore? Do you have any suggestions for models that could have been part of this selection? Last updated in November 2024. Older comments were kept and may refer to older versions of this guide. The best smartphones under $400 Editorial tip Price tip 3rd place 4th place 5th place Product Google Pixel 6a Apple iPhone SE (2022) Samsung Galaxy A53 OnePlus Nord N20 Motorola Moto G Stylus 5G (2023) Image Review Review: Google Pixel 6a Review: Apple iPhone SE (2022) Review: Samsung Galaxy A53 Not yet tested Not yet tested Price (MSRP) $449.00 $429.00 $449.99 $299.00 $399.00 Offer* Check offer $ 299 . 99 (Amazon -  new) * Check offer (BestBuy) * Check offer (Walmart) * Check offer $ 313 . 17 (64 GB -  new) * Free w/ trade-in (T-Mobile) * $170.24 w/ plan (Walmart) * Check offer $ 329 . 99 (128 GB -  new) * Check offer (Samsung) * Check offer (Walmart) * Check offer (BestBuy) * Check offer (OnePlus) * Find on Amazon (Amazon) * Check offer (Motorola) * Free w/ trade-in (T-Mobile) * Find on Amazon (Amazon) * Explore our guide for phones under $400 Samsung Samsung Galaxy Z Flip 5 ⭐ Google Google Pixel 8 Pro ⭐ Samsung Galaxy S24 ⭐ Apple Apple iPhone 16 ⭐ + Previous article Previous article Next article Next article nextpit receives a commission for purchases made via the marked links. This has no influence on the editorial content and there are no costs for you. You can find out more about how we make money on our transparency page . Go to comment (0) Rubens Eishima Writer Having written about technology since 2008 for a number of websites in Brazil, Spain, Denmark, and Germany, I specialize in the mobile ecosystem, including various models, components, and apps. I tend to not only value performance and specifications, but also things like repairability, durability, and manufacturer support. I tend to prioritize the end-user's point of view whenever possible. To the author profile Liked this article? Share now! Follow us: Recommended articles Android Tablets Compared: These Are the Best Models to Buy in 2024 Rubens Eishima 2 days ago The Best Phones Under $400 That Are Worth Your Money Camila Rinaldi 1 week ago Xiaomi or Samsung? Phones, Ecosystems and Updates Compared Carsten Drees 3 weeks ago Hidden iOS 18 Trick: iPhone Reboots Itself to Boost Anti-Theft Security Jade Bryan 20 hours ago How to Use Your Phone as a Wi-Fi Extender Rahul Srinivas 1 day ago Android 15: Which Phones Are Expected to Get the Update? Rubens Eishima 2 days ago Latest articles Buying Guide: Smart Scales With And Without Handles Compared Stefan Möllenhoff 16 hours ago Top 5 Apps of the Week: Carrion, Pokémon TCG Pocket, and More! Edwin Kee 22 hours ago One of the Most Famous Video Game Classics is Free This Week Corinna Oettinger 1 day ago Our Favorite Bluetti AC200L Power Station is $1000 Off Jade Bryan 1 day ago Get These Android and iOS Apps for Free This Weekend Edwin Kee 1 day ago Up Comments Push notification All offers Share article Save in Pocket Next article No comments Write new comment: All changes will be saved. No drafts are saved when editing Submit Write new comment: All changes will be saved. No drafts are saved when editing Submit Cancel ✕ Top Content Our Favorite Bluetti AC200L Power Station is $1000 Off Recommended editorial content Jobbio With your consent, external content is loaded here. Activate content By clicking on the button above, you agree that external content may be displayed to you. Personal data may be transmitted to third-party providers in the process. You can find more information about this in our Privacy Policy . Sign in Sign up Continue with Google Continue with Facebook or Email address Password Forgot password? Stay signed in Login Don't have an nextpit account yet? Register here Register for free and become part of our community. As a registered member, you can leave comments, exchange ideas with others, take part in raffles and be active in the forum. User name This name will be shown publicly on your posts in the community. (At least 3 characters!) Email address Password Stay signed in I accept the terms and conditions and the privacy policy . This site is protected by reCAPTCHA and the Google Privacy Policy and Terms of Service apply. Finish Our Formats News Best List How To Opinion Polls Deals Reviews Versus Our Topics Smartphone Headphones Wearables Apps eMobility Smart Home More Top Content Samsung Galaxy S23 review Apple iPhone 15 Pro Max review Secret codes for Android phones Best offline games for Android Best iOS Apps Best Android Apps iPhone comparison 2024 iOS 17: Best features in a nutshell Android 14: Everything you need to know about nextpit International Deutsch : nextpit.de Deutsch : inside-digital.de English : nextpit.com Español : nextpit.es Português : nextpit.com.br Français : nextpit.fr Italiano : nextpit.it nextpit since 2009 Follow us: Home Staff Jobs at nextpit About us Site notice Terms & Conditions Privacy Policy Help Manage notifications Advertising", 'score': 0.99910307}]
{'tavily_search': {'blogs_content': [{'title': 'The Best Phones Under $1000 to Buy Today - NextPit', 'url': 'https://www.nextpit.com/best-smartphones-under-1000', 'content': "The Best Phones Under $1000 to Buy Today Buying Guide Buying Guide Smartphones The best smartphones Best under $400 Best under $200 Wearables The best Bluetooth headphones to buy in 2024 Which Garmin smartwatch is the best for me? The best Apple and Android smartwatches of 2024 Deals Samsung Galaxy S22: Should you buy it now? Buying the iPhone 13? Best OnePlus 11 offer: Where to buy it! Our Favorite Bluetti AC200L Power Station is $1000 Off Apple's M2 MacBook Air with 16 GB RAM for $749 is a Must-Have Laptop Apple's Thinner Watch Series 10 Falls to a New Low for 13% Off Reviews Reviews Smartphone Samsung Galaxy S23 Ultra review Samsung Galaxy S23 review Samsung Galaxy S23+ review Apple iPhone 14 Plus review Wearables OnePlus Buds Pro 2 review Sony WH-1000XM5 review Apple Watch Ultra review Nuki Smart Lock Pro 4.0 Review: Bid Your Keys Goodbye How This Withings Smart Scale Transformed My Understanding of My Body Samsung Galaxy Tab S10+: The Productivity Powerhouse You Pay For News News Apple iPhone Comparison Best iPad Samsung Galaxy S23 Ultra vs S22 Ultra Which Samsung phones will receive Android 13? Smartphone Android 13 iOS 16.4 Beta Wearables Apps Hidden iOS 18 Trick: iPhone Reboots Itself to Boost Anti-Theft Security Top 5 Apps of the Week: Carrion, Pokémon TCG Pocket, and More! One of the Most Famous Video Game Classics is Free This Week How To How To Smarthome How to check the battery status Samsung One UI Top 10 Android 13 gestures Apps Contacts not showing in WhatsApp? Get fast and easy translations on your Android Activate the incognito mode on YouTube How to Use Your Phone as a Wi-Fi Extender How to Install Xiaomi's Super Wallpapers on Compatible Android Smartphones Possible Fix for YouTube's New Update Glitches Topics Topics Smartphones Wearables Apps eMobility Smart Home More Forum Forum Latest forum posts Whatsapp non officiel avec mon numero de telephone The transition from Dalvik to Android Runtime (ART) has significantly enhanced app performance via JIT and AOT compilation. How do these techniques di Using AI to create Images Hello from a Newbie! Waze is not showing the map! Additional Macro Lens for Moto G Fast Hello everyone Gimbooks Pay Unanswered The transition from Dalvik to Android Runtime (ART) has significantly enhanced app performance via JIT and AOT compilation. How do these techniques di Gimbooks Pay [APP] Unique Zipper Lock Screen - Zip Lock New Member Introduction [App] 3D Parallax Wallpaper & Wallpaper 4K PHOTOS OUT OF ORDER ON WHATSAPP Kids Match Game: Play & Learn WhatsApp is pausing the audio when it’s on my ear Quick Links Recent posts Ask a question Post new thread Our forum rules Mods + Admins Hall of Fame Community Guide Search Login Hot topics Android 15 iOS 18 iPhone 16 iPhone 16 Pro Home Smartphone Hardware The Best Sub-$1,000 Smartphones You Can Buy in 2024 7 min read 7 min No comments 0 Nov 11, 2024, 10:46 AM © nextpit Rubens Eishima Writer Which high-end smartphone should you buy for under $1,000 in 2024? To help you choose the most powerful smartphone, the best camera smartphone, or simply a compact option, we have selected for you the best affordable flagships of the moment. Such as the Galaxy S24, the iPhone 16, and the Google Pixel 8 Pro. Table of Contents: The best sub-$1,000 smartphones in 2024 The best Android sub-$1000: Samsung Galaxy S24 The best sub-$1,000 iPhone: Apple iPhone 16 The best camera alternative under $1,000: Googl e Pixel 8\xa0Pro The best sub-$1,000 foldable: Galaxy Z Flip 5 Why we select these\xa0sub-$1,000 phones The best sub-$1,000 smartphones in 2024 Editor's choice The best iPhone Camera alternative Compact option Product Samsung Galaxy S24 Apple iPhone 16 Google Pixel 8 Pro Samsung Galaxy Z Flip 5 Picture Review Review: Samsung Galaxy S24 Review: Apple iPhone 16 Review: Google Pixel 8 Pro Review: Samsung Galaxy Z Flip 5 Performance Snapdragon 8 Gen 3 (US) Exynos 2400 (global) 8 GB LPDDR5X RAM 128 GB UFS 3.1 storage 256 GB UFS 4.0 storage No storage expansion Apple A18 8 GB RAM 128 / 256 / 512 GB storage No storage expansion Google Tensor G3 12 GB LPDDR5x RAM 128 / 256 / 512 / 1024 GB UFS 3.1 storage No storage expansion Snapdragon 8 Gen 2 8 GB RAM 256 / 512 GB UFS 4.0 storage No storage expansion Camera Wide: 50 MP, f/1.8, OIS Ultra-wide: 12 MP, f/2.2 3x telephoto: 10 MP, f/2.4, OIS Selfie: 12 MP, f/2.2 Main: 48 MP, f/1.6, OIS Ultra-wide: 12 MP, f/2.2 - Selfie: 12 MP, f/1.9 Main: 50 MP, f/1.68, OIS Ultra-wide: 48 MP, f/1.95 5x telephoto: 48 MP, f/2.8 Selfie: 10.5 MP, f/2.2 Main: 12 MP, f/1.8, OIS Ultra-wide: 12 MP, f/2.2 - Selfie: 10 MP, f/2.2 Offer* Check offer $ 799 . 99 (128GB -  new) * Check offer (Samsung) * Free w/ trade-in (T-Mobile) * Check offer $ 0 . 01 (128 GB -  new) * Check offer (BestBuy) * Find on eBay (eBay) * Check offer $ 709 . 99 (128 GB -  new) * Check offer (Google) * Check offer (BestBuy) * Check offer $ 667 . 99 (128GB -  new) * Check offer (Samsung) * Free w/ trade-in (T-Mobile) * The best sub-$1000 Android: Samsung Galaxy S24 The S24 beautiful back can also delight smartphone customers. / © nextpit Shortly after its release, the Samsung Galaxy S24 quickly became our top pick for the best smartphones under $1,000. With this new model, Samsung has outdone itself once again, launching an impressive high-end device that not only features an exceptional 6.2-inch display but also delivers powerful performance with the Snapdragon 8 Gen 3 processor. Additionally, users can expect up to seven years of updates and innovative AI functions . Our review of the Galaxy S24 dives into what you can expect from these features. Also read: Best Samsung smartphones to buy in 2024 Samsung has remained true to its dimensions, and you can expect a compact semi-flagship with a good feel. Unfortunately, there are no innovations in the camera area, which is not necessarily a bad thing as the camera setup is still one of the best on the market. The battery also lasts a long time, but you are missing a modern quick-charging feature. Summary Buy Samsung Galaxy S24 Good Powerful AI functions Outstanding display Compact and good feel Commendable update policy Performance is absolutely okay Bad No camera upgrade 128 GB UFS 3.1 memory Larger battery, shorter runtime Charging not up to date Check offer $ 799 . 99 (128GB -  new) * Check offer (Samsung) * Free w/ trade-in (T-Mobile) * Go to review Samsung Galaxy S24 Check offer $ 799 . 99 (128GB -  new) Check offer (Samsung) Free w/ trade-in (T-Mobile) The best sub-$1,000 iPhone: Apple iPhone 16 The new aligned camera arrangement makes it easy to spot the new model. / © nextpit With a streamlined selection of phones and with the discontinuation of the previous generation Pro model, the vanilla iPhone is the usual suggestion in this price category. For 2024, the demands of the AI trend dictated two discreet upgrades on the base model: Expanded RAM and a new A18 processor ready to power all the Apple Intelligence the phone can get and bring better energy efficiency to boot. Additionally, the iPhone 16 has not one but two new buttons, the Action button which debuted on the iPhone 15 Pro family, and the Camera Control, a capacitive and dual-stage button that can be used as a shutter button, shortcut, and camera settings selector. All these upgrades make the iPhone a more versatile camera for both stills and video. Summary Buy Apple iPhone 16 Good New shortcuts with the Action Button and Camera Control Major hardware upgrade thanks to A18 SoC and 8 GB RAM Image quality can be customized in many ways Outstanding battery life Bad Lags behind the competition without AI integration Only 60 Hz refresh rate for the display Camera Control is only really practical in landscape mode Check offer $ 0 . 01 (128 GB -  new) * Check offer (BestBuy) * Find on eBay (eBay) * Go to review Apple iPhone 16 $829.99 Check offer $ 0 . 01 (128 GB -  new) Check offer (BestBuy) Find on eBay (eBay) The best camera phone under $1,000: Google Pixel 8\xa0Pro The Pixel 8 Pro is the king of smartphone photography. / © nextpit The Google Pixel 8 Pro is a pricier option than before, starting at $999, and it comes in cool colors like light blue, black, and beige. It offers 12 GB of RAM and 128 GB of storage as a base model, but you can choose versions with more storage (256 GB or 512 GB). It's important to note that you can't expand storage with a microSD card. The phone's display is exceptional, with super bright settings that make it easy to use outdoors. It has a solid processor for everyday tasks, but it might not handle really demanding games as well as some competitors. When it comes to photos, Google's software and artificial intelligence make the Pixel 8 Pro stand out. The battery life is decent for a day of use, but it could be better. The main downside is the higher price compared to previous generations, even though Google promises seven years of updates . Read also: Best camera phones to buy in\xa02024 Some people might compare it to iPhones, Samsung Galaxy phones, or Xiaomi phones, which also cost a lot. Those phones may have faster processors, but the Pixel 8 Pro shines in display quality and camera performance. However, it charges slowly, doesn't come with a power adapter, and some promised features aren't available right away. We'll have to wait and see if Google can keep its promise of long-term updates. Summary Buy Google Pixel 8 Pro Good A smartphone camera at its best Merciless update promise Better haptics than the predecessor Sufficient everyday performance Great AI functions 1-120 Hz display Bad G3 is not a flagship processor Price hike No charger included Some promised features are still missing Check offer $ 709 . 99 (128 GB -  new) * Check offer (Google) * Check offer (BestBuy) * Go to review Google Pixel 8 Pro $999.00 Check offer $ 709 . 99 (128 GB -  new) Check offer (Google) Check offer (BestBuy) The best sub-$1,000 compact/foldable: Galaxy Z Flip 5 Bigger and more functional: The cover screen offers many more possibilities in 2024. / © nextpit With the discontinuation of compact phones such as the iPhone mini and the Asus Zenfone, flip phones are the de facto compact smartphones nowadays. The Galaxy Z Flip 5 may not be the newest of those, but it offers almost the same features and performance as its successor, with a lower price (and more frequent deals). The external screen was expanded to display selected apps ( but there are workarounds here ) so you don't need to open the phone all the time. And the Z Flip 5 got a couple of Galaxy AI features since its launch , with more to come. There are a few compromises in the Flip experience though: The camera is not as versatile, and battery life is shorter than our other selections. Summary Buy Samsung Galaxy Z Flip 5 Good Truly useful cover display Improved hinge mechanics Balanced display image quality Fluid software experience Above-average camera quality Bad Slightly larger crease in the display Only average battery life Charging time exceeds one hour No charger included in the box Check offer $ 667 . 99 (128GB -  new) * Check offer (Samsung) * Free w/ trade-in (T-Mobile) * Go to review Samsung Galaxy Z Flip 5 $999.99 Check offer $ 667 . 99 (128GB -  new) Check offer (Samsung) Free w/ trade-in (T-Mobile) Why are\xa0sub-$1,000 smartphones not real\xa0flagships anymore? With phones long past the $1000 mark, we will inevitably deal with trade-offs when looking for an option under that price. A few features like 5G, eSIM, NFC, and wireless charging are still standard in this category, but in other categories, we still find some differentiation. So for this selection we concentrated on the following specs: Our selection criteria Display: The screen characteristics influence not only how sharp (resolution) or smooth (refresh rate) content is displayed, they also indicate how big or small the phone is. We chose options that range from the pocket-friendly Galaxy Flip all the way to the big 6.7-inch camera alternative. Performance: Although all phones above should perform pretty well in both apps and games with their flagship SoCs.\xa0The amount of RAM will determine how fluid will be the multitasking performance, especially with the memory demands of AI features. Also, be careful to avoid 128 GB storage models if you like to have a lot of apps, photos, and videos stored on your device. Camera: The feature that separates these phones from those in the cheaper selections is mainly the cameras: Better image quality with bigger sensors, models with telephoto lenses for zoomed shots, and better image processing for night images and filters. If you like to photograph big vistas, make sure the ultra-wide lens has enough resolution for your photos, on the other hand, prioritize a telephoto lens if you usually struggle to get a good enough zoom on your subjects. Buying advice What to expect from a $1,000 smartphone in 2024 Smartphones that cost less than $1,000 have become less premium in nature, but they are still considered high-end smartphones. When buying a smartphone close to the $1,000 mark, it is clear that compromises will have to be made, although not to the extent of a $400 smartphone. To remain relevant at the $1,000 price point, these smartphones offer everything you need to have an almost flawless user experience. You will benefit from an excellent update policy with at least five years of security updates. The finish and workmanship should be impeccable with IP68 certification and a glass back to boot. Battery life is not to be sneezed at, thanks to the huge battery capacities that lie between 4,000 and 5,000 mAh. When it comes to the camera, you can expect very good image quality and even a telephoto lens. Compromises made in a sub-$1,000 smartphones As mentioned earlier, there are compromises made in a smartphone that falls within this price range that will not make it a crippling experience. The user experience is still pleasant enough, and you can do almost anything you want with your smartphone. However, just like the cameras help separate $1000 phones from $600 models, true flagship phones have even more advanced cameras. Manufacturers also differentiate their high-end smartphones with hardware elements such as a less impressive primary lens, an older SoC, or by using older connectivity and fast charging standards. That's it for our buying guide of the best sub-$1,000 smartphones. Depending on what you are looking for, we hope you found your next flagship! Upcoming sales events Black Week 25 to 29 November 2024 Black Friday 29 November 2024 Cyber Monday 2 December 2024 Amazon Prime Day tbc What do you think of the fact that smartphones under $1,000 are not the 'real' flagships anymore? Do you have any suggestions for models that could have been part of this selection? Last updated in November 2024. Older comments were kept and may refer to older versions of this guide. The best smartphones under $400 Editorial tip Price tip 3rd place 4th place 5th place Product Google Pixel 6a Apple iPhone SE (2022) Samsung Galaxy A53 OnePlus Nord N20 Motorola Moto G Stylus 5G (2023) Image Review Review: Google Pixel 6a Review: Apple iPhone SE (2022) Review: Samsung Galaxy A53 Not yet tested Not yet tested Price (MSRP) $449.00 $429.00 $449.99 $299.00 $399.00 Offer* Check offer $ 299 . 99 (Amazon -  new) * Check offer (BestBuy) * Check offer (Walmart) * Check offer $ 313 . 17 (64 GB -  new) * Free w/ trade-in (T-Mobile) * $170.24 w/ plan (Walmart) * Check offer $ 329 . 99 (128 GB -  new) * Check offer (Samsung) * Check offer (Walmart) * Check offer (BestBuy) * Check offer (OnePlus) * Find on Amazon (Amazon) * Check offer (Motorola) * Free w/ trade-in (T-Mobile) * Find on Amazon (Amazon) * Explore our guide for phones under $400 Samsung Samsung Galaxy Z Flip 5 ⭐ Google Google Pixel 8 Pro ⭐ Samsung Galaxy S24 ⭐ Apple Apple iPhone 16 ⭐ + Previous article Previous article Next article Next article nextpit receives a commission for purchases made via the marked links. This has no influence on the editorial content and there are no costs for you. You can find out more about how we make money on our transparency page . Go to comment (0) Rubens Eishima Writer Having written about technology since 2008 for a number of websites in Brazil, Spain, Denmark, and Germany, I specialize in the mobile ecosystem, including various models, components, and apps. I tend to not only value performance and specifications, but also things like repairability, durability, and manufacturer support. I tend to prioritize the end-user's point of view whenever possible. To the author profile Liked this article? Share now! Follow us: Recommended articles Android Tablets Compared: These Are the Best Models to Buy in 2024 Rubens Eishima 2 days ago The Best Phones Under $400 That Are Worth Your Money Camila Rinaldi 1 week ago Xiaomi or Samsung? Phones, Ecosystems and Updates Compared Carsten Drees 3 weeks ago Hidden iOS 18 Trick: iPhone Reboots Itself to Boost Anti-Theft Security Jade Bryan 20 hours ago How to Use Your Phone as a Wi-Fi Extender Rahul Srinivas 1 day ago Android 15: Which Phones Are Expected to Get the Update? Rubens Eishima 2 days ago Latest articles Buying Guide: Smart Scales With And Without Handles Compared Stefan Möllenhoff 16 hours ago Top 5 Apps of the Week: Carrion, Pokémon TCG Pocket, and More! Edwin Kee 22 hours ago One of the Most Famous Video Game Classics is Free This Week Corinna Oettinger 1 day ago Our Favorite Bluetti AC200L Power Station is $1000 Off Jade Bryan 1 day ago Get These Android and iOS Apps for Free This Weekend Edwin Kee 1 day ago Up Comments Push notification All offers Share article Save in Pocket Next article No comments Write new comment: All changes will be saved. No drafts are saved when editing Submit Write new comment: All changes will be saved. No drafts are saved when editing Submit Cancel ✕ Top Content Our Favorite Bluetti AC200L Power Station is $1000 Off Recommended editorial content Jobbio With your consent, external content is loaded here. Activate content By clicking on the button above, you agree that external content may be displayed to you. Personal data may be transmitted to third-party providers in the process. You can find more information about this in our Privacy Policy . Sign in Sign up Continue with Google Continue with Facebook or Email address Password Forgot password? Stay signed in Login Don't have an nextpit account yet? Register here Register for free and become part of our community. As a registered member, you can leave comments, exchange ideas with others, take part in raffles and be active in the forum. User name This name will be shown publicly on your posts in the community. (At least 3 characters!) Email address Password Stay signed in I accept the terms and conditions and the privacy policy . This site is protected by reCAPTCHA and the Google Privacy Policy and Terms of Service apply. Finish Our Formats News Best List How To Opinion Polls Deals Reviews Versus Our Topics Smartphone Headphones Wearables Apps eMobility Smart Home More Top Content Samsung Galaxy S23 review Apple iPhone 15 Pro Max review Secret codes for Android phones Best offline games for Android Best iOS Apps Best Android Apps iPhone comparison 2024 iOS 17: Best features in a nutshell Android 14: Everything you need to know about nextpit International Deutsch : nextpit.de Deutsch : inside-digital.de English : nextpit.com Español : nextpit.es Português : nextpit.com.br Français : nextpit.fr Italiano : nextpit.it nextpit since 2009 Follow us: Home Staff Jobs at nextpit About us Site notice Terms & Conditions Privacy Policy Help Manage notifications Advertising", 'score': 0.99910307}]}}
{'schema_mapping': {'product_schema': [{'title': 'Samsung Galaxy S24', 'url': 'https://www.nextpit.com/best-smartphones-under-1000', 'content': 'The Samsung Galaxy S24 is a high-end smartphone with a 6.2-inch display, powerful performance, and up to seven years of updates.', 'pros': ['Powerful AI functions', 'Outstanding display', 'Compact and good feel', 'Commendable update policy', 'Performance is absolutely okay'], 'cons': ['No camera upgrade', '128 GB UFS 3.1 memory', 'Larger battery, shorter runtime', 'Charging not up to date'], 'highlights': {'Processor': 'Snapdragon 8 Gen 3 (US) / Exynos 2400 (global)', 'RAM': '8 GB LPDDR5X RAM', 'Storage': '128 GB UFS 3.1 storage / 256 GB UFS 4.0 storage', 'Camera': 'Wide: 50 MP, f/1.8, OIS / Ultra-wide: 12 MP, f/2.2 / 3x telephoto: 10 MP, f/2.4, OIS / Selfie: 12 MP, f/2.2'}, 'score': 0.0}, {'title': 'Apple iPhone 16', 'url': 'https://www.nextpit.com/best-smartphones-under-1000', 'content': 'The Apple iPhone 16 is a high-end smartphone with a streamlined selection of phones and two discreet upgrades on the base model: Expanded RAM and a new A18 processor.', 'pros': ['New shortcuts with the Action Button and Camera Control', 'Major hardware upgrade thanks to A18 SoC and 8 GB RAM', 'Image quality can be customized in many ways', 'Outstanding battery life'], 'cons': ['Lags behind the competition without AI integration', 'Only 60 Hz refresh rate for the display', 'Camera Control is only really practical in landscape mode'], 'highlights': {'Processor': 'Apple A18', 'RAM': '8 GB RAM', 'Storage': '128 / 256 / 512 GB storage', 'Camera': 'Main: 48 MP, f/1.6, OIS / Ultra-wide: 12 MP, f/2.2 / Selfie: 12 MP, f/1.9'}, 'score': 0.0}, {'title': 'Google Pixel 8 Pro', 'url': 'https://www.nextpit.com/best-smartphones-under-1000', 'content': 'The Google Pixel 8 Pro is a high-end smartphone with a 6.7-inch display, 12 GB of RAM, and 128 GB of storage as a base model.', 'pros': ['A smartphone camera at its best', 'Merciless update promise', 'Better haptics than the predecessor', 'Sufficient everyday performance', 'Great AI functions', '1-120 Hz display'], 'cons': ['G3 is not a flagship processor', 'Price hike', 'No charger included', 'Some promised features are still missing'], 'highlights': {'Processor': 'Google Tensor G3', 'RAM': '12 GB LPDDR5x RAM', 'Storage': '128 / 256 / 512 / 1024 GB UFS 3.1 storage', 'Camera': 'Main: 50 MP, f/1.68, OIS / Ultra-wide: 48 MP, f/1.95 / 5x telephoto: 48 MP, f/2.8 / Selfie: 10.5 MP, f/2.2'}, 'score': 0.0}, {'title': 'Samsung Galaxy Z Flip 5', 'url': 'https://www.nextpit.com/best-smartphones-under-1000', 'content': 'The Samsung Galaxy Z Flip 5 is a compact flip smartphone with a 6.7-inch display, 8 GB of RAM, and 256 GB of storage.', 'pros': ['Truly useful cover display', 'Improved hinge mechanics', 'Balanced display image quality', 'Fluid software experience', 'Above-average camera quality'], 'cons': ['Slightly larger crease in the display', 'Only average battery life', 'Charging time exceeds one hour', 'No charger included in the box'], 'highlights': {'Processor': 'Snapdragon 8 Gen 2', 'RAM': '8 GB RAM', 'Storage': '256 / 512 GB UFS 4.0 storage', 'Camera': 'Main: 12 MP, f/1.8, OIS / Ultra-wide: 12 MP, f/2.2 / Selfie: 10 MP, f/2.2'}, 'score': 0.0}]}}
```

```text
{'comparisons': [{'product_name': 'Samsung Galaxy S24',
   'specs_comparison': {'processor': 'Snapdragon 8 Gen 3 (US) / Exynos 2400 (global)',
    'battery': 'Unknown',
    'camera': 'Wide: 50 MP, f/1.8, OIS / Ultra-wide: 12 MP, f/2.2 / 3x telephoto: 10 MP, f/2.4, OIS / Selfie: 12 MP, f/2.2',
    'display': '6.2-inch',
    'storage': '128 GB UFS 3.1 storage / 256 GB UFS 4.0 storage'},
   'ratings_comparison': {'overall_rating': 4.2,
    'performance': 4.5,
    'battery_life': 3.8,
    'camera_quality': 4.5,
    'display_quality': 4.5},
   'reviews_summary': 'The Samsung Galaxy S24 has a powerful AI function, an outstanding display, and a compact design. However, it lacks a camera upgrade, has limited storage, and a shorter battery life.'},
  {'product_name': 'Apple iPhone 16',
   'specs_comparison': {'processor': 'Apple A18',
    'battery': 'Unknown',
    'camera': 'Main: 48 MP, f/1.6, OIS / Ultra-wide: 12 MP, f/2.2 / Selfie: 12 MP, f/1.9',
    'display': 'Unknown',
    'storage': '128 / 256 / 512 GB storage'},
   'ratings_comparison': {'overall_rating': 4.4,
    'performance': 4.7,
    'battery_life': 4.5,
    'camera_quality': 4.3,
    'display_quality': 4.2},
   'reviews_summary': 'The Apple iPhone 16 has a major hardware upgrade with the A18 SoC and 8 GB RAM, outstanding battery life, and customizable image quality. However, it lags behind the competition without AI integration and has a limited display refresh rate.'},
  {'product_name': 'Google Pixel 8 Pro',
   'specs_comparison': {'processor': 'Google Tensor G3',
    'battery': 'Unknown',
    'camera': 'Main: 50 MP, f/1.68, OIS / Ultra-wide: 48 MP, f/1.95 / 5x telephoto: 48 MP, f/2.8 / Selfie: 10.5 MP, f/2.2',
    'display': '6.7-inch',
    'storage': '128 / 256 / 512 / 1024 GB UFS 3.1 storage'},
   'ratings_comparison': {'overall_rating': 4.6,
    'performance': 4.4,
    'battery_life': 4.1,
    'camera_quality': 4.8,
    'display_quality': 4.6},
   'reviews_summary': 'The Google Pixel 8 Pro has an exceptional camera, a merciless update promise, and sufficient everyday performance. However, it has a non-flagship processor, a price hike, and some missing features.'},
  {'product_name': 'Samsung Galaxy Z Flip 5',
   'specs_comparison': {'processor': 'Snapdragon 8 Gen 2',
    'battery': 'Unknown',
    'camera': 'Main: 12 MP, f/1.8, OIS / Ultra-wide: 12 MP, f/2.2 / Selfie: 10 MP, f/2.2',
    'display': '6.7-inch',
    'storage': '256 / 512 GB UFS 4.0 storage'},
   'ratings_comparison': {'overall_rating': 4.3,
    'performance': 4.2,
    'battery_life': 3.9,
    'camera_quality': 4.2,
    'display_quality': 4.3},
   'reviews_summary': 'The Samsung Galaxy Z Flip 5 has a truly useful cover display, improved hinge mechanics, and balanced display image quality. However, it has a slightly larger crease in the display, only average battery life, and a long charging time.'}],
 'best_product': {'product_name': 'Google Pixel 8 Pro',
  'justification': 'Chosen for its exceptional camera, sufficient everyday performance, and outstanding display quality. Although it has some drawbacks, its overall rating and camera quality make it the best choice among the compared products.'}}
```

```text
{'product_comparison': {'best_product': {'product_name': 'Google Pixel 8 Pro', 'justification': 'Chosen for its exceptional camera, sufficient everyday performance, and outstanding display quality. Although it has some drawbacks, its overall rating and camera quality make it the best choice among the compared products.'}, 'comparison': [{'product_name': 'Samsung Galaxy S24', 'specs_comparison': {'processor': 'Snapdragon 8 Gen 3 (US) / Exynos 2400 (global)', 'battery': 'Unknown', 'camera': 'Wide: 50 MP, f/1.8, OIS / Ultra-wide: 12 MP, f/2.2 / 3x telephoto: 10 MP, f/2.4, OIS / Selfie: 12 MP, f/2.2', 'display': '6.2-inch', 'storage': '128 GB UFS 3.1 storage / 256 GB UFS 4.0 storage'}, 'ratings_comparison': {'overall_rating': 4.2, 'performance': 4.5, 'battery_life': 3.8, 'camera_quality': 4.5, 'display_quality': 4.5}, 'reviews_summary': 'The Samsung Galaxy S24 has a powerful AI function, an outstanding display, and a compact design. However, it lacks a camera upgrade, has limited storage, and a shorter battery life.'}, {'product_name': 'Apple iPhone 16', 'specs_comparison': {'processor': 'Apple A18', 'battery': 'Unknown', 'camera': 'Main: 48 MP, f/1.6, OIS / Ultra-wide: 12 MP, f/2.2 / Selfie: 12 MP, f/1.9', 'display': 'Unknown', 'storage': '128 / 256 / 512 GB storage'}, 'ratings_comparison': {'overall_rating': 4.4, 'performance': 4.7, 'battery_life': 4.5, 'camera_quality': 4.3, 'display_quality': 4.2}, 'reviews_summary': 'The Apple iPhone 16 has a major hardware upgrade with the A18 SoC and 8 GB RAM, outstanding battery life, and customizable image quality. However, it lags behind the competition without AI integration and has a limited display refresh rate.'}, {'product_name': 'Google Pixel 8 Pro', 'specs_comparison': {'processor': 'Google Tensor G3', 'battery': 'Unknown', 'camera': 'Main: 50 MP, f/1.68, OIS / Ultra-wide: 48 MP, f/1.95 / 5x telephoto: 48 MP, f/2.8 / Selfie: 10.5 MP, f/2.2', 'display': '6.7-inch', 'storage': '128 / 256 / 512 / 1024 GB UFS 3.1 storage'}, 'ratings_comparison': {'overall_rating': 4.6, 'performance': 4.4, 'battery_life': 4.1, 'camera_quality': 4.8, 'display_quality': 4.6}, 'reviews_summary': 'The Google Pixel 8 Pro has an exceptional camera, a merciless update promise, and sufficient everyday performance. However, it has a non-flagship processor, a price hike, and some missing features.'}, {'product_name': 'Samsung Galaxy Z Flip 5', 'specs_comparison': {'processor': 'Snapdragon 8 Gen 2', 'battery': 'Unknown', 'camera': 'Main: 12 MP, f/1.8, OIS / Ultra-wide: 12 MP, f/2.2 / Selfie: 10 MP, f/2.2', 'display': '6.7-inch', 'storage': '256 / 512 GB UFS 4.0 storage'}, 'ratings_comparison': {'overall_rating': 4.3, 'performance': 4.2, 'battery_life': 3.9, 'camera_quality': 4.2, 'display_quality': 4.3}, 'reviews_summary': 'The Samsung Galaxy Z Flip 5 has a truly useful cover display, improved hinge mechanics, and balanced display image quality. However, it has a slightly larger crease in the display, only average battery life, and a long charging time.'}]}}
{'youtube_review': {'youtube_link': 'https://www.youtube.com/watch?v=1uSHiNkVGsc'}}
{'display': {'products': [{'title': 'Samsung Galaxy S24', 'url': 'https://www.nextpit.com/best-smartphones-under-1000', 'content': 'The Samsung Galaxy S24 is a high-end smartphone with a 6.2-inch display, powerful performance, and up to seven years of updates.', 'pros': ['Powerful AI functions', 'Outstanding display', 'Compact and good feel', 'Commendable update policy', 'Performance is absolutely okay'], 'cons': ['No camera upgrade', '128 GB UFS 3.1 memory', 'Larger battery, shorter runtime', 'Charging not up to date'], 'highlights': {'Processor': 'Snapdragon 8 Gen 3 (US) / Exynos 2400 (global)', 'RAM': '8 GB LPDDR5X RAM', 'Storage': '128 GB UFS 3.1 storage / 256 GB UFS 4.0 storage', 'Camera': 'Wide: 50 MP, f/1.8, OIS / Ultra-wide: 12 MP, f/2.2 / 3x telephoto: 10 MP, f/2.4, OIS / Selfie: 12 MP, f/2.2'}, 'score': 0.0}, {'title': 'Apple iPhone 16', 'url': 'https://www.nextpit.com/best-smartphones-under-1000', 'content': 'The Apple iPhone 16 is a high-end smartphone with a streamlined selection of phones and two discreet upgrades on the base model: Expanded RAM and a new A18 processor.', 'pros': ['New shortcuts with the Action Button and Camera Control', 'Major hardware upgrade thanks to A18 SoC and 8 GB RAM', 'Image quality can be customized in many ways', 'Outstanding battery life'], 'cons': ['Lags behind the competition without AI integration', 'Only 60 Hz refresh rate for the display', 'Camera Control is only really practical in landscape mode'], 'highlights': {'Processor': 'Apple A18', 'RAM': '8 GB RAM', 'Storage': '128 / 256 / 512 GB storage', 'Camera': 'Main: 48 MP, f/1.6, OIS / Ultra-wide: 12 MP, f/2.2 / Selfie: 12 MP, f/1.9'}, 'score': 0.0}, {'title': 'Google Pixel 8 Pro', 'url': 'https://www.nextpit.com/best-smartphones-under-1000', 'content': 'The Google Pixel 8 Pro is a high-end smartphone with a 6.7-inch display, 12 GB of RAM, and 128 GB of storage as a base model.', 'pros': ['A smartphone camera at its best', 'Merciless update promise', 'Better haptics than the predecessor', 'Sufficient everyday performance', 'Great AI functions', '1-120 Hz display'], 'cons': ['G3 is not a flagship processor', 'Price hike', 'No charger included', 'Some promised features are still missing'], 'highlights': {'Processor': 'Google Tensor G3', 'RAM': '12 GB LPDDR5x RAM', 'Storage': '128 / 256 / 512 / 1024 GB UFS 3.1 storage', 'Camera': 'Main: 50 MP, f/1.68, OIS / Ultra-wide: 48 MP, f/1.95 / 5x telephoto: 48 MP, f/2.8 / Selfie: 10.5 MP, f/2.2'}, 'score': 0.0}, {'title': 'Samsung Galaxy Z Flip 5', 'url': 'https://www.nextpit.com/best-smartphones-under-1000', 'content': 'The Samsung Galaxy Z Flip 5 is a compact flip smartphone with a 6.7-inch display, 8 GB of RAM, and 256 GB of storage.', 'pros': ['Truly useful cover display', 'Improved hinge mechanics', 'Balanced display image quality', 'Fluid software experience', 'Above-average camera quality'], 'cons': ['Slightly larger crease in the display', 'Only average battery life', 'Charging time exceeds one hour', 'No charger included in the box'], 'highlights': {'Processor': 'Snapdragon 8 Gen 2', 'RAM': '8 GB RAM', 'Storage': '256 / 512 GB UFS 4.0 storage', 'Camera': 'Main: 12 MP, f/1.8, OIS / Ultra-wide: 12 MP, f/2.2 / Selfie: 10 MP, f/2.2'}, 'score': 0.0}], 'best_product': {'product_name': 'Google Pixel 8 Pro', 'justification': 'Chosen for its exceptional camera, sufficient everyday performance, and outstanding display quality. Although it has some drawbacks, its overall rating and camera quality make it the best choice among the compared products.'}, 'comparison': [{'product_name': 'Samsung Galaxy S24', 'specs_comparison': {'processor': 'Snapdragon 8 Gen 3 (US) / Exynos 2400 (global)', 'battery': 'Unknown', 'camera': 'Wide: 50 MP, f/1.8, OIS / Ultra-wide: 12 MP, f/2.2 / 3x telephoto: 10 MP, f/2.4, OIS / Selfie: 12 MP, f/2.2', 'display': '6.2-inch', 'storage': '128 GB UFS 3.1 storage / 256 GB UFS 4.0 storage'}, 'ratings_comparison': {'overall_rating': 4.2, 'performance': 4.5, 'battery_life': 3.8, 'camera_quality': 4.5, 'display_quality': 4.5}, 'reviews_summary': 'The Samsung Galaxy S24 has a powerful AI function, an outstanding display, and a compact design. However, it lacks a camera upgrade, has limited storage, and a shorter battery life.'}, {'product_name': 'Apple iPhone 16', 'specs_comparison': {'processor': 'Apple A18', 'battery': 'Unknown', 'camera': 'Main: 48 MP, f/1.6, OIS / Ultra-wide: 12 MP, f/2.2 / Selfie: 12 MP, f/1.9', 'display': 'Unknown', 'storage': '128 / 256 / 512 GB storage'}, 'ratings_comparison': {'overall_rating': 4.4, 'performance': 4.7, 'battery_life': 4.5, 'camera_quality': 4.3, 'display_quality': 4.2}, 'reviews_summary': 'The Apple iPhone 16 has a major hardware upgrade with the A18 SoC and 8 GB RAM, outstanding battery life, and customizable image quality. However, it lags behind the competition without AI integration and has a limited display refresh rate.'}, {'product_name': 'Google Pixel 8 Pro', 'specs_comparison': {'processor': 'Google Tensor G3', 'battery': 'Unknown', 'camera': 'Main: 50 MP, f/1.68, OIS / Ultra-wide: 48 MP, f/1.95 / 5x telephoto: 48 MP, f/2.8 / Selfie: 10.5 MP, f/2.2', 'display': '6.7-inch', 'storage': '128 / 256 / 512 / 1024 GB UFS 3.1 storage'}, 'ratings_comparison': {'overall_rating': 4.6, 'performance': 4.4, 'battery_life': 4.1, 'camera_quality': 4.8, 'display_quality': 4.6}, 'reviews_summary': 'The Google Pixel 8 Pro has an exceptional camera, a merciless update promise, and sufficient everyday performance. However, it has a non-flagship processor, a price hike, and some missing features.'}, {'product_name': 'Samsung Galaxy Z Flip 5', 'specs_comparison': {'processor': 'Snapdragon 8 Gen 2', 'battery': 'Unknown', 'camera': 'Main: 12 MP, f/1.8, OIS / Ultra-wide: 12 MP, f/2.2 / Selfie: 10 MP, f/2.2', 'display': '6.7-inch', 'storage': '256 / 512 GB UFS 4.0 storage'}, 'ratings_comparison': {'overall_rating': 4.3, 'performance': 4.2, 'battery_life': 3.9, 'camera_quality': 4.2, 'display_quality': 4.3}, 'reviews_summary': 'The Samsung Galaxy Z Flip 5 has a truly useful cover display, improved hinge mechanics, and balanced display image quality. However, it has a slightly larger crease in the display, only average battery life, and a long charging time.'}], 'youtube_link': 'https://www.youtube.com/watch?v=1uSHiNkVGsc'}}
Email sent successfully to asadsher2324@gmail.com.
{'send_email': None}
```

![](https://europe-west1-genai-agents-views-tracker.cloudfunctions.net/genai-agents-tracker?notebook=all-agents-tutorials--shopgenie)
