import os
import re
import json
from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


def extract_video_id(url_or_id: str) -> str:
    """Safely extracts the 11-character video ID from a full YouTube URL."""
    if "youtu.be/" in url_or_id:
        return url_or_id.split("youtu.be/")[1][:11]
    elif "youtube.com/watch?v=" in url_or_id:
        return url_or_id.split("watch?v=")[1][:11]
    
    # If they just passed the ID directly, return it
    return url_or_id[:11]


class VideoResponse(BaseModel):
    answer: str = Field(description="The clear, detailed answer to the user's question based on the text.")
    timestamp_links: List[str] = Field(description="List of exact YouTube URLs with timestamps (e.g., https://youtu.be/XYZ?t=120) supporting the answer.")


def generate_timestamped_answer(user_query: str, retrieved_chunks: list, video_link: str) -> dict:
    """
    Takes the question, the retrieved chunks from Team 3, and the YouTube link from Team 5, 
    and returns the final JSON using SambaNova.
    """
    # Clean the video link to ensure formatting works
    video_id = extract_video_id(video_link)

    # Guardrail: If Team 3 finds absolutely nothing, return a graceful fallback
    if not retrieved_chunks:
        return {"answer": "I couldn't find any relevant information in the video.", "timestamp_links": []}

    # Initialize the LLM pointing to SambaNova's API
    llm = ChatOpenAI(
        base_url="https://api.sambanova.ai/v1",
        api_key=os.environ.get("SAMBANOVA_API_KEY"), 
        model="Meta-Llama-3.3-70B-Instruct", 
        temperature=0.1
    )
    
    # Set up the LangChain parser to strictly enforce our JSON structure
    parser = PydanticOutputParser(pydantic_object=VideoResponse)
    
    # Format the chunks from Team 3 into a readable block for the LLM
    formatted_context = ""
    for chunk in retrieved_chunks:
        start_time = chunk['metadata']['start_time_seconds']
        text = chunk['text']
        formatted_context += f"[Time: {start_time}s] {text}\n"

    # The Strict System Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an advanced researcher and smart video lecture assistant. Answer the user's question using ONLY the provided transcript context.
        
        Follow these strict rules:
        1. Answer clearly and concisely.
        2. Identify the exact start times (in seconds) from the context used to formulate your answer.
        3. Generate clickable YouTube links for those timestamps using the format: https://youtu.be/{video_id}?t={{start_time}}s
        4. If the answer is not in the context, your 'answer' field should politely state that you cannot find it in the lecture, and 'timestamp_links' should be an empty list.
        
        CRITICAL: Your entire response MUST be a single, valid JSON object. 
        Do NOT output any conversational text, greetings, explanations, or markdown formatting (like ```json) before or after the JSON object.
        
        {format_instructions}"""),
        ("user", "Context:\n{context}\n\nQuestion: {query}")
    ])

    chain = prompt | llm | parser

    # Run the pipeline
    response = chain.invoke({
        "query": user_query,
        "context": formatted_context,
        "video_id": video_id, # Passes the clean ID to the prompt for link generation
        "format_instructions": parser.get_format_instructions()
    })

    return response.model_dump()

