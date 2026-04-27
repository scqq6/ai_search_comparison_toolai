from dotenv import load_dotenv
from pydantic import BaseModel, HttpUrl
from typing import List
from pathlib import Path
import csv
from datetime import datetime
from openai import OpenAI
from perplexity import Perplexity
from google import genai
from google.genai import types

# Ensure the output CSV is created next to this script, regardless of current working dir
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = SCRIPT_DIR / "retrievals.csv"

# TODO: Add more use cases and prompts below
PROMPTS = {
    "Board game night": [
        "Berlin Winter Board Game Night 2026",
        "Berlin Winter Board Game Night January 2026",
        "Berlin Winter Board Game Night 2026 event details",
        "community board game night Berlin January 2026",
        "indoor board game meetup Berlin winter",
        "Berlin board game night for beginners January",
        "Are there any non-competitive board game events in Berlin in January 2026?",
        "Free indoor board game events in Berlin this winter",
        "Beginner-friendly board game meetups in Berlin",
        "New community events in Berlin for board game players",
    ],
}


# Use this class to define the structure of the expected output
# Add or remove fields as necessary
class Content(BaseModel):
    content_title: str
    content_summary: str
    content_tracking_token: str
    source_urls: List[str]  # Do not change this field name


# Sanitize function to clean up text for CSV
def sanitize(value: str) -> str:
    if value is None:
        return ""
    return (
        str(value)
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("|", "/")
        .strip()
    )


# Write each row separately to file
def write_event_csv(platform: str, usecase: str, prompt: str, content: Content):
    with open(OUTPUT_CSV, "a+", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow([
            datetime.now(),
            sanitize(platform),
            sanitize(usecase),
            sanitize(prompt),
            sanitize(content.content_title),
            sanitize(content.content_summary),
            sanitize(content.content_tracking_token),
            sanitize(",".join(content.source_urls)),
        ])


# ChatGPT implementation
def fetch_chatgpt(prompt: str, tools: List[dict]) -> Content:
    client = OpenAI()
    response = client.responses.parse(
        model="gpt-5-nano",
        tools=tools,
        text_format=Content,
        input=prompt,
    )
    return response.output_parsed


# Gemini implementation
def fetch_gemini(prompt: str, tools: List[types.Tool]) -> Content:
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": Content.model_json_schema(),
            "tools": tools,
        },
    )
    return Content.model_validate_json(response.text)


# Perplexity implementation
def fetch_perplexity(prompt: str, web_search_options: dict) -> Content:
    client = Perplexity()
    response = client.chat.completions.create(
        model="sonar",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "schema": Content.model_json_schema()
            },
        },
        web_search_options=web_search_options,
    )
    return Content.model_validate_json(response.choices[0].message.content)


if __name__ == "__main__":
    load_dotenv()
    print("----------START----------")
    for usecase in PROMPTS:
        print(f"[{datetime.now()}] Use case: {usecase}")
        for prompt in PROMPTS[usecase]:
            print(f"[{datetime.now()}] Prompt: {prompt}")

            # Run prompt against ChatGPT chat and web search
            print(f"[{datetime.now()}] Running Chat GPT...")
            write_event_csv("chatgpt", usecase, prompt, fetch_chatgpt(prompt, []))
            print(f"[{datetime.now()}] Running Chat GPT web search...")
            write_event_csv("chatgpt-websearch", usecase, prompt, fetch_chatgpt(prompt, [{"type": "web_search"}]))

            # Run prompt against Gemini chat and web search
            print(f"[{datetime.now()}] Running Gemini...")
            write_event_csv("gemini", usecase, prompt, fetch_gemini(prompt, []))
            print(f"[{datetime.now()}] Running Gemini websearch...")
            write_event_csv("gemini-websearch", usecase, prompt, fetch_gemini(prompt, [types.Tool(google_search=types.GoogleSearch())]))

            # Run prompt against Perplexity chat and web search
            print(f"[{datetime.now()}] Running Perplexity...")
            write_event_csv("perplexity", usecase, prompt, fetch_perplexity(prompt, {}))
            print(f"[{datetime.now()}] Running Perplexity websearch...")
            write_event_csv("perplexity-websearch", usecase, prompt, fetch_perplexity(prompt, {"search_type": "auto"}))
    print("----------END----------")
