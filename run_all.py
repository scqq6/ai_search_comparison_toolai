import os
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel

# Официальные SDK платформ
from openai import OpenAI
from google import genai
from google.genai import types
from anthropic import Anthropic

# Настройка путей для сохранения результатов
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = SCRIPT_DIR / "retrievals.csv"

# Промпты для твоего исследования по Discoverability
PROMPTS = {
    "Board game night": [
        "Berlin Winter Board Game Night 2026",
        "Berlin Winter Board Game Night January 2026"
    ]
}

# Базовая схема данных Pydantic (структура твоего отчета)
class Content(BaseModel):
    content_title: str
    content_summary: str
    content_tracking_token: str
    source_urls: List[str]

def sanitize(value: str) -> str:
    if value is None: return ""
    return str(value).replace("\n", " ").replace("\r", " ").replace("|", "/").strip()

def write_event_csv(platform: str, usecase: str, prompt: str, content: Content):
    file_exists = OUTPUT_CSV.exists()
    with open(OUTPUT_CSV, "a+", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        if not file_exists:
            writer.writerow(["Timestamp", "Platform", "UseCase", "Prompt", "Title", "Summary", "Token", "URLs"])
        writer.writerow([
            datetime.now(),
            sanitize(platform),
            sanitize(usecase),
            sanitize(prompt),
            sanitize(content.content_title),
            sanitize(content.content_summary),
            sanitize(content.content_tracking_token),
            sanitize(",".join(content.source_urls))
        ])

# --- РЕАЛИЗАЦИЯ ИНТЕГРАЦИЙ С AI ---

# 1. ChatGPT (OpenAI)
def fetch_chatgpt(prompt: str) -> Content:
    api_key = os.getenv("OPENAI_API_KEY")
    try:
        client = OpenAI(api_key=api_key)
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format=Content,
        )
        return response.choices[0].message.parsed
    except Exception as e:
        print(f"   [!] OpenAI Error: {str(e)[:60]}")
        return Content(content_title="Error", content_summary="OpenAI Issue", content_tracking_token="N/A", source_urls=[])

# 2. Gemini & Gemini WebSearch (Google)
def fetch_gemini(prompt: str, use_web_search: bool = False) -> Content:
    time.sleep(2)  # Обязательная пауза, чтобы платный API не выдавал ошибку 503
    api_key = os.getenv("GOOGLE_API_KEY")
    try:
        client = genai.Client(api_key=api_key)
        
        if use_web_search:
            # Для веб-поиска убираем строгую схему, чтобы избежать конфликтов и ошибки 400
            config = types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=config
            )
            return Content(
                content_title="Google Search Result",
                content_summary=response.text[:500] if response.text else "No content match",
                content_tracking_token="WEB_SEARCH_ACTIVE",
                source_urls=[]
            )
        else:
            # Для стандартного чата собираем строгий JSON
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=Content.model_json_schema(),
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=config
            )
            return Content.model_validate_json(response.text)
            
    except Exception as e:
        print(f"   [!] Gemini Error: {str(e)[:60]}")
        return Content(content_title="Error", content_summary="Gemini Issue", content_tracking_token="N/A", source_urls=[])

# 3. Claude (Anthropic)
def fetch_claude(prompt: str) -> Content:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    try:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",  # 20240229, 20241022 not working
            max_tokens=1000,
            temperature=0.0,
            system=f"You must respond ONLY with a raw JSON object that matches this schema: {Content.model_json_schema()}. Never wrap output in markdown code blocks.",
            messages=[{"role": "user", "content": prompt}]
        )
        raw_text = response.content[0].text.strip()
        
        # Обрезка markdown-кавычек, если модель их добавит
        if "```" in raw_text:
            start = raw_text.find("{")
            end = raw_text.rfind("}") + 1
            if start != -1 and end != 0:
                raw_text = raw_text[start:end]
                
        return Content.model_validate_json(raw_text)
    except Exception as e:
        print(f"   [!] Claude Error: {str(e)[:60]}")
        return Content(content_title="Error", content_summary="Claude Issue", content_tracking_token="N/A", source_urls=[])

# --- ГЛАВНЫЙ ЦИКЛ ЗАПУСКА ---

if __name__ == "__main__":
    load_dotenv()
    print("----------START----------")
    
    # Сбрасываем старую таблицу, чтобы отчет наполнялся только чистыми данными
    if OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()
        
    for usecase in PROMPTS:
        print(f"\n[{datetime.now()}] Use case: {usecase}")
        for prompt in PROMPTS[usecase]:
            print(f" -> Prompt: {prompt}")

            # 1. Сбор данных с ChatGPT
            print("    Running ChatGPT...")
            write_event_csv("chatgpt", usecase, prompt, fetch_chatgpt(prompt))

            # 2. Сбор данных со стандартного Gemini
            print("    Running Gemini...")
            write_event_csv("gemini", usecase, prompt, fetch_gemini(prompt, use_web_search=False))
            
            # 3. Сбор данных с Gemini WebSearch (Google Поиск)
            print("    Running Gemini WebSearch...")
            write_event_csv("gemini-websearch", usecase, prompt, fetch_gemini(prompt, use_web_search=True))

            # 4. Сбор данных с Claude
            print("    Running Claude...")
            write_event_csv("claude", usecase, prompt, fetch_claude(prompt))
            
    print("\n----------FINISHED! All data successfully saved to retrievals.csv ----------")
