import asyncio
import json
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from config import MODEL_API_KEY, MODEL_BASE_URL, MODEL


# --- Label enums ---
class Purpose(str, Enum):
    INFORM = "Inform"
    REQUEST = "Request"
    OPINE = "Opine"
    PROMOTE = "Promote"
    ENTERTAIN = "Entertain"
    ORGANIZE = "Organize"
    MOTIVATE = "Motivate"
    GREET = "Greet"


class Tone(str, Enum):
    HUMOROUS = "Humorous"
    ANGER = "Anger"
    SORROW = "Sorrow"
    JOY = "Joy"
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"


# --- Label enums ---


class Analysis(BaseModel):
    # Add or remove label fields
    purpose: Purpose = Field(description="The main purpose or intent of the text")
    tone: Tone = Field(description="The emotional tone of the text")


client = AsyncOpenAI(
    base_url=MODEL_BASE_URL,
    api_key=MODEL_API_KEY,
)

# Change system prompt to include label fields if required
SYSTEM_PROMPT = """You are an expert social media post analyst. Your task is to analyze text and determine:
1. PURPOSE: What is the main intent or purpose of this text? (one of - Inform, Request, Opine, Promote, Entertain, Organize, Motivate, Greet)
2. TONE: What is the emotional tone or sentiment? (one of Humorous, Anger, Sorrow, Joy, Positive, Negative, Neutral)

Always respond with valid JSON following the provided schema."""


async def analyze_single_text(text: str, index: int) -> tuple[int, Optional[Analysis]]:
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this text: {text}"},
            ],
            extra_body={"guided_json": Analysis.model_json_schema()},
            max_tokens=150,
            temperature=0.3,
        )

        json_content = response.choices[0].message.content
        result_dict = json.loads(json_content)
        result = Analysis(**result_dict)

        return index, result

    except Exception as e:
        print(f"Error processing text at index {index}: {e}")
        return index, None


async def analyze_batch(texts: List[str]) -> List[Optional[Analysis]]:
    tasks = [analyze_single_text(text, i) for i, text in enumerate(texts)]

    results_with_indices = await asyncio.gather(*tasks, return_exceptions=True)

    ordered_results = [None] * len(texts)

    for result in results_with_indices:
        if isinstance(result, tuple) and len(result) == 2:
            index, analysis = result
            if 0 <= index < len(texts):
                ordered_results[index] = analysis
        elif isinstance(result, Exception):
            print(f"Task failed with exception: {result}")

    return ordered_results


async def analyze_texts_in_batches(
    texts: List[str], batch_size: int = 10
) -> List[Optional[Analysis]]:
    all_results = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch = texts[start_idx:end_idx]

        print(f"Processing batch {batch_num + 1}/{total_batches}: {len(batch)} texts")

        try:
            batch_results = await analyze_batch(batch)
            all_results.extend(batch_results)

            if batch_num < total_batches - 1:
                await asyncio.sleep(0.1)

        except Exception as e:
            print(f"Error processing batch {batch_num + 1}: {e}")
            all_results.extend([None] * len(batch))

    return all_results


async def analyze_texts(
    texts: List[str], batch_size: int = 10
) -> List[Optional[Analysis]]:
    if not texts:
        return []

    print(f"Starting analysis of {len(texts)} texts with batch size {batch_size}")
    results = await analyze_texts_in_batches(texts, batch_size)

    assert len(results) == len(texts), (
        f"Mismatch: {len(results)} results for {len(texts)} inputs"
    )

    success_count = sum(1 for r in results if r is not None)
    print(f"Analysis complete: {success_count}/{len(texts)} successful")

    return results
