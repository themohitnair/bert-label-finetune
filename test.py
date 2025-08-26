import asyncpg
import asyncio
import json
import logging
import time
import datetime
import traceback
import ollama
import re
from typing import List
from pydantic import BaseModel

DB_CONN_STR = "postgresql://nfthing_admin:nfthing@157.90.51.8:5432/nfthing"
MODEL_NAME = "gemma3:27b"
POST_LIMIT = 20


class PostAnalysis(BaseModel):
    post_id: str
    purpose: str
    tone: str


class AnalysisResponse(BaseModel):
    analyses: List[PostAnalysis]


logging.basicConfig(
    filename="ollama_processing.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_table_name(model_name):
    """Convert model name to valid PostgreSQL table name"""
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", model_name.lower())
    sanitized = re.sub(r"_{2,}", "_", sanitized)
    sanitized = sanitized.strip("_")
    return f"ollama_{sanitized}"


async def fetch_posts(pool):
    logger.info("Fetching posts from database")
    query = f"""
    SELECT post_id, type, source, description, content_url, canonical_url, image, handle, time, followers
    FROM public.social_search_2
    WHERE time IS NOT NULL AND description IS NOT NULL
    ORDER BY time DESC LIMIT {POST_LIMIT}
    """

    async with pool.acquire() as con:
        rows = await con.fetch(query)
        posts = [dict(row) for row in rows]
        logger.info(f"Fetched {len(posts)} posts")
        return posts


async def analyze_posts(posts):
    logger.info(f"Analyzing {len(posts)} posts sequentially with model: {MODEL_NAME}")

    analyses = []
    client = ollama.Client(host="http://136.243.7.50:11434")

    for i, post in enumerate(posts, 1):
        logger.info(f"Processing post {i}/{len(posts)}: {post['post_id']}")

        prompt = f"""
Analyze this social media post and determine its Purpose and Tone.

Purpose options: Inform, Request, Opine, Promote, Entertain, Organize, Motivate, Greet
Tone options: Humorous, Anger, Sorrow, Joy, Positive, Negative, Neutral

Post to analyze:
Post ID: {post["post_id"]}
Description: {post["description"]}

Return a JSON object with this exact format:
{{"post_id": "{post["post_id"]}", "purpose": "your_analysis", "tone": "your_analysis"}}
"""

        try:
            response = await asyncio.to_thread(
                client.generate,
                model=MODEL_NAME,
                prompt=prompt,
                options={"temperature": 0.1, "top_p": 0.9},
            )

            # Parse the response
            result = json.loads(response["response"])
            analyses.append(result)

            logger.info(
                f"✅ Post {i}: {result.get('purpose', 'N/A')} / {result.get('tone', 'N/A')}"
            )

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error for post {post['post_id']}: {e}")
            # Add empty analysis so we don't break the flow
            analyses.append({"post_id": post["post_id"], "purpose": "", "tone": ""})
        except Exception as e:
            logger.error(f"❌ Error analyzing post {post['post_id']}: {e}")
            analyses.append({"post_id": post["post_id"], "purpose": "", "tone": ""})

    logger.info(f"Successfully analyzed {len(analyses)} posts")
    return analyses


async def update_database(pool, posts, analyses):
    table_name = get_table_name(MODEL_NAME)
    logger.info(f"Updating database table '{table_name}' with {len(analyses)} analyses")

    async with pool.acquire() as con:
        await con.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            post_id TEXT PRIMARY KEY,
            type TEXT,
            source TEXT,
            description TEXT,
            content_url TEXT,
            canonical_url TEXT,
            image TEXT,
            handle TEXT,
            time BIGINT,
            followers BIGINT,
            purpose TEXT,
            tone TEXT,
            model_used TEXT,
            analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        analysis_map = {a["post_id"]: a for a in analyses}

        records = []
        for post in posts:
            analysis = analysis_map.get(post["post_id"], {})

            time_val = post.get("time")
            if isinstance(time_val, datetime.datetime):
                time_val = int(time_val.timestamp() * 1000)

            records.append(
                (
                    post["post_id"],
                    post.get("type"),
                    post.get("source"),
                    post.get("description"),
                    post.get("content_url"),
                    post.get("canonical_url"),
                    post.get("image"),
                    post.get("handle"),
                    time_val,
                    post.get("followers"),
                    analysis.get("purpose", ""),
                    analysis.get("tone", ""),
                    MODEL_NAME,
                )
            )

        await con.executemany(
            f"""
        INSERT INTO {table_name} (
            post_id, type, source, description, content_url, canonical_url,
            image, handle, time, followers, purpose, tone, model_used
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
        ON CONFLICT (post_id) DO UPDATE SET
            purpose=EXCLUDED.purpose,
            tone=EXCLUDED.tone,
            model_used=EXCLUDED.model_used,
            analyzed_at=CURRENT_TIMESTAMP;
        """,
            records,
        )

        logger.info(f"Updated {len(records)} records in table '{table_name}'")


async def main():
    start_time = time.time()
    table_name = get_table_name(MODEL_NAME)
    logger.info(
        f"Starting structured analysis with model: {MODEL_NAME} -> table: {table_name}"
    )

    try:
        pool = await asyncpg.create_pool(DB_CONN_STR)
        logger.info("Connected to database")

        posts = await fetch_posts(pool)
        if not posts:
            logger.warning("No posts found to process")
            return

        analyses = await analyze_posts(posts)

        await update_database(pool, posts, analyses)
        logger.info(f"Successfully processed {len(analyses)} posts")

        await pool.close()

    except Exception as e:
        logger.error(f"Structured analysis failed: {str(e)}\n{traceback.format_exc()}")
        raise

    total_time = round(time.time() - start_time, 2)
    print(
        f"✅ Structured Analysis Complete: {len(analyses) if 'analyses' in locals() else 0} posts in {total_time}s"
    )
    print(f"Model used: {MODEL_NAME} -> Table: {table_name}")
    logger.info(f"Structured analysis completed in {total_time}s")


if __name__ == "__main__":
    asyncio.run(main())
