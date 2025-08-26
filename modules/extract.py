import asyncpg
import logging
import re
from config import (
    PSQL_DB,
    PSQL_DB_HOSTNAME,
    PSQL_DB_PORT,
    PSQL_DB_PWD,
    PSQL_DB_USERNAME,
    PSQL_TABLE,
)

URL_PATTERN = re.compile(
    r"(https?://\S+|www\.\S+|\b[\w-]+\.(?:com|org|net|io|xyz|me|news|info)(?:/\S*)?)",
    re.IGNORECASE,
)
MENTION_PATTERN = re.compile(r"@\w+")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


def replace_urls_with_placeholder(text: str) -> str:
    return URL_PATTERN.sub("[URL]", text) if text else text


def replace_mentions_with_placeholder(text: str) -> str:
    return MENTION_PATTERN.sub("[USER]", text)


def lowercase_text(text: str) -> str:
    return text.lower()


def remove_newlines(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ") if text else text


def normalize_numbers(text: str) -> str:
    # Fixed: removed double asterisk
    return re.sub(r"\d+(\.\d+)?", "[NUM]", text)


def pipeline(text: str) -> str:
    text = replace_urls_with_placeholder(text)
    text = replace_mentions_with_placeholder(text)
    text = normalize_numbers(text)
    text = lowercase_text(text)
    text = remove_newlines(text)
    return text


_pool = None


async def get_connection_pool():
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            user=PSQL_DB_USERNAME,
            password=PSQL_DB_PWD,
            database=PSQL_DB,
            host=PSQL_DB_HOSTNAME,
            port=PSQL_DB_PORT,
            min_size=1,
            max_size=5,
        )
    return _pool


async def fetch_batch(batch_size=10, max_posts=20):
    try:
        pool = await get_connection_pool()
        logging.info("Connected to PostgreSQL")

        offset = 0
        batch_num = 0
        total_yielded = 0
        # configure what platforms to include
        valid_types = ("odysse", "nfthing", "bluesky", "farcaster", "lens")

        while total_yielded < max_posts:
            remaining_needed = max_posts - total_yielded
            current_batch_size = min(batch_size, remaining_needed)

            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""
                    SELECT description
                    FROM {PSQL_TABLE}
                    WHERE type = ANY($3)
                    ORDER BY time DESC
                    LIMIT $1 OFFSET $2
                    """,
                    current_batch_size,
                    offset,
                    valid_types,
                )

            if not rows:
                logging.info("No more rows found - stopping fetch")
                break

            batch_num += 1
            logging.info(f"Processing batch {batch_num} ({len(rows)} rows)")

            cleaned_rows = [
                pipeline(r["description"]) for r in rows if r["description"] is not None
            ]

            if cleaned_rows:
                if total_yielded + len(cleaned_rows) > max_posts:
                    cleaned_rows = cleaned_rows[: max_posts - total_yielded]
                    logging.info(
                        f"Trimming batch to {len(cleaned_rows)} records to reach max_posts limit"
                    )

                yield cleaned_rows
                total_yielded += len(cleaned_rows)

                logging.info(
                    f"Yielded {len(cleaned_rows)} records | Total yielded: {total_yielded}"
                )

            offset += len(rows)

            if total_yielded >= max_posts:
                logging.info(f"Reached max_posts limit ({max_posts}) — stopping fetch.")
                break

        logging.info(
            f"Connection closed. Total batches: {batch_num}, Total records yielded: {total_yielded}"
        )

    except Exception as e:
        logging.error(f"Async connection failed: {e}")
