import asyncio
import logging
import time
import argparse
from typing import List, Dict, Any
from modules.extract import fetch_batch
from modules.model import analyze_texts
from modules.store import store_to_parquet


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


def count_tokens(text: str) -> int:
    return len(text) // 4


async def process_batch_worker(
    batch_id: int,
    text_batch: List[str],
    analysis_batch_size: int,
    semaphore: asyncio.Semaphore,
) -> tuple[int, List[Dict[str, Any]], int, int]:
    """Process a single batch with just text analysis."""

    async with semaphore:
        batch_tokens = sum(count_tokens(text) for text in text_batch)
        logging.info(
            f"[Batch {batch_id}] Processing {len(text_batch)} texts, {batch_tokens:,} tokens"
        )

        try:
            analyses = await analyze_texts(text_batch, batch_size=analysis_batch_size)
            logging.info(
                f"[Batch {batch_id}] Completed analysis for {len(analyses)} texts"
            )

            batch_records = []
            failed_count = 0

            for i, (text, analysis) in enumerate(zip(text_batch, analyses)):
                if analysis is not None:
                    record = {
                        "description": text,
                        # Add your label columns here
                        "purpose": analysis.purpose,
                        "tone": analysis.tone,
                    }
                    batch_records.append(record)
                else:
                    failed_count += 1
                    logging.warning(
                        f"[Batch {batch_id}] Skipping record {i} due to failed analysis"
                    )

            success_rate = (
                (len(batch_records) / len(text_batch)) * 100 if text_batch else 0
            )
            logging.info(
                f"[Batch {batch_id}] ✅ Complete: {len(batch_records)} success, {failed_count} failed ({success_rate:.1f}% success rate)"
            )
            return batch_id, batch_records, batch_tokens, failed_count

        except Exception as e:
            logging.error(f"[Batch {batch_id}] ❌ Failed: {e}")
            return batch_id, [], 0, len(text_batch)


async def process_pipeline(
    max_posts: int = 1000,
    fetch_batch_size: int = 100,
    analysis_batch_size: int = 10,
    max_concurrent_batches: int = 8,
    output_file: str = "labelled_posts.parquet",
):
    """
    Simplified pipeline that:
    1. Fetches posts from database
    2. Processes multiple batches concurrently
    3. Analyzes texts (no embedding generation)
    4. Stores results and tracks failures
    """
    all_records = []
    total_tokens = 0
    total_failed_posts = 0
    start_time = time.time()

    semaphore = asyncio.Semaphore(max_concurrent_batches)

    logging.info("🚀 Starting analysis-only pipeline:")
    logging.info(f"   • Target posts: {max_posts:,}")
    logging.info(f"   • Max concurrent batches: {max_concurrent_batches}")
    logging.info(f"   • Analysis batch size: {analysis_batch_size}")

    try:
        batches = []
        batch_id = 0
        total_input_posts = 0

        async for text_batch in fetch_batch(
            batch_size=fetch_batch_size, max_posts=max_posts
        ):
            if text_batch:
                batches.append((batch_id, text_batch))
                total_input_posts += len(text_batch)
                batch_id += 1

        logging.info(
            f"📦 Collected {len(batches)} batches ({total_input_posts} total posts) for processing"
        )

        if not batches:
            logging.warning("No batches to process!")
            return {
                "records": 0,
                "tokens": 0,
                "time": 0,
                "failed_posts": 0,
                "records_per_second": 0,
                "tokens_per_second": 0,
            }

        tasks = [
            process_batch_worker(batch_id, text_batch, analysis_batch_size, semaphore)
            for batch_id, text_batch in batches
        ]

        completed_tasks = 0

        for completed_task in asyncio.as_completed(tasks):
            batch_id, batch_records, batch_tokens, batch_failed = await completed_task
            completed_tasks += 1

            all_records.extend(batch_records)
            total_tokens += batch_tokens
            total_failed_posts += batch_failed

            elapsed_time = time.time() - start_time
            progress = (completed_tasks / len(tasks)) * 100

            logging.info(
                f"📊 Progress: {completed_tasks}/{len(tasks)} ({progress:.1f}%) | "
                f"Records: {len(all_records)} | Failed: {total_failed_posts} | "
                f"Tokens: {total_tokens:,} | Time: {elapsed_time:.1f}s"
            )

            if len(all_records) >= 500 and len(all_records) % 100 < len(batch_records):
                try:
                    store_to_parquet(all_records, f"temp_{output_file}")
                    logging.info(
                        f"💾 Saved intermediate results: {len(all_records)} records"
                    )
                except Exception as e:
                    logging.error(f"Failed to save intermediate results: {e}")

        total_time = time.time() - start_time

        if all_records:
            try:
                final_file = store_to_parquet(all_records, output_file)

                total_processed = len(all_records) + total_failed_posts
                success_rate = (
                    (len(all_records) / total_processed) * 100
                    if total_processed > 0
                    else 0
                )
                avg_tokens_per_record = (
                    total_tokens / len(all_records) if all_records else 0
                )
                records_per_second = (
                    len(all_records) / total_time if total_time > 0 else 0
                )
                tokens_per_second = total_tokens / total_time if total_time > 0 else 0

                print("\n" + "=" * 70)
                print("🎯 ANALYSIS-ONLY PIPELINE COMPLETION SUMMARY")
                print("=" * 70)
                print(
                    f"✅ Successfully saved {len(all_records)} records to {final_file}"
                )
                print(f"❌ Failed posts: {total_failed_posts}")
                print(
                    f"📈 Success rate: {success_rate:.1f}% ({len(all_records)}/{total_processed})"
                )
                print(
                    f"⏱️  Total time: {total_time:.2f} seconds ({total_time / 60:.1f} minutes)"
                )
                print(f"🔤 Total tokens processed: {total_tokens:,}")
                print(f"📊 Average tokens per record: {avg_tokens_per_record:.1f}")
                print(f"🚀 Processing speed: {records_per_second:.1f} records/sec")
                print(f"⚡ Token throughput: {tokens_per_second:,.0f} tokens/sec")
                print(f"🧵 Concurrency: {max_concurrent_batches} batches")
                print("=" * 70)

                logging.info(
                    f"✅ Analysis pipeline complete! {len(all_records)} records | "
                    f"{total_failed_posts} failed | {total_tokens:,} tokens | {total_time:.2f}s"
                )

            except Exception as e:
                logging.error(f"Failed to save final results: {e}")
        else:
            logging.warning("No records to save!")

        return {
            "records": len(all_records),
            "failed_posts": total_failed_posts,
            "total_processed": len(all_records) + total_failed_posts,
            "success_rate": (len(all_records) / (len(all_records) + total_failed_posts))
            * 100
            if (len(all_records) + total_failed_posts) > 0
            else 0,
            "tokens": total_tokens,
            "time": total_time,
            "records_per_second": len(all_records) / total_time
            if total_time > 0
            else 0,
            "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
            "batches_processed": len(batches),
            "concurrent_batches": max_concurrent_batches,
        }

    except Exception as e:
        logging.error(f"Analysis pipeline failed: {e}")
        raise


async def main():
    parser = argparse.ArgumentParser(description="Text analysis pipeline CLI")
    parser.add_argument(
        "--max-posts",
        type=int,
        default=10000,
        help="Maximum number of posts to process",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="labelled.parquet",
        help="Output Parquet file name",
    )
    args = parser.parse_args()

    try:
        stats = await process_pipeline(
            max_posts=args.max_posts,
            fetch_batch_size=100,
            analysis_batch_size=10,
            max_concurrent_batches=8,
            output_file=args.output_file,
        )

        print("\n🎯 Final Analysis-Only Stats:")
        print(f"   • {stats['records']} successful records")
        print(f"   • {stats['failed_posts']} failed posts")
        print(f"   • {stats['success_rate']:.1f}% success rate")
        print(f"   • {stats['tokens']:,} tokens processed")
        print(f"   • {stats['time']:.2f}s total time")
        print(f"   • {stats['records_per_second']:.1f} records/sec")
        print(
            f"   • {stats['batches_processed']} batches with {stats['concurrent_batches']} max concurrent"
        )

    except Exception as e:
        logging.error(f"Analysis pipeline failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
