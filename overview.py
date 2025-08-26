import pyarrow as pa
import pyarrow.parquet as pq
import os
import logging
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def store_to_parquet(
    records: List[Dict[str, Any]], filename: str = "output.parquet"
) -> Optional[str]:
    try:
        valid_records = [r for r in records if r is not None]

        if not valid_records:
            logger.error("No valid records to store")
            return None

        logger.info(f"Storing {len(valid_records)} records")

        table = pa.Table.from_pydict(
            {
                "description": [r["description"] for r in valid_records],
                # you may add fields here
                "purpose": [r["purpose"] for r in valid_records],
                "tone": [r["tone"] for r in valid_records],
            }
        )

        if os.path.exists(filename):
            os.remove(filename)

        pq.write_table(table, filename)

        logger.info(
            f"✅ Successfully stored {len(valid_records)} records to {filename}"
        )
        return filename

    except Exception as e:
        logger.error(f"Failed to store records: {e}")
        return None
