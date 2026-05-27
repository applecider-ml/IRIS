"""
iris.fritz.pipeline
===================
High-level pipeline: client -> processing -> storage.

Supports single/batch download and auto-discovery of Fritz sources.
"""

import logging
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from iris.fritz.client import FritzClient
from iris.fritz.data_processing import build_object_result
from iris.fritz.storage import save_object

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def process_object(
    client: FritzClient,
    obj_id: str,
    output_dir: str = "fritz_data",
    request_delay: float = 0.5,
) -> "tuple[str, dict | None]":
    """
    Download source + photometry + spectra for one object.

    Returns (status, result) where status is "success", "skipped", or "failed".
    """
    try:
        source = client.get_source(obj_id)
        time.sleep(request_delay)
        phot = client.get_photometry(obj_id)
        time.sleep(request_delay)
        spectra = client.get_spectra(obj_id)

        if not spectra:
            logger.info(f"[{obj_id}] no spectra, skipping.")
            return "skipped", None

        result = build_object_result(source, phot, spectra, obj_id)
        paths  = save_object(result, output_dir=output_dir)
        logger.info(f"[{obj_id}] saved {len(paths)} spectrum file(s).")
        return "success", result

    except Exception as e:
        logger.error(f"[{obj_id}] failed: {e}")
        return "failed", None


def discover_source_ids(client: FritzClient, page_size: int = 100, **filters) -> "list[str]":
    """Paginate through /api/sources and return all obj_ids."""
    all_ids = []
    page    = 1

    while True:
        data    = client.list_sources(page=page, num_per_page=page_size, **filters)
        sources = data.get("sources", [])

        if not sources:
            break

        all_ids.extend(s["id"] for s in sources)
        total = data.get("totalMatches", len(all_ids))
        logger.info(f"Discovered {len(all_ids)}/{total} sources...")

        if len(all_ids) >= total:
            break
        page += 1

    return all_ids


def run_pipeline(
    client: FritzClient,
    obj_ids: "list[str]",
    output_dir: str = "fritz_data",
    max_workers: int = 4,
    request_delay: float = 0.5,
) -> dict:
    """Download and save all objects using a thread pool."""
    summary = {"success": [], "skipped": [], "failed": []}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(process_object, client, obj_id, output_dir, request_delay): obj_id
            for obj_id in obj_ids
        }

        for future in as_completed(futures):
            obj_id = futures[future]
            try:
                status, _ = future.result()
            except Exception as e:
                logger.error(f"[{obj_id}] unexpected error: {e}")
                status = "failed"
            summary[status].append(obj_id)

    if summary["failed"]:
        os.makedirs(output_dir, exist_ok=True)
        failed_path = os.path.join(output_dir, "failed_ids.txt")
        with open(failed_path, "a") as f:
            for obj_id in summary["failed"]:
                f.write(obj_id + "\n")
        logger.warning(f"{len(summary['failed'])} failed IDs saved to {failed_path}")

    logger.info(
        f"Done. success={len(summary['success'])}  "
        f"skipped={len(summary['skipped'])}  "
        f"failed={len(summary['failed'])}"
    )
    return summary


def run_full_pipeline(
    client: FritzClient,
    output_dir: str = "fritz_data",
    max_workers: int = 4,
    page_size: int = 100,
    request_delay: float = 0.5,
    **source_filters,
) -> dict:
    """Discover all sources with spectra and download them."""
    source_filters.setdefault("hasSpectrum", True)
    obj_ids = discover_source_ids(client, page_size=page_size, **source_filters)
    logger.info(f"Starting download for {len(obj_ids)} sources -> {output_dir}")
    return run_pipeline(
        client, obj_ids, output_dir=output_dir,
        max_workers=max_workers, request_delay=request_delay,
    )
