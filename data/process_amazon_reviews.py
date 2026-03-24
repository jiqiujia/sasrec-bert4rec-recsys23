"""
Amazon Reviews Dataset Preprocessor
====================================

This script processes raw Amazon Reviews data (2014 format) into the interaction
file format used by sasrec-bert4rec-recsys23.

Input:
    reviews_{dataset_name}.json.gz
        - Raw Amazon Reviews file in gzipped JSON format (one JSON object per line).
        - Each record contains fields: reviewerID, asin, overall, unixReviewTime, etc.
        - Download from: http://jmcauley.ucsd.edu/data/amazon/

    meta_{dataset_name}.json.gz  (optional but recommended)
        - Raw Amazon product metadata file in gzipped JSON format.
        - Each record contains fields: asin, title, description, categories, brand, price, etc.
        - Download from the same source as the reviews file.

Output:
    1. reviews_{dataset_name}.txt
        - Intermediate file with all raw interactions (for debugging/inspection).
        - Each line: "reviewerID asin rating unixReviewTime"

    2. {dataset_name}.txt  (e.g. Beauty.txt -> beauty.txt)
        - Final interaction file used for model training.
        - Each line: "user_id item_id"
        - user_id and item_id are remapped to contiguous integers starting from 1.
        - Lines are grouped by user_id, and within each user sorted by timestamp (ascending).

    3. {dataset_name}_item_metadata.json  (e.g. beauty_item_metadata.json)
        - Item metadata file keyed by the remapped integer item_id.
        - Each entry contains: asin, title, description, categories, brand, price.

Processing Pipeline:
    ┌─────────────────────────────────────────────────────────────────┐
    │  Pass 1: Count interactions per user and per item              │
    │          Write intermediate file (reviews_{name}.txt)          │
    ├─────────────────────────────────────────────────────────────────┤
    │  Pass 2: 5-core filtering (keep users & items with >= 5       │
    │          interactions), remap IDs to contiguous integers,      │
    │          collect per-user interaction sequences                 │
    ├─────────────────────────────────────────────────────────────────┤
    │  Sort each user's interactions by timestamp                    │
    ├─────────────────────────────────────────────────────────────────┤
    │  Write final output file ({name}.txt)                          │
    ├─────────────────────────────────────────────────────────────────┤
    │  Load metadata, remap item IDs, write metadata JSON            │
    └─────────────────────────────────────────────────────────────────┘

Note:
    - The 5-core filtering is a SINGLE PASS filter (not iterative k-core).
      It removes interactions where the user has < 5 reviews OR the item has
      < 5 reviews based on the ORIGINAL counts. This means some users/items
      in the output may end up with fewer than 5 interactions after filtering.
    - This is consistent with the original SASRec paper's preprocessing.

Usage:
    python process_amazon_reviews.py

    Make sure reviews_Beauty.json.gz and meta_Beauty.json.gz are in the same directory.
    Change `dataset_name` variable below to process other Amazon categories.
"""

import gzip
import json
import os
from collections import defaultdict
from datetime import datetime


def parse(path):
    """Read a gzipped JSON file line by line, yielding parsed Python dicts.

    The Amazon 2014 dataset uses a non-standard JSON format (Python repr),
    so we use eval() to parse each line instead of json.loads().
    """
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


# ============================================================================
# Stage 1: Count interactions per user (countU) and per item (countP)
#
# Purpose: Determine which users and items have >= 5 interactions,
#          so we can apply 5-core filtering in Stage 2.
# Side effect: Write an intermediate text file with all raw interactions
#              for inspection/debugging.
# ============================================================================

countU = defaultdict(lambda: 0)  # reviewerID -> number of reviews
countP = defaultdict(lambda: 0)  # asin -> number of reviews
line = 0

dataset_name = 'Beauty'

# Write intermediate file: "reviewerID asin rating timestamp" per line
f = open('reviews_' + dataset_name + '.txt', 'w')
for l in parse('reviews_' + dataset_name + '.json.gz'):
    line += 1
    f.write(" ".join([l['reviewerID'], l['asin'], str(l['overall']), str(l['unixReviewTime'])]) + ' \n')
    asin = l['asin']
    rev = l['reviewerID']
    time = l['unixReviewTime']
    countU[rev] += 1
    countP[asin] += 1
f.close()

# ============================================================================
# Stage 2: 5-core filtering + ID remapping + collect user sequences
#
# Second pass over the raw data:
#   - Skip any interaction where the user has < 5 total reviews
#     OR the item has < 5 total reviews (based on original counts from Stage 1).
#   - Remap raw string IDs (reviewerID, asin) to contiguous integer IDs
#     starting from 1, in order of first appearance.
#   - Collect each user's interactions as a list of [timestamp, item_id] pairs.
# ============================================================================

usermap = dict()   # reviewerID (str) -> new integer user_id
usernum = 0        # counter for assigning new user IDs
itemmap = dict()   # asin (str) -> new integer item_id
itemnum = 0        # counter for assigning new item IDs
User = dict()      # user_id (int) -> list of [timestamp, item_id] pairs

for l in parse('reviews_' + dataset_name + '.json.gz'):
    line += 1
    asin = l['asin']
    rev = l['reviewerID']
    time = l['unixReviewTime']

    # 5-core filter: skip if user or item has fewer than 5 interactions
    if countU[rev] < 5 or countP[asin] < 5:
        continue

    # Remap user ID: assign a new integer ID on first encounter
    if rev in usermap:
        userid = usermap[rev]
    else:
        usernum += 1
        userid = usernum
        usermap[rev] = userid
        User[userid] = []

    # Remap item ID: assign a new integer ID on first encounter
    if asin in itemmap:
        itemid = itemmap[asin]
    else:
        itemnum += 1
        itemid = itemnum
        itemmap[asin] = itemid

    # Record this interaction: [timestamp, item_id]
    User[userid].append([time, itemid])

# ============================================================================
# Stage 3: Sort each user's interaction sequence by timestamp (ascending)
#
# This ensures chronological order, which is critical for sequential
# recommendation models (SASRec, BERT4Rec, etc.) that treat the item
# sequence as a time-ordered history.
# ============================================================================

for userid in User.keys():
    User[userid].sort(key=lambda x: x[0])

print(usernum, itemnum)

# ============================================================================
# Stage 4: Write the final output file
#
# Format: each line is "user_id item_id"
#   - Lines are grouped by user_id (dict insertion order in Python 3.7+)
#   - Within each user, items appear in chronological order (sorted in Stage 3)
#   - This is the exact format expected by sasrec-bert4rec-recsys23's run.py:
#       pd.read_csv(path, sep=' ', header=None, names=['user_id', 'item_id'])
# ============================================================================

f = open(f'{dataset_name.lower()}.txt', 'w')
for user in User.keys():
    for i in User[user]:
        f.write('%d %d\n' % (user, i[1]))
f.close()

# ============================================================================
# Stage 5: Load metadata and build remapped item_id -> metadata mapping
#
# Reads the product metadata file (meta_{dataset_name}.json.gz), which contains
# item attributes such as title, description, categories, brand, price, etc.
#
# Using the itemmap (asin -> integer item_id) built in Stage 2, we create a
# JSON file keyed by the new integer item_id, so downstream tasks (e.g., GRLM,
# text-based recommendation) can look up item descriptions by the same IDs
# used in the interaction file.
#
# Input:  meta_{dataset_name}.json.gz
# Output: {dataset_name}_item_metadata.json
#
# Output format (similar to GRLM's beauty.item.json):
# {
#   "1": {
#     "asin": "B000FI4S1E",
#     "title": "Phyto Phytocitrus Restructuring Mask ...",
#     "description": "True Colors ...",
#     "categories": "Beauty > Hair Care > Conditioners",
#     "brand": "Phyto",
#     "price": "$29.00"
#   },
#   ...
# }
# ============================================================================

meta_path = 'meta_' + dataset_name + '.json.gz'
metadata_output_path = dataset_name.lower() + '_item_metadata.json'

if os.path.exists(meta_path):
    print(f'Loading metadata from {meta_path} ...')

    # Build a raw metadata dict: asin -> {title, description, categories, brand, price}
    raw_metadata = {}
    for entry in parse(meta_path):
        asin = entry.get('asin', '')
        if not asin:
            continue

        # Extract title
        title = entry.get('title', '')

        # Extract description (may be a list of strings in 2014 format)
        description = entry.get('description', '')
        if isinstance(description, list):
            description = ' '.join(str(d) for d in description if d)

        # Extract categories (2014 format: list of lists, e.g. [["Beauty", "Hair Care", ...]])
        categories_raw = entry.get('categories', [])
        if isinstance(categories_raw, list) and categories_raw:
            if isinstance(categories_raw[0], list):
                # 2014 format: take the first category path
                categories = ' > '.join(str(c) for c in categories_raw[0] if c)
            else:
                categories = ' > '.join(str(c) for c in categories_raw if c)
        else:
            categories = str(categories_raw) if categories_raw else ''

        # Extract brand and price
        brand = entry.get('brand', '')
        price = entry.get('price', '')

        raw_metadata[asin] = {
            'asin': asin,
            'title': title,
            'description': description,
            'categories': categories,
            'brand': brand,
            'price': price,
        }

    print(f'Loaded metadata for {len(raw_metadata)} items from {meta_path}')

    # Build remapped metadata: new integer item_id -> metadata dict
    # Only include items that survived the 5-core filtering (i.e., in itemmap)
    remapped_metadata = {}
    missing_count = 0

    for asin, new_id in itemmap.items():
        if asin in raw_metadata:
            remapped_metadata[str(new_id)] = raw_metadata[asin]
        else:
            # Item in reviews but not in metadata — create empty placeholder
            missing_count += 1
            remapped_metadata[str(new_id)] = {
                'asin': asin,
                'title': '',
                'description': '',
                'categories': '',
                'brand': '',
                'price': '',
            }

    # Write the remapped metadata file
    with open(metadata_output_path, 'w', encoding='utf-8') as f:
        json.dump(remapped_metadata, f, ensure_ascii=False, indent=2)

    print(f'Saved remapped item metadata ({len(remapped_metadata)} items) to {metadata_output_path}')
    if missing_count > 0:
        print(f'  WARNING: {missing_count} items had no metadata (empty placeholders created)')
else:
    print(f'Metadata file not found: {meta_path}')
    print(f'  Skipping metadata output. To generate {metadata_output_path},')
    print(f'  download meta_{dataset_name}.json.gz and place it in the same directory.')