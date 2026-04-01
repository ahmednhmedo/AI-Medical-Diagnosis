"""
05a_train_test_split_by_illness.py
Split BEFORE augmentation: pick test illnesses, keep originals only.

Input  : MongoDB -> Synonym_Expanded_Illnesses (fields: illness_name, symptoms, causes, warnings, recommendations)
Output : MongoDB -> Train_Illnesses_Base, Base_Illnesses_Test
"""

import random
from pymongo import MongoClient

MONGO_URI   = "mongodb://localhost:27017/"
DB_NAME     = "Medical_Diagnosis"
SRC_COLL    = "Synonym_Expanded_Illnesses"
TRAIN_COLL  = "Train_Illnesses_Base"
TEST_COLL   = "Base_Illnesses_Test"
TEST_RATIO  = 0.20  # 20% of illnesses go to test

random.seed(42)

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
src  = db[SRC_COLL]
train = db[TRAIN_COLL]
test  = db[TEST_COLL]

# read all
docs = list(src.find({}, {"_id": 0}))
if not docs:
    raise RuntimeError(f"No data found in '{SRC_COLL}'")

# group by illness
illnesses = sorted({d["illness_name"] for d in docs})
random.shuffle(illnesses)

n_total = len(illnesses)
n_test  = max(1, int(round(n_total * TEST_RATIO)))

test_illnesses  = set(illnesses[:n_test])
train_illnesses = set(illnesses[n_test:])

# wipe outputs
train.delete_many({})
test.delete_many({})

# route each original record to train/test based on illness_name
train_batch, test_batch = [], []
for d in docs:
    target = test_batch if d["illness_name"] in test_illnesses else train_batch
    target.append(d)

if train_batch:
    train.insert_many(train_batch)
if test_batch:
    test.insert_many(test_batch)

print(f"✅ Split complete.")
print(f"   Illnesses total : {n_total}")
print(f"   Illnesses -> test: {len(test_illnesses)} | train: {len(train_illnesses)}")
print(f"   Docs -> test: {len(test_batch)} | train: {len(train_batch)}")
