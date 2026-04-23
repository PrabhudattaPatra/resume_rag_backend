import os
from uuid import uuid4
from rag.retrievers import doc_store
from rag.ingestion.loaders import load_cgu_md, load_fee_structure_md


def clear_static_docs():
    """
    Deletes all existing static documents from the doc-index in Pinecone.
    Uses metadata filter to only remove FAQ and fee_schedule docs,
    leaving any other doc types untouched.
    """
    print("Clearing old static documents from Pinecone (doc-index)...")
    try:
        doc_store.delete(filter={
            "doc_type": {"$in": ["faq", "fee_schedule"]}
        })
        print("✅ Old static documents cleared successfully.")
    except Exception as e:
        print(f"⚠️  Warning: Could not clear old documents: {e}")
        print("Proceeding with ingestion anyway...")


def ingest_static_docs(data_dir="data"):
    """
    Loads static markdown docs (cgu.md and fee_structure.md) and ingests
    them into the doc-index in Pinecone.
    """
    print("Starting Static Data Ingestion...")
    cgu_path = os.path.join(data_dir, "cgu.md")
    fee_path = os.path.join(data_dir, "fee_structure.md")

    try:
        cgu_splits = load_cgu_md(file_path=cgu_path)
        if cgu_splits:
            uuids = [str(uuid4()) for _ in range(len(cgu_splits))]
            doc_store.add_documents(cgu_splits, ids=uuids)
            print(f"✅ Ingested {len(cgu_splits)} chunks from CGU FAQ into Pinecone.")
    except Exception as e:
        print(f"❌ Error loading CGU FAQ: {e}")

    try:
        fee_splits = load_fee_structure_md(file_path=fee_path)
        if fee_splits:
            uuids = [str(uuid4()) for _ in range(len(fee_splits))]
            doc_store.add_documents(fee_splits, ids=uuids)
            print(f"✅ Ingested {len(fee_splits)} chunks from Fee Structure into Pinecone.")
    except Exception as e:
        print(f"❌ Error loading Fee Structure: {e}")
