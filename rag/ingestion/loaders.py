from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

def clean_text(text: str) -> str:
    # Remove or replace invalid surrogate characters
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")

def load_cgu_md(file_path="data/cgu.md"):
    """
    Loads and splits the main CGU markdown FAQ file.
    """
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    headers_to_split_on = [
        ("##", "Question Header"), 
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    splits = splitter.split_text(docs[0].page_content)

    for doc in splits:
        doc.metadata.update({
            "source": "cgu.md", 
            "doc_type": "faq",
            "university": "CGU"
        })
        doc.page_content = clean_text(doc.page_content)

    return splits

def load_fee_structure_md(file_path="data/fee_structure.md"):
    """
    Loads and splits the fee structure markdown file.
    """
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    headers_to_split_on = [
        ("#", "Doc_Title"),
        ("##", "Fee_Category"),  
    ]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    splits = splitter.split_text(docs[0].page_content)

    for chunk in splits:
        chunk.metadata.update({
            "source": "fee_structure.md",
            "doc_type": "fee_schedule",
            "academic_year": "2025-26",
            "currency": "INR" if "USD" not in chunk.page_content else "USD"
        })
        chunk.page_content = clean_text(chunk.page_content)

    return splits
