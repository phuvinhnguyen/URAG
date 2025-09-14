from typing import List
from datasets import load_dataset

def get_library_documentation() -> List[str]:
    """Get library documentation"""
    dataset = load_dataset("code-rag-bench/library-documentation", split="train")

    format_doc = '''Title: {doc_id}\nContent: {doc_content}'''
    return [format_doc.format(doc_id=row["doc_id"], doc_content=row["doc_content"]) for row in dataset]

def get_proof_pile() -> List[str]:
    """Get proof-pile"""
    dataset = load_dataset("DKYoon/proofpile2-200k", split="train")
    return [row["input"] for row in dataset]

def get_storage(names: List[str]) -> List[str]:
    """Get storage from name"""
    texts = []
    for name in names:
        if name == "code-rag-bench/library-documentation":
            texts.extend(get_library_documentation())
        elif name == "hoskinson-center/proof-pile":
            texts.extend(get_proof_pile())
    return texts