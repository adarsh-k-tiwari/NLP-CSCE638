from config import DOCUMENTS

def simple_retrieval(query, top_k=2):
    """
    A very simple retrieval function that returns the top_k documents
    that match keywords in the query.
    """
    query_keywords = query.lower().split()
    scored_docs = []
    for doc in DOCUMENTS:
        score = sum(word in doc["content"].lower() for word in query_keywords)
        if score > 0:
            scored_docs.append((score, doc))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:top_k]]
