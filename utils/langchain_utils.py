from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
from config import settings
from utils.chroma_utils import get_chroma_db
import logging


logger = logging.getLogger(__name__)




def get_llm():
    try:
        return ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            temperature=0.3,
            google_api_key=settings.gemini_api_key
        )
    except Exception as e:
        logger.error(f"LLM initialization error: {e}")
        raise




def get_rag_chain():
    from config import settings
    db = get_chroma_db()


    retriever = db.as_retriever(search_kwargs={"k": settings.retrieval_k})


    template = """
You are a helpful assistant for Zendalona, a company providing accessibility solutions.
Answer the question based only on the following context:
{context}


Question: {question}


Instructions:
- Answer the question directly using information from the context
- If asked about Zendalona in general, provide a concise 1-2 sentence definition
- If asked about specific products, list them with brief descriptions using bullet points
- If asked for instructions or how-to information, provide clear step-by-step guidance
- Use simple bullet points with "* " at the start of each item when listing multiple items
- Put product names in CAPITAL LETTERS
- Put a colon and space after product names
- Each bullet point should be on its own line
- Do not use markdown formatting like **bold**
- Write "Zendalona" correctly
- Keep responses focused and under 10 sentences
- NEVER start with greetings
- If unknown say:
"I don't have specific information about that"
"""


    prompt = PromptTemplate.from_template(template)


    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )


    return chain




def get_streaming_chain():
    from config import settings
    db = get_chroma_db()


    retriever = db.as_retriever(search_kwargs={"k": settings.retrieval_k})


    template = """
You are a helpful assistant for Zendalona, a company providing accessibility solutions.
Answer the question based only on the following context:
{context}


Question: {question}
"""


    prompt = PromptTemplate.from_template(template)


    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )


    return chain




def process_query(chain, query):


    try:
        # Vector DB retrieval
        all_docs_with_scores = get_chroma_db().similarity_search_with_score(
            query, k=settings.retrieval_k
        )


        filtered_docs = [
            doc for doc, score in all_docs_with_scores
            if score <= settings.retrieval_threshold
        ]


        docs = filtered_docs[:settings.max_context_docs] if filtered_docs else []


        sources = [
            doc.metadata.get("source", "")
            for doc in docs if doc.metadata.get("source")
        ]


    except Exception as e:
        logger.error(f"Embedding error: {e}")
        msg = str(e).lower()


        if "api key" in msg:
            return "⚠️ AI service configuration issue.", []


        if "quota" in msg or "429" in msg:
            return "⚠️ AI quota exceeded. Try again later.", []


        return "⚠️ Knowledge retrieval error.", []


    try:
        # Gemini call
        response = chain.invoke(query)


    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        msg = str(e).lower()


        if "quota" in msg or "429" in msg:
            response = "⚠️ AI quota reached. Please try later."


        elif "api key" in msg:
            response = "⚠️ AI configuration issue."


        elif "timeout" in msg:
            response = "⚠️ AI response timeout."


        else:
            response = "⚠️ AI service temporarily unavailable."


    return response, sources



