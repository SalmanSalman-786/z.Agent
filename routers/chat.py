import json
import logging
from fastapi import APIRouter, HTTPException, Depends, Body, WebSocket
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from typing import Dict, Any, AsyncGenerator, Optional, List
from utils.models import ChatRequest, ChatResponse, StreamingChatRequest, FeedbackRequest
from utils.langchain_utils import get_rag_chain, process_query, get_streaming_chain
from utils.cache_utils import get_from_cache, cache_chatbot_response
from utils.mongo_utils import save_feedback, save_to_temp_cache
from config import settings
import asyncio
import uuid
import time


router = APIRouter(prefix="/chat", tags=["Chat"])


# Track active chat sessions
active_sessions: Dict[str, Dict[str, Any]] = {}


from utils.chroma_utils import get_chroma_db


@router.post(
    "",
    response_model=ChatResponse,
    summary="Get a complete chat response",
    description="Process a chat query and return a complete response with sources",
    response_description="A chat response with answer and sources",
)
async def chat(request: ChatRequest):
    """
    Process a chat query and return a complete response with sources.
   
    - **query**: The user's question or message
    - **session_id**: Optional unique identifier for the chat session
    """
    try:
        logging.info(f"Checking cache for query: {request.query}")
        cached_results = get_from_cache(request.query)
        logging.info(f"Cache results: {len(cached_results) if cached_results else 0} items")
       
        if cached_results:
            top_hit, score = cached_results[0]
            logging.info(f"Top hit: {top_hit.page_content[:50]}..., score: {score}")
            similarity = 1.0 - score
            # Use a more lenient threshold for short queries (like greetings)
            # For longer queries, use a more reasonable threshold
            threshold = 0.65 if len(request.query) > 10 else 0.50
            logging.info(f"Similarity: {similarity}, threshold: {threshold}")
            if similarity >= threshold:
                try:
                    logging.info("Cache match found, returning cached response")
                    answer = top_hit.metadata.get("answer", "")
                    sources = [f"[CACHED - {top_hit.metadata.get('source', 'unknown')} - similarity: {similarity:.2f}]"]
                    # Save to temporary cache with source information
                    await save_to_temp_cache(request.query, answer, sources, source="cache")
                    logging.info("Returning cached response")
                    return ChatResponse(response=answer, sources=sources, suggestions=[], feedback_enabled=True)
                except Exception as e:
                    logging.error(f"Error processing cached response: {str(e)}")
                    # Even if there's an error with saving to cache,
                    # we should still return the cached answer
                    answer = top_hit.metadata.get("answer", "")
                    sources = [f"[CACHED - {top_hit.metadata.get('source', 'unknown')} - similarity: {similarity:.2f}]"]
                    logging.info("Returning cached response despite error")
                    return ChatResponse(response=answer, sources=sources, suggestions=[], feedback_enabled=True)


        # Check if Gemini calls are disabled
        if settings.disable_gemini_call:
            logging.info("Gemini calls disabled, returning formal message")
            # Return formal message when no cache match and Gemini is disabled
            answer = "We don't have much information about that please ask chatgpt regarding this"
            sources = ["[NO_MATCH_IN_CACHE]"]
            # Save to temporary cache with source information
            await save_to_temp_cache(request.query, answer, sources, source="no_match")
            return ChatResponse(response=answer, sources=sources, suggestions=[], feedback_enabled=False)


        logging.info("Calling Gemini API")
        chain = get_rag_chain()
        response, sources = process_query(chain, request.query)
        logging.info(f"Processed query with Gemini: {request.query}")
       
        # Save to temporary cache with source information
        await save_to_temp_cache(request.query, response, sources, source="gemini")
       
        return ChatResponse(response=response, sources=sources, suggestions=[], feedback_enabled=True)
    except Exception as e:
        logging.error(f"Error processing chat query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_response(query: str, session_id: str) -> AsyncGenerator[dict, None]:
    """Generate a streaming response for the chat query."""
    try:
        # Check cache first with dynamic threshold based on query length
        logging.info(f"Checking cache for streaming query: {query}")
        cached_results = get_from_cache(query)
        logging.info(f"Cache results: {len(cached_results) if cached_results else 0} items")
        if cached_results:
            top_hit, score = cached_results[0]
            logging.info(f"Top hit: {top_hit.page_content[:50]}..., score: {score}")
            similarity = 1.0 - score
            # Use a more lenient threshold for short queries (like greetings)
            # For longer queries, use a more reasonable threshold
            threshold = 0.65 if len(query) > 10 else 0.40
            logging.info(f"Similarity: {similarity}, threshold: {threshold}")
            if similarity >= threshold:
                logging.info(f"Streaming cached response for query: {query}")
                answer = top_hit.metadata.get("answer", "")
                sources = [f"[CACHED - {top_hit.metadata.get('source', 'unknown')} - similarity: {similarity:.2f}]"]


                # Stream the cached answer word by word
                words = answer.split()
                for word in words:
                    yield {"event": "message", "data": f"{word} "}
                    await asyncio.sleep(0.05)
               
                yield {"event": "sources", "data": ",".join(sources)}
                yield {"event": "suggestions", "data": ""}
                yield {"event": "metadata", "data": json.dumps({"feedback_enabled": True})}
                yield {"event": "done", "data": ""}
               
                # Save to temporary cache with source information
                await save_to_temp_cache(query, answer, sources, source="cache")
                logging.info("Finished streaming cached response")
                return
            else:
                logging.info("Cache match found but similarity below threshold")


        # Check if Gemini calls are disabled
        if settings.disable_gemini_call:
            logging.info("Gemini calls disabled, returning formal message")
            # Return formal message when no cache match and Gemini is disabled
            answer = "We don't have much information about that please ask chatgpt regarding this"
            sources = ["[NO_MATCH_IN_CACHE]"]
           
            # Stream the formal message word by word
            words = answer.split()
            for word in words:
                yield {"event": "message", "data": f"{word} "}
                await asyncio.sleep(0.05)
           
            yield {"event": "sources", "data": ",".join(sources)}
            yield {"event": "suggestions", "data": ""}
            yield {"event": "metadata", "data": json.dumps({"feedback_enabled": False})}
            yield {"event": "done", "data": ""}
           
            # Save to temporary cache with source information
            await save_to_temp_cache(query, answer, sources, source="no_match")
            logging.info("Finished streaming formal message")
            return


        logging.info("Calling Gemini API for streaming response")
        # Use the streaming chain for all responses to ensure consistency
        chain = get_streaming_chain()
       
        # Get retriever separately to collect sources
        from utils.chroma_utils import get_chroma_db
        # Retrieve more documents initially and filter by similarity threshold
        retriever = get_chroma_db().as_retriever(search_kwargs={"k": settings.retrieval_k})
        # Get all results with scores
        all_docs_with_scores = get_chroma_db().similarity_search_with_score(query, k=settings.retrieval_k)
        # Filter by similarity threshold (higher score = less similar)
        filtered_docs = [doc for doc, score in all_docs_with_scores if score <= settings.retrieval_threshold]
        # Limit to maximum number of context documents
        docs = filtered_docs[:settings.max_context_docs] if filtered_docs else []
        sources = [doc.metadata.get("source", "") for doc in docs if doc.metadata.get("source")]


        # If not in cache and Gemini is enabled, proceed with the streaming chain
        if session_id not in active_sessions:
            active_sessions[session_id] = {"created_at": time.time(), "queries": []}
        active_sessions[session_id]["queries"].append(query)
        active_sessions[session_id]["last_active"] = time.time()


        full_response = []


        async for chunk in chain.astream(query):  # Now we can pass just the query
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_response.append(content)
            yield {"event": "message", "data": content}
            await asyncio.sleep(0.01)


        final_response = "".join(full_response)


        yield {"event": "sources", "data": ",".join(sources)}
        yield {"event": "suggestions", "data": ""}
        yield {"event": "metadata", "data": json.dumps({"feedback_enabled": True})}
        yield {"event": "done", "data": ""}


        # Save the complete response to the temporary cache with source information
        await save_to_temp_cache(query, final_response, sources, source="gemini")
        logging.info("Finished streaming Gemini response")


    except Exception as e:
        error_message = str(e).lower()
        logging.error(f"Error in streaming response: {error_message}")


        if "api key" in error_message:
            user_message = "⚠️ AI service configuration issue."


        elif "quota" in error_message or "429" in error_message:
            user_message = "⚠️ AI quota exceeded. Please try later."


        elif "timeout" in error_message:
            user_message = "⚠️ AI response timeout."


        else:
            user_message = "⚠️ AI service temporarily unavailable."


        yield {"event": "error", "data": user_message}


@router.post(
    "/stream",
    summary="Stream a chat response",
    description="Stream a chat response in real-time using Server-Sent Events (SSE)",
    response_description="An EventSourceResponse that streams the response in real-time",
)
async def stream_chat(request: StreamingChatRequest):
    """
    Stream a chat response in real-time using Server-Sent Events (SSE).
   
    - **query**: The user's question or message
    - **session_id**: Optional unique identifier for the chat session
    """
    if not request.session_id:
        request.session_id = str(uuid.uuid4())
   
    return EventSourceResponse(
        stream_response(request.query, request.session_id),
        media_type="text/event-stream"
    )


@router.post(
    "/feedback",
    summary="Submit feedback for a chat response",
    description="Submit thumbs up or thumbs down feedback for a chat response, storing negative feedback in MongoDB",
    response_description="Status of the feedback submission",
)
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit feedback for a chat response.
   
    - **session_id**: Unique identifier for the chat session
    - **query**: The user's query
    - **response**: The AI-generated response
    - **feedback**: 'positive' or 'negative'
    - **timestamp**: When the feedback was submitted
    - **additional_comments**: Optional user comments
    - **user_id**: Optional Firebase user ID
    - **user_email**: Optional user email
    - **user_name**: Optional user display name
    """
    try:
        feedback_data = {
            "session_id": feedback.session_id,
            "query": feedback.query,
            "response": feedback.response,
            "feedback": feedback.feedback,
            "timestamp": feedback.timestamp,
            "additional_comments": feedback.additional_comments,
            "user_id": feedback.user_id,
            "user_email": feedback.user_email,
            "user_name": feedback.user_name
        }
        success = await save_feedback(feedback_data)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save feedback")
       
        return {"message": "Feedback submitted successfully"}
    except Exception as e:
        logging.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/sessions/{session_id}",
    summary="Delete a chat session",
    description="Delete a chat session by its ID",
    response_description="A confirmation message",
)
async def delete_session(session_id: str):
    """
    Delete a chat session by its ID.
   
    - **session_id**: The ID of the session to delete
    """
    if session_id in active_sessions:
        del active_sessions[session_id]
        return {"message": f"Session {session_id} deleted successfully"}
    raise HTTPException(status_code=404, detail="Session not found")


@router.get(
    "/sessions",
    summary="List all active chat sessions",
    description="Get a list of all active chat sessions",
    response_description="A list of session IDs and their metadata",
)
async def list_sessions():
    """List all active chat sessions with metadata."""
    return {
        "sessions": [
            {"id": session_id, "created_at": data["created_at"],
             "last_active": data.get("last_active"), "query_count": len(data.get("queries", []))}
            for session_id, data in active_sessions.items()
        ]
    }


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for streaming chat responses specifically for React Native apps.
   
    - **session_id**: Unique identifier for the chat session
    """
    await websocket.accept()
   
    try:
        while True:
            # Receive message from React Native app
            data = await websocket.receive_text()
           
            # Try to parse as JSON first
            try:
                json_data = json.loads(data)
                query = json_data.get("query")
            except json.JSONDecodeError:
                # If not JSON, treat the entire message as the query
                query = data
           
            if not query:
                await websocket.send_json({"event": "error", "data": "No query provided"})
                continue
               
            # Process the query using the existing streaming logic
            async for event in stream_response(query, session_id):
                # Send each event over WebSocket
                await websocket.send_json(event)
               
    except Exception as e:
        logging.error(f"WebSocket error: {str(e)}")
        await websocket.close()



