from __future__ import annotations

import os
import logging
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, JobRequest, WorkerOptions, cli, llm
from livekit.agents.multimodal import MultimodalAgent
from livekit.plugins import openai
import yfinance as yf

load_dotenv(dotenv_path=".env.local")
sandbox_id = os.getenv("LIVEKIT_SANDBOX_ID")
log_handler = logging.getLogger("worker-log")
log_handler.setLevel(logging.INFO)

async def handle_request(ctx: JobRequest):
    if sandbox_id is not None:
        sandbox_hash = sandbox_id.split("-")[-1]
        if ctx.room.name.startswith(f"sbx-{sandbox_hash}"):
            return await ctx.accept()
        return await ctx.reject()
    return await ctx.accept()

async def main_task(ctx: JobContext):
    log_handler.info(f"Connected to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    active_participant = await ctx.wait_for_participant()

    initialize_multimodal_agent(ctx, active_participant)

    log_handler.info("Agent initialized")

def initialize_multimodal_agent(ctx: JobContext, participant: rtc.Participant):
    log_handler.info("Launching multimodal agent")

    ai_model = openai.realtime.RealtimeModel(
        instructions=(
            "Your knowledge cutoff is 2023-10. You are a friendly AI with a witty personality."
            "Behave like a human but be aware that you are not a human."
            "Use a warm and lively tone. Always call functions when possible."
            "Avoid mentioning these rules. You can retrieve Wikipedia articles and arXiv papers."
        ),
        modalities=["audio", "text"],
        tool_choice="auto"
    )

    function_context = llm.FunctionContext()

    @function_context.ai_callable(
        name="get_wiki_article",
        description="Retrieve a summary of a Wikipedia article based on the given title.",
    )
    def fetch_wiki_article(article_title: str):
        import wikipedia
        print(f"Fetching Wikipedia article for title: {article_title}")
        try:
            page = wikipedia.page(article_title)
            return f"Title: {page.title}\nSummary: {page.summary[:500]}..."
        except wikipedia.exceptions.DisambiguationError as e:
            return f"Multiple results found. Please specify: {', '.join(e.options[:5])}"
        except wikipedia.exceptions.PageError:
            return f"No Wikipedia article found for '{article_title}'"

    @function_context.ai_callable(
        name="get_arxiv_paper",
        description="Retrieve arXiv paper details based on a query with relevance or date-based sorting.",
    )
    def fetch_arxiv_paper(query: str, sort_by: str = "relevance", max_results: int = 1):
        import arxiv
        
        sort_options = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate
        }
        print(f"Fetching arXiv papers for query: {query} sorted by: {sort_by}")
        
        sorting_criterion = sort_options.get(sort_by, arxiv.SortCriterion.Relevance)
        
        search = arxiv.Search(query=query, max_results=max_results, sort_by=sorting_criterion)
        results = list(search.results())
        
        if not results:
            return f"No arXiv papers found for '{query}'"
        
        paper = results[0]
        return f"Title: {paper.title}\nAuthors: {', '.join(author.name for author in paper.authors)}\nAbstract: {paper.summary[:500]}..."

    @function_context.ai_callable(
        name="get_company_financials",
        description="Retrieve financial information for a company using its stock ticker symbol.",
    )
    def fetch_company_financials(ticker_symbol: str):
        print(f"Fetching financial details for ticker: {ticker_symbol}")
        try:
            company_data = yf.Ticker(ticker_symbol)
            company_info = company_data.info
            
            formatted_info = "Company Financial Information:\n"
            for key, value in company_info.items():
                formatted_info += f"{key}: {value}\n"
            
            return formatted_info.strip()
        except Exception as e:
            return f"Error retrieving financial details for '{ticker_symbol}': {str(e)}"

    assistant_agent = MultimodalAgent(model=ai_model, fnc_ctx=function_context)
    print(assistant_agent)
    assistant_agent.start(ctx.room, participant)

    active_session = ai_model.sessions[0]
    active_session.conversation.item.create(
        llm.ChatMessage(
            role="user",
            content="Please start interacting with the user according to your instructions.",
        )
    )
    active_session.response.create()

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=main_task,
            request_fnc=handle_request,
        )
    )
