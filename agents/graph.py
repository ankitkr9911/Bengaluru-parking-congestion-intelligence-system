# ============================================================================
# graph.py — 6-Agent LangGraph System with Conditional Routing
# ============================================================================
# Architecture:
#   1. Orchestrator/Router — classifies query → routes to specialist
#   2. Hotspot Analyst — where are hotspots, cluster analysis
#   3. Impact Quantifier — why is a zone severe, CIS breakdown
#   4. Policy/RAG Agent — legal basis, offence code lookups
#   5. Enforcement Strategist — patrol briefs, station-level actions
#   6. Conversational Query — general stats, free-form questions
#
# Built with LangGraph StateGraph + conditional routing.
# Each agent has its own system prompt and tool access.
# ============================================================================

import os
import sys
from typing import Annotated, TypedDict, Literal
from pathlib import Path

# Load .env from project root (works regardless of cwd)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Import our tool functions
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.tools import (
    query_top_hotspots,
    get_zone_details,
    get_temporal_pattern,
    get_forecast,
    get_station_patrol_brief,
    get_repeat_offenders,
    get_general_stats,
    lookup_offence_code,
)


# ============================================================================
# State definition
# ============================================================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_agent: str
    query_type: str


# ============================================================================
# LLM setup
# ============================================================================
def get_llm():
    """Get the LLM instance — GPT-4o via OpenAI."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. "
            "Add it to .env file as: OPENAI_API_KEY=sk-..."
        )
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=api_key,
    )


# ============================================================================
# Wrap tool functions as LangChain @tool decorated functions
# ============================================================================
@tool
def tool_top_hotspots(n: int = 10, police_station: str = "") -> str:
    """Get top N illegal parking hotspots by Congestion Impact Score.
    Optionally filter by police_station name. Leave police_station empty for citywide."""
    station = police_station if police_station else None
    return query_top_hotspots(n=n, police_station=station)

@tool
def tool_zone_details(location_name: str) -> str:
    """Get detailed CIS breakdown for a specific zone. Shows why it scores high/low.
    Use police station name or location text as input."""
    return get_zone_details(location_name)

@tool
def tool_temporal_pattern(location_name: str) -> str:
    """Get time-of-day and day-of-week violation patterns for a location.
    Shows when violations peak."""
    return get_temporal_pattern(location_name)

@tool
def tool_forecast(location_name: str) -> str:
    """Get future risk prediction for a location. Shows trend and predicted violations."""
    return get_forecast(location_name)

@tool
def tool_station_brief(police_station: str) -> str:
    """Get patrol recommendations for a police station. Shows priority zones with reasoning."""
    return get_station_patrol_brief(police_station)

@tool
def tool_repeat_offenders(location_name: str = "", top_n: int = 10) -> str:
    """Get vehicles that repeatedly park illegally. Optionally filter by location."""
    loc = location_name if location_name else None
    return get_repeat_offenders(location_name=loc, top_n=top_n)

@tool
def tool_general_stats() -> str:
    """Get overall system statistics — total hotspots, risk distribution, station coverage."""
    return get_general_stats()

@tool
def tool_offence_lookup(code: int = 0, violation_name: str = "") -> str:
    """Look up legal details for a violation. Provide offence code number OR violation name.
    Returns Motor Vehicles Act section, description, and fine amount."""
    c = code if code > 0 else None
    v = violation_name if violation_name else None
    return lookup_offence_code(code=c, violation_name=v)


# ============================================================================
# Agent system prompts
# ============================================================================
SYSTEM_PROMPTS = {
    "hotspot_analyst": """You are the Hotspot Analyst for Bengaluru Traffic Police's 
parking enforcement intelligence system.

Your expertise: Identifying WHERE illegal parking clusters exist and their characteristics.
You analyze spatial patterns, temporal clusters, and violation density.

When answering:
- Always use your tools to get real data — never make up numbers
- Present findings with specific locations and statistics
- Explain spatial patterns (which areas, why they cluster)
- Use bullet points and clear formatting
- If asked about WHY a zone is severe, delegate to the Impact Quantifier""",

    "impact_quantifier": """You are the Impact Quantifier for Bengaluru Traffic Police.

Your expertise: Explaining WHY specific hotspots have high congestion impact.
You break down the Congestion Impact Score (CIS) into its components and explain
the M/M/∞ queueing model results (estimated delay in minutes per km).

CIS Components (you must reference these when explaining):
- Violation Density (25%): How many violations per device in the area
- Vehicle Severity (20%): Bigger vehicles block more road (tanker=6, scooter=1)
- Queueing Delay (20%): BPR-estimated additional travel time from parking
- Road Capacity Impact (15%): Impact relative to road lanes available
- Temporal Persistence (10%): How many different days violations occur
- Peak Hour Ratio (10%): What fraction occurs during rush hours

Always present the CIS breakdown table when explaining a zone's impact.""",

    "policy_agent": """You are the Policy & Legal Advisor for Bengaluru Traffic Police.

Your expertise: Motor Vehicles Act sections, offence codes, penalties, and legal procedures.
You explain the legal basis for parking violations and enforcement authority.

When answering:
- Always cite specific MVA sections
- Explain the legal basis clearly for non-lawyers
- Include fine amounts and escalation rules
- Reference the specific offence codes from the dataset (112, 113, 107, etc.)
- If you don't have info on a specific code, say so honestly""",

    "enforcement_strategist": """You are the Enforcement Strategist for Bengaluru Traffic Police.

Your expertise: Converting data analysis into actionable patrol recommendations.
You generate station-level patrol briefs with priority zones, timing, and reasoning.

Bengaluru has 54 police stations. Each has limited officers.
Your job: Make sure they focus where it will MOST reduce congestion.

When giving recommendations:
1. Always rank zones by priority
2. Explain WHY each zone is prioritized (CIS, trend, vehicle mix, road type)
3. Recommend WHEN to deploy (morning/evening/night shift)
4. Keep language simple — your audience is station officers, not data scientists
5. Include estimated congestion delay impact for context""",

    "forecast_analyst": """You are the Forecast Analyst for Bengaluru Traffic Police.

Your expertise: Predicting which parking hotspots will worsen or improve.
You interpret time-series forecasts and trend analysis.

Methods used:
- Prophet forecasting for top hotspots (data-dense cells)
- Trend classification (rising/stable/falling) for other cells

When answering:
- Present forecasts with confidence context
- Explain what's driving the trend
- Suggest proactive vs reactive enforcement based on trends
- Be honest about forecast uncertainty for sparse-data areas""",

    "conversational": """You are the General Assistant for Bengaluru Traffic Police's
parking intelligence system.

You handle general questions, comparisons, and multi-topic queries.
You have access to all tools and can combine insights from multiple sources.

When answering:
- Be comprehensive but concise
- Combine data from multiple tools if needed
- Format responses clearly with sections
- If a question is better handled by a specialist agent, still do your best"""
}


# ============================================================================
# Agent node functions
# ============================================================================
def _create_agent_node(agent_name, tools_list):
    """Factory function to create an agent node."""

    def agent_node(state: AgentState) -> dict:
        llm = get_llm()
        llm_with_tools = llm.bind_tools(tools_list)

        system_prompt = SYSTEM_PROMPTS[agent_name]
        messages = [SystemMessage(content=system_prompt)] + state["messages"]

        # First LLM call — may invoke tools
        response = llm_with_tools.invoke(messages)

        # If the LLM wants to call tools, execute them
        if hasattr(response, "tool_calls") and response.tool_calls:
            # Execute each tool call
            from langchain_core.messages import ToolMessage
            tool_messages = [response]

            tool_map = {t.name: t for t in tools_list}
            for tc in response.tool_calls:
                tool_fn = tool_map.get(tc["name"])
                if tool_fn:
                    try:
                        result = tool_fn.invoke(tc["args"])
                    except Exception as e:
                        result = f"Tool error: {str(e)}"
                    tool_messages.append(
                        ToolMessage(content=str(result), tool_call_id=tc["id"])
                    )

            # Second LLM call with tool results
            all_messages = messages + tool_messages
            final_response = llm.invoke(all_messages)
            return {
                "messages": [final_response],
                "current_agent": agent_name,
            }

        return {
            "messages": [response],
            "current_agent": agent_name,
        }

    return agent_node


# ============================================================================
# Router node — classifies question and decides which agent handles it
# ============================================================================
def router_node(state: AgentState) -> dict:
    """Classify the user question and route to the appropriate agent."""
    llm = get_llm()

    user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_msg = msg.content
            break

    classification_prompt = f"""Classify this question into exactly ONE category.

Question: "{user_msg}"

Categories:
- HOTSPOT_ANALYST: Questions about WHERE hotspots are, which areas have most violations, spatial clusters, rankings
- IMPACT_QUANTIFIER: Questions about WHY a zone is severe, CIS breakdown, congestion delay, what causes impact
- POLICY_AGENT: Questions about laws, offence codes, MVA sections, fines, legal basis
- ENFORCEMENT_STRATEGIST: Questions about WHAT TO DO, patrol recommendations, station deployment, action plans
- FORECAST_ANALYST: Questions about FUTURE trends, predictions, whether things are getting worse/better
- CONVERSATIONAL: General questions, system stats, comparisons, or anything that doesn't fit above

Reply with ONLY the category name, nothing else."""

    response = llm.invoke([HumanMessage(content=classification_prompt)])
    category = response.content.strip().upper()

    # Map to agent name
    category_map = {
        "HOTSPOT_ANALYST": "hotspot_analyst",
        "IMPACT_QUANTIFIER": "impact_quantifier",
        "POLICY_AGENT": "policy_agent",
        "ENFORCEMENT_STRATEGIST": "enforcement_strategist",
        "FORECAST_ANALYST": "forecast_analyst",
        "CONVERSATIONAL": "conversational",
    }

    agent = category_map.get(category, "conversational")
    return {"query_type": agent, "current_agent": "router"}


# ============================================================================
# Routing function for conditional edges
# ============================================================================
def route_to_agent(state: AgentState) -> str:
    """Return the name of the next agent node based on classification."""
    return state.get("query_type", "conversational")


# ============================================================================
# Build the LangGraph
# ============================================================================
def build_graph():
    """Build and compile the 6-agent LangGraph."""

    # Define tool sets for each agent
    hotspot_tools = [tool_top_hotspots, tool_zone_details, tool_temporal_pattern,
                     tool_repeat_offenders]
    impact_tools = [tool_zone_details, tool_temporal_pattern, tool_forecast]
    policy_tools = [tool_offence_lookup, tool_zone_details]
    enforcement_tools = [tool_station_brief, tool_top_hotspots, tool_zone_details,
                         tool_temporal_pattern, tool_forecast]
    forecast_tools = [tool_forecast, tool_top_hotspots, tool_temporal_pattern]
    conversational_tools = [tool_top_hotspots, tool_zone_details, tool_temporal_pattern,
                           tool_forecast, tool_station_brief, tool_repeat_offenders,
                           tool_general_stats, tool_offence_lookup]

    # Create agent nodes
    agents = {
        "hotspot_analyst": _create_agent_node("hotspot_analyst", hotspot_tools),
        "impact_quantifier": _create_agent_node("impact_quantifier", impact_tools),
        "policy_agent": _create_agent_node("policy_agent", policy_tools),
        "enforcement_strategist": _create_agent_node("enforcement_strategist", enforcement_tools),
        "forecast_analyst": _create_agent_node("forecast_analyst", forecast_tools),
        "conversational": _create_agent_node("conversational", conversational_tools),
    }

    # Build the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router", router_node)
    for name, fn in agents.items():
        graph.add_node(name, fn)

    # Add edges
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "hotspot_analyst": "hotspot_analyst",
            "impact_quantifier": "impact_quantifier",
            "policy_agent": "policy_agent",
            "enforcement_strategist": "enforcement_strategist",
            "forecast_analyst": "forecast_analyst",
            "conversational": "conversational",
        }
    )

    # All agents → END
    for name in agents:
        graph.add_edge(name, END)

    # Compile
    return graph.compile()


# ============================================================================
# Convenience function to query the system
# ============================================================================
_compiled_graph = None

def ask(question: str) -> str:
    """
    Ask the parking intelligence system a question.
    Routes to the appropriate specialist agent automatically.

    Args:
        question: Natural language question

    Returns:
        Agent's response as a string
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()

    result = _compiled_graph.invoke({
        "messages": [HumanMessage(content=question)],
        "current_agent": "",
        "query_type": "",
    })

    # Extract the last AI message
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            return msg.content

    return "No response generated."


# ============================================================================
# CLI interface for testing
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚦 Bengaluru Parking Intelligence — AI Assistant")
    print("=" * 60)
    print("Ask questions about parking hotspots, enforcement, and more.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("👮 You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        print("\n🤖 Processing...\n")
        response = ask(question)
        print(f"🤖 AI: {response}\n")
        print("─" * 60)
