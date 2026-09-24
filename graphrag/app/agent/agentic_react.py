# Copyright (c) 2024-2026 TigerGraph, Inc.
#
# This program may be redistributed and/or modified under the terms of the GNU
# Affero General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Agentic react orchestrator - free tool-calling loop.

The configured chat model freely calls registry tools in a reason-act loop:
each iteration is one LLM round-trip that may emit zero or more tool calls,
whose results are fed back as ToolMessage observations on the next
iteration. The loop ends when the model answers without tool calls, or
when the per-graph iteration cap is hit.

This is the alternative to the planner-executor engine in
agentic_graph.run_agentic; both are reachable from AgenticAgent
based on graphrag_config.agent_style.
"""

import json
import logging
import time

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import PydanticOutputParser

from agent.agentic_executor import (
    _usage_since,
    cap_for_trace,
    retrieved_chunk_ids,
)
from common.llm_services.base_llm import get_collected_usage
from common.py_schemas import (
    GraphRAGAnswerOutput,
    GraphRAGResponse,
)
from src.router import classify_question
from tools import tool_registry as registry


logger = logging.getLogger(__name__)


MAX_ITERATIONS = 12
MAX_AGGREGATION_ANSWER_REJECTIONS = 2

_DETERMINISTIC_AGGREGATE_TOOL = (
    "graphrag__deterministic_aggregate"
)


# User-facing labels for tool calls.
_TOOL_LABELS = {
    "graphrag__get_schema": "Reading the graph schema",
    "graphrag__structural_retrieve": (
        "Searching the knowledge graph"
    ),
    "graphrag__hybrid_search": "Searching the documents",
    "graphrag__contextual_search": "Searching the documents",
    "graphrag__similarity_search": "Searching the documents",
    "graphrag__community_search": (
        "Searching community summaries"
    ),
    "graphrag__deterministic_aggregate": (
        "Computing a verified aggregation"
    ),
    "tg_run_query": "Running a graph query",
}


def _tool_label(name: str) -> str:
    """Return a user-friendly progress label for a tool."""

    return _TOOL_LABELS.get(
        name,
        "Gathering information",
    )


def _document_id_from_citation(citation_id: str) -> str:
    """Return the source document ID represented by a chunk citation."""

    value = str(citation_id).strip()
    return value.split("_chunk_", 1)[0].upper()


def _authoritative_aggregation_answer(context: dict) -> str:
    """Render an aggregation result without asking the LLM to recount it."""

    answer_value = context.get("answer_value")
    field_name = context.get("field_name", "value")
    comparison = context.get("comparison", "")
    threshold = context.get("threshold")
    qualifying_records = context.get("qualifying_records") or []
    operation = context.get("operation", "count")

    if operation in {"max", "min"}:
        label = "highest" if operation == "max" else "lowest"
        opening = (
            f"According to the provided corpus, the {label} result is "
            f"**{answer_value}**."
        )
    elif operation == "lookup":
        opening = (
            "According to the provided corpus, the verified result is "
            f"**{answer_value:g}**."
            if isinstance(answer_value, (int, float))
            else f"According to the provided corpus, the verified result is **{answer_value}**."
        )
    else:
        opening = ""

    condition = " ".join(
        str(part)
        for part in (field_name, comparison, threshold)
        if part not in (None, "")
    )

    if not opening and condition:
        opening = (
            f"According to the provided corpus, **{answer_value}** "
            f"record(s) satisfy `{condition}`."
        )
    elif not opening:
        opening = (
            "According to the provided corpus, the verified result is "
            f"**{answer_value}**."
        )

    if not qualifying_records:
        return opening

    lines = []
    for record in qualifying_records:
        entity = record.get("entity") or record.get("document_id")
        value = record.get("value")
        lines.append(f"- **{entity}**: {value}")

    return opening + "\n\nQualifying records:\n" + "\n".join(lines)


def run_react(
    ctx,
    llm,
    question,
    conversation=None,
) -> GraphRAGResponse:
    """Run the free tool-calling loop for one question."""

    emit = ctx.emit
    config = ctx.graphrag_cfg or {}

    max_iters = int(
        config.get(
            "agent_max_iterations",
            MAX_ITERATIONS,
        )
    )

    user = (
        f"## Question\n{question}\n\n"
        f"## Conversation\n"
        f"{json.dumps(conversation or [])[:2000]}"
    )

    # Base prompt configured by TigerGraph.
    system_prompt = llm.agentic_agent_prompt

    answer_parser = PydanticOutputParser(
        pydantic_object=GraphRAGAnswerOutput
    )

    try:
        answer_style = llm.get_user_portion(
            "chatbot_response.txt"
        )
    except Exception:
        answer_style = ""

    final_answer_block = [
        "\n\n## Final Answer",
        (
            "When the gathered context can answer the question, "
            "STOP calling tools and reply with a SINGLE JSON "
            "object, with no tool call, of this shape:"
        ),
        answer_parser.get_format_instructions(),
        (
            "Put the full natural-language answer in "
            "`generated_answer`, and in `citation` list the "
            "keys or IDs of the context parts actually used."
        ),
    ]

    if answer_style:
        final_answer_block.append(
            "Follow these guidelines when writing "
            "`generated_answer`:\n"
            + answer_style
        )

    system_prompt += "\n".join(final_answer_block)

    # --------------------------------------------------------------
    # Deterministic question routing
    # --------------------------------------------------------------

    route_decision = classify_question(question)
    is_aggregation = route_decision.question_type.value in {
        "aggregation",
        "superlative",
    }

    if is_aggregation:
        system_prompt += """

## Deterministic Aggregation Route

This question has been classified as an aggregation question.

Follow this required process:

1. Use hybrid or contextual search to collect all candidate source
   documents relevant to the requested category, event and time period.
2. Call graphrag__deterministic_aggregate with the complete original
   user question. Document IDs are optional trace evidence, not the
   calculation boundary.
3. Let that tool scan and strictly filter the complete local corpus.
4. Treat answer_value returned by that tool as authoritative.
5. Do not manually recount the records.
6. After deterministic aggregation succeeds, stop calling tools and
   immediately return the final answer with the qualifying citations.

Do not attempt to generate or run GSQL for this aggregation question.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user),
    ]

    tools = registry.lc_tools_spec(ctx)

    # For aggregation questions, prevent the LLM from selecting
    # structural/GSQL tools. Only retrieval and verified computation
    # tools are exposed.
    if is_aggregation:
        aggregation_tool_names = {
            "graphrag__get_schema",
            "graphrag__hybrid_search",
            "graphrag__contextual_search",
            "graphrag__similarity_search",
            "graphrag__deterministic_aggregate",
        }

        tools = [
            tool
            for tool in tools
            if tool.name in aggregation_tool_names
        ]

    # --------------------------------------------------------------
    # Agent execution state
    # --------------------------------------------------------------

    agent_steps = []
    final_answer = None

    # Retrieved citations are every chunk fetched by tools.
    retrieved_citations: list = []
    retrieved_seen: set = set()

    # Selected citations are those used in the final answer.
    selected_citations: list = []

    # Aggregation answers are not allowed to finish until this is populated
    # by a successful deterministic tool result.
    authoritative_aggregation: dict | None = None
    authoritative_tool_citations: list = []
    aggregation_answer_rejections = 0
    verification_failed = False

    # --------------------------------------------------------------
    # Reactive tool-calling loop
    # --------------------------------------------------------------

    for iteration in range(max_iters):
        emit("Thinking")

        usage_start = len(
            get_collected_usage() or []
        )

        iteration_started = time.time()

        try:
            response = llm.invoke_with_tools(
                messages,
                tools,
                caller_name=f"react_iter_{iteration}",
            )
        except Exception as exc:
            logger.warning(
                "react iter %s LLM failed: %s",
                iteration,
                exc,
            )
            break

        iteration_duration = round(
            time.time() - iteration_started,
            3,
        )

        messages.append(response)

        tool_calls = list(
            getattr(response, "tool_calls", []) or []
        )

        if isinstance(response.content, str):
            ai_text = response.content
        else:
            ai_text = "".join(
                content.get("text", "")
                for content in (response.content or [])
                if isinstance(content, dict)
            )

        # ----------------------------------------------------------
        # Final answer turn
        # ----------------------------------------------------------

        if not tool_calls:
            if is_aggregation and authoritative_aggregation is None:
                aggregation_answer_rejections += 1

                agent_steps.append(
                    {
                        "node": (
                            f"iter {iteration + 1}: "
                            "unverified answer rejected"
                        ),
                        "kind": "verification",
                        "duration_s": iteration_duration,
                        "input": {
                            "messages_so_far": len(messages) - 1,
                        },
                        "output": {
                            "accepted": False,
                            "reason": (
                                "Aggregation requires a successful "
                                "deterministic aggregation result."
                            ),
                        },
                        "usage": _usage_since(usage_start),
                    }
                )

                if (
                    aggregation_answer_rejections
                    >= MAX_AGGREGATION_ANSWER_REJECTIONS
                ):
                    final_answer = (
                        "I could not produce a verified aggregation "
                        "because the deterministic aggregation tool "
                        "did not complete successfully."
                    )
                    verification_failed = True
                    break

                messages.append(
                    HumanMessage(
                        content=(
                            "Your proposed final answer was rejected. "
                            "This is an aggregation question, so you must "
                            "call graphrag__deterministic_aggregate and "
                            "receive ok=true before answering. Use the "
                            "complete original question as the question "
                            "argument. Do not manually calculate.\n"
                            f"Original question: {question}\n"
                            f"Optional retrieved IDs: {json.dumps(retrieved_citations)}"
                        )
                    )
                )
                continue

            parsed = llm.parse_answer_output(ai_text)

            final_answer = (
                parsed.generated_answer or ""
            ).strip()

            if not final_answer:
                final_answer = "(no answer produced)"

            selected_citations = list(
                parsed.citation or []
            )

            agent_steps.append(
                {
                    "node": (
                        f"iter {iteration + 1}: answer"
                    ),
                    "kind": "answer",
                    "duration_s": iteration_duration,
                    "input": {
                        "messages_so_far": (
                            len(messages) - 1
                        ),
                    },
                    "output": {
                        "answer": final_answer[:4000],
                        "citations": selected_citations,
                    },
                    "usage": _usage_since(
                        usage_start
                    ),
                }
            )

            break

        # ----------------------------------------------------------
        # Tool execution turn
        # ----------------------------------------------------------

        per_call_traces = []

        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                name = tool_call.get("name")
                arguments = tool_call.get(
                    "args",
                    {},
                )
                tool_call_id = tool_call.get("id")
            else:
                name = getattr(
                    tool_call,
                    "name",
                    None,
                )
                arguments = getattr(
                    tool_call,
                    "args",
                    {},
                )
                tool_call_id = getattr(
                    tool_call,
                    "id",
                    None,
                )

            emit(_tool_label(name))

            # Never trust an LLM paraphrase to preserve strict benchmark
            # constraints. The deterministic tool always receives the exact
            # original question, while retrieved IDs remain optional evidence.
            if name == _DETERMINISTIC_AGGREGATE_TOOL:
                arguments = dict(arguments or {})
                arguments["question"] = question
                arguments.setdefault(
                    "document_ids",
                    list(retrieved_citations),
                )

            tool_started = time.time()

            output = registry.run(
                name or "",
                arguments or {},
                ctx,
            )

            output_context = output.get("context")

            if (
                is_aggregation
                and name == _DETERMINISTIC_AGGREGATE_TOOL
                and bool(output.get("ok"))
                and isinstance(output_context, dict)
                and output_context.get("authoritative") is True
            ):
                authoritative_aggregation = output_context
                authoritative_tool_citations = list(
                    output.get("citations") or []
                )

            tool_duration = round(
                time.time() - tool_started,
                3,
            )

            # Feed the complete result back to the model.
            observation = {
                "summary": output.get(
                    "summary",
                    "",
                )
            }

            if output.get("context") is not None:
                observation["result"] = output.get(
                    "context"
                )

            messages.append(
                ToolMessage(
                    content=json.dumps(
                        observation,
                        default=str,
                    ),
                    tool_call_id=tool_call_id or "",
                )
            )

            per_call_traces.append(
                {
                    "tool": name,
                    "args": cap_for_trace(
                        arguments
                    ),
                    "ok": bool(
                        output.get("ok")
                    ),
                    "summary": output.get(
                        "summary",
                        "",
                    ),
                    "duration_s": tool_duration,
                }
            )

            # Save all retrieved chunk identifiers.
            for citation_id in retrieved_chunk_ids(
                output.get("context")
            ):
                if citation_id in retrieved_seen:
                    continue

                retrieved_seen.add(citation_id)
                retrieved_citations.append(
                    citation_id
                )

        agent_steps.append(
            {
                "node": (
                    f"iter {iteration + 1}: tool calls"
                ),
                "kind": "react",
                "duration_s": iteration_duration,
                "input": {
                    "reasoning_preview": (
                        ai_text[:600]
                        if ai_text
                        else ""
                    ),
                },
                "output": {
                    "tool_calls": per_call_traces,
                },
                "usage": _usage_since(
                    usage_start
                ),
            }
        )

        # A successful deterministic aggregation is terminal. Build the
        # response directly from its authoritative context so a later LLM
        # turn cannot change (or recount) answer_value.
        if authoritative_aggregation is not None:
            final_answer = _authoritative_aggregation_answer(
                authoritative_aggregation
            )

            qualifying_document_ids = {
                str(record.get("document_id", "")).upper()
                for record in (
                    authoritative_aggregation.get(
                        "qualifying_records"
                    )
                    or []
                )
            }

            selected_citations = [
                citation_id
                for citation_id in retrieved_citations
                if _document_id_from_citation(citation_id)
                in qualifying_document_ids
            ]

            if not selected_citations:
                selected_citations = authoritative_tool_citations

            agent_steps.append(
                {
                    "node": "verified deterministic answer",
                    "kind": "answer",
                    "duration_s": 0.0,
                    "input": {
                        "answer_value": (
                            authoritative_aggregation.get(
                                "answer_value"
                            )
                        ),
                        "authoritative": True,
                    },
                    "output": {
                        "answer": final_answer[:4000],
                        "citations": selected_citations,
                    },
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "cost": 0.0,
                        "calls": [],
                    },
                }
            )
            break

    # --------------------------------------------------------------
    # Final fallback and response
    # --------------------------------------------------------------

    hit_iteration_cap = final_answer is None

    if final_answer is None:
        final_answer = (
            "I gathered some information but couldn't finalize "
            "an answer within the iteration budget "
            f"({max_iters})."
        )

    return GraphRAGResponse(
        natural_language_response=final_answer,
        answered_question=bool(
            final_answer
            and not hit_iteration_cap
            and not verification_failed
        ),
        response_type="agentic",
        query_sources={
            "engine": "react",
            "agent_steps": agent_steps,
            "iterations": len(
                [
                    step
                    for step in agent_steps
                    if step["kind"]
                    in ("react", "answer")
                ]
            ),
            "max_iterations": max_iters,
            "hit_iteration_cap": (
                hit_iteration_cap
            ),
            "citations": selected_citations,
            "retrieved_citations": (
                retrieved_citations
            ),
            "verification": {
                "required": is_aggregation,
                "authoritative": (
                    authoritative_aggregation is not None
                ),
                "failed": verification_failed,
                "answer_value": (
                    authoritative_aggregation.get("answer_value")
                    if authoritative_aggregation is not None
                    else None
                ),
            },
            "route": {
                "question_type": (
                    route_decision.question_type.value
                ),
                "confidence": (
                    route_decision.confidence
                ),
                "matched_rules": (
                    route_decision.matched_rules
                ),
                "required_tools": (
                    route_decision.required_tools
                ),
            },
        },
    )
