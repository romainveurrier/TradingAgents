"""Derivatives analyst: the crypto perpetual's analogue of fundamentals.

A coin has no statements to read, but its perpetual future carries the market's
positioning: who pays whom to stay in (funding), whether the perp trades rich or
cheap to the oracle (premium), and how much the price is moving (realized
volatility). This analyst reads those from the venue and turns them into the
carry and crowding view the rest of the desk lacks. It files its report in the
fundamentals slot, so the researchers and risk debaters read it where they read
fundamentals for a stock.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.turn import take_turn
from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.tools import get_fear_greed, get_perp_funding, get_perp_market_structure

TOOLS = (
    get_perp_funding,
    get_perp_market_structure,
    get_fear_greed,
)

SYSTEM_MESSAGE = """You are a crypto derivatives analyst covering a perpetual future. Your report is the desk's view of positioning, carry and volatility, the crypto analogue of a fundamentals report.

Call get_perp_funding and get_perp_market_structure for the current date before writing; call get_fear_greed for the crowd's mood. Then cover, with numbers from the tool output:

1. **Funding and carry.** Where funding sits relative to the venue's baseline rate, its trend over 24 hours, the window and 30 days, and what holding a long or a short for the next 24 hours costs or earns in basis points of notional. Funding persistently above the baseline means longs are crowded and paying up; below the baseline or negative means shorts are crowded.
2. **Premium.** Whether the perp trades above or below the oracle and what that says about aggressive buyers or sellers.
3. **Volatility regime.** Realized volatility over 24 hours, 7 days and the window: is risk expanding or compressing? Give the one-sigma daily move it implies, which sets how large a move is noise.
4. **Trend and range context.** Returns over 1/3/7/30 days, distance from the 30-day high and low, and volume relative to its mean (participation behind the move).
5. **Sentiment index.** The Fear & Greed level and its change, read as a contrarian gauge at extremes and as confirmation in the middle.
6. **Implications.** What the evidence says for the next 24 hours: is positioning crowded on one side (squeeze risk), does carry favor one side, and is the volatility regime a reason to size down? Separate what the data shows from what it suggests.

Report only what the tools return; if a tool returns an unavailable or no-data notice, say so and do not estimate the missing values. Append a Markdown table summarizing each signal, its reading, and its implication for a long or a short."""


def create_derivatives_analyst(llm):

    def derivatives_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " Report what your tools support; another agent decides the trade."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=SYSTEM_MESSAGE + get_language_instruction())
        prompt = prompt.partial(tool_names=", ".join(tool.name for tool in TOOLS))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        result, report = take_turn(prompt, llm, TOOLS, state["messages"])
        return {"messages": [result], "fundamentals_report": report}

    return derivatives_analyst_node
