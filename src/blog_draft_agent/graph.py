from langgraph.graph import END, START, StateGraph

from blog_draft_agent.nodes import check_tone, keyword_extract, title, write_posting, write_posting_campaign
from blog_draft_agent.routing import route_after_check, route_after_title
from blog_draft_agent.state import State


g = StateGraph(State)
g.add_node("keyword_extract", keyword_extract)
g.add_node("title", title)
g.add_node("write_posting", write_posting)
g.add_node("write_posting_campaign", write_posting_campaign)
g.add_node("check_tone", check_tone)

g.add_edge(START, "keyword_extract")
g.add_edge("keyword_extract", "title")
g.add_conditional_edges("title", route_after_title, ["write_posting", "write_posting_campaign"])
g.add_edge("write_posting", "check_tone")
g.add_edge("write_posting_campaign", "check_tone")
g.add_conditional_edges("check_tone", route_after_check, ["write_posting", "write_posting_campaign", END])

graph = g.compile()
