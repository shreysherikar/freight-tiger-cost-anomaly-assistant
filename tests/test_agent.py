import sys
import os
sys.path.insert(0, os.path.abspath("."))
import pytest
from src.agent_tools import AssistantTools
from cli_assistant import ToolCallingAgent

@pytest.fixture
def agent():
    tools = AssistantTools(output_csv_path="output.csv", notes_csv_path="context_notes.csv")
    return ToolCallingAgent(tools, use_llm=False)

def test_q1_chennai_march(agent):
    ans, calls = agent.run("Why did Chennai-Bangalore get pricier in March 2025?")
    assert any("get_flagged_weeks" in c for c in calls)
    assert "N001" in ans
    assert "2025-03-03" in ans

def test_q2_ahmedabad_january(agent):
    ans, calls = agent.run("What happened on Ahmedabad-Mumbai in January 2025?")
    assert any("get_flagged_weeks" in c for c in calls)
    assert "N002" in ans
    assert "2025-01-20" in ans

def test_q3_delhi_jaipur_specific_week(agent):
    ans, calls = agent.run("What were the weekly metrics for Delhi-Jaipur on 2024-11-11?")
    assert any("get_weekly_metrics" in c and "Delhi-Jaipur" in c for c in calls)
    assert "4.17" in ans
    assert "+35.5%" in ans

def test_q4_mumbai_pune_specific_week(agent):
    ans, calls = agent.run("What were the weekly metrics for Mumbai-Pune on 2025-09-15?")
    assert any("get_weekly_metrics" in c and "Mumbai-Pune" in c for c in calls)
    assert "3.98" in ans
    assert "+9.2%" in ans

def test_q5_peer_comparison(agent):
    ans, calls = agent.run("How did Mumbai-Pune compare to its peers on 2025-12-01?")
    assert any("compare_to_peers" in c for c in calls)
    assert "+37.4%" in ans

def test_q6_get_note_content(agent):
    ans, calls = agent.run("Can you show me the contents of note N001?")
    assert any("get_note" in c and "N001" in c for c in calls)
    assert "flooding" in ans.lower()

def test_q7_routes_affected_by_note(agent):
    ans, calls = agent.run("Which routes were affected by note N003?")
    assert any("get_note" in c for c in calls)
    assert any("get_flagged_weeks" in c for c in calls)
    assert "N003" in ans
    assert "Delhi-Jaipur" in ans or "Mumbai-Delhi" in ans

def test_q8_refusal_untracked_route(agent):
    ans, calls = agent.run("Why did shipments in Hyderabad get delayed in June 2024?")
    assert "Refusal" in ans
    assert "not tracked" in ans

def test_q9_refusal_out_of_bounds_year(agent):
    ans, calls = agent.run("What was the freight cost for Delhi-Jaipur in 2030?")
    assert "Refusal" in ans
    assert "2030" in ans

def test_q10_refusal_out_of_domain(agent):
    ans, calls = agent.run("How do I bake chocolate chip cookies?")
    assert "Refusal" in ans
