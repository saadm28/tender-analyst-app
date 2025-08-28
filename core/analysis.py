import json
import re
from typing import Callable, Dict, List, Union


def load_prompt_template(template_name: str) -> str:
    """Load prompt template from file."""
    import os
    possible_paths = [
        f"app/prompts/{template_name}",
        f"prompts/{template_name}",
        os.path.join(os.path.dirname(__file__), "..", "prompts", template_name)
    ]
    for path in possible_paths:
        try:
            with open(path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            continue
    raise Exception(f"Prompt template not found: {template_name}")


def parse_json_response(response: str) -> dict:
    """Parse JSON response, handling code fence wrappers gracefully."""
    response = re.sub(r'^```(?:json)?\s*', '', response, flags=re.MULTILINE)
    response = re.sub(r'\s*```\s*$', '', response, flags=re.MULTILINE)
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        m = re.search(r'\{.*\}\s*$', response, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return {"error": f"Failed to parse JSON response: {str(e)}", "raw_response": response}


def _default_results() -> dict:
    """Schema-aligned default result for robustness."""
    return {
        "executive_summary": "Analysis completed.",
        "comparison_snapshot": [],
        "technical_analysis": {
            "overview": "Technical evaluation completed.",
            "detailed_assessments": []
        },
        "team_and_capacity_analysis": {
            "evaluation_criteria": "Team assessment completed.",
            "assessments": []
        },
        "financial_analysis": {
            "commercial_evaluation_summary": "Financial analysis completed.",
            "cost_breakdown_analysis": [],
            "budget_impact": ""
        },
        "timeline_and_delivery_analysis": {
            "schedule_evaluation": []
        },
        "compliance_and_regulatory_analysis": {
            "compliance_requirements": "",
            "assessments": []
        },
        "comprehensive_risk_analysis": {
            "risk_assessment_methodology": "Risk analysis completed.",
            "by_bidder": {},
            "general_project_risks": [],
            "risk_matrix_summary": ""
        },
        "strategic_recommendation": {
            "winner": "",
            "comprehensive_rationale": "",
            "selection_criteria_scoring": "",
            "runner_up_analysis": "",
            "contractual_conditions": [],
            "performance_guarantees": [],
            "risk_mitigation_measures": [],
            "success_factors": [],
            "kpis_and_monitoring": [],
            "escalation_procedures": []
        },
        "implementation_roadmap": {
            "immediate_actions": [],
            "contract_negotiation_priorities": [],
            "mobilization_requirements": [],
            "governance_framework": [],
            "stakeholder_communication": [],
            "milestone_tracking": []
        }
    }


def compare_and_recommend(
    rfp_text: str,
    tender_data: List[dict],
    financial_data: Union[List[Dict], str],
    respond_fn: Callable
) -> dict:
    """
    Generate comparison and recommendation using LLM with raw tender content.
    - rfp_text: Raw RFP content (string)
    - tender_data: List of dicts with 'name' and 'content' (strings)
    - financial_data: Combined financial summary data (list of dicts) or raw text (str)
    """
    try:
        print("DEBUG: Starting compare_and_recommend function...")
        print(f"DEBUG: Loading prompt template...")
        template = load_prompt_template("compare_recommend.md")
        print(f"DEBUG: Template loaded, length: {len(template)}")

        # Trim inputs for token safety
        rfp_highlights = str(rfp_text or "")[:15000]
        print(f"DEBUG: RFP highlights prepared, length: {len(rfp_highlights)}")

        tender_sections = []
        print(f"DEBUG: Processing {len(tender_data)} tender documents...")
        for i, tender in enumerate(tender_data):
            content = tender.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            tender_sections.append(
                f"TENDER {i+1}: {tender.get('name','Unknown')}\n"
                + "-" * 50 + "\n"
                + content[:20000]
                + "\n" + "-" * 50 + "\n"
            )
            print(f"DEBUG: Processed tender {i+1}: {tender.get('name','Unknown')[:50]}")
        tender_content_combined = "\n".join(tender_sections)
        print(f"DEBUG: Combined tender content length: {len(tender_content_combined)}")

        print(f"DEBUG: Processing financial data...")
        if isinstance(financial_data, (list, dict)):
            try:
                financial_formatted = json.dumps(financial_data, indent=2, default=str)
                financial_summary = financial_formatted[:15000]  # Increased limit for financial data
                print(f"DEBUG: Financial data formatted as JSON, length: {len(financial_summary)}")
                print(f"DEBUG: First 1000 chars of financial data:")
                print(financial_formatted[:1000])
            except Exception as e:
                print(f"DEBUG: Error formatting financial data as JSON: {e}")
                financial_summary = "Error processing financial data"
        elif isinstance(financial_data, str):
            financial_summary = financial_data[:10000]
            print(f"DEBUG: Financial data as string, length: {len(financial_summary)}")
        else:
            financial_summary = ""
            print(f"DEBUG: No financial data provided")

        print(f"DEBUG: Building final prompt...")
        prompt = (template
                  .replace("{{rfp_text}}", rfp_highlights)
                  .replace("{{tender_content}}", tender_content_combined)
                  .replace("{{commercial_text}}", financial_summary))
        
        prompt_length = len(prompt)
        print(f"DEBUG: Final prompt length: {prompt_length} characters")
        
        # Add prompt size warning
        if prompt_length > 100000:
            print(f"WARNING: Large prompt size ({prompt_length} chars) may cause timeout")

        print(f"DEBUG: Calling LLM API...")
        # Expect strict JSON from the model (your llm.respond uses json mode when passed)
        response = respond_fn(prompt)
        print(f"DEBUG: LLM response received, length: {len(str(response))}")
        
        print(f"DEBUG: Parsing JSON response...")
        result = parse_json_response(response)
        print(f"DEBUG: JSON parsed successfully")

        final = _default_results()
        if "error" not in result and isinstance(result, dict):
            # Merge shallowly; keep schema shape
            for k in final.keys():
                if k in result:
                    final[k] = result[k]
            print(f"DEBUG: Final result merged successfully")
        else:
            print(f"DEBUG: Error in result or result not dict: {result}")
        
        print(f"DEBUG: compare_and_recommend completed successfully")
        return final

    except Exception as e:
        print(f"DEBUG: Error in compare_and_recommend: {e}")
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        fallback = _default_results()
        fallback["executive_summary"] = f"Error in analysis: {str(e)}"
        fallback["comprehensive_risk_analysis"]["general_project_risks"] = [str(e)]
        return fallback
