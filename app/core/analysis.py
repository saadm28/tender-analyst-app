# app/core/analysis.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.llm import respond


def compare_and_recommend(rfp_text: str, tender_data: list, financial_data: str, llm_function):
    """
    Compare tender responses and generate comprehensive recommendation with external risk analysis
    """
    try:
        print("DEBUG: Starting comprehensive tender analysis with external risk assessment...")
        
        # Import news functions here to avoid circular imports
        from streamlit_app import fetch_company_news, analyze_company_news_risks
        
        # Extract company names from tender data
        company_names = []
        for tender in tender_data:
            company_name = tender.get('company', '')
            if not company_name:
                # Try to extract from filename
                filename = tender.get('name', '')
                if ' - ' in filename:
                    company_name = filename.split(' - ')[0].strip()
                else:
                    company_name = filename.rsplit('.', 1)[0].strip()
            
            if company_name and company_name not in company_names:
                company_names.append(company_name)
        
        print(f"DEBUG: Identified companies for news analysis: {company_names}")
        
        # Fetch and analyze news for each company
        external_risk_data = ""
        news_analysis_results = {}
        
        for company_name in company_names:
            try:
                print(f"DEBUG: Fetching news for {company_name}...")
                articles = fetch_company_news(company_name, page_size=5)
                
                if articles:
                    print(f"DEBUG: Analyzing {len(articles)} articles for {company_name}...")
                    risk_analysis = analyze_company_news_risks(company_name, articles, llm_function)
                    news_analysis_results[company_name] = {
                        'articles_count': len(articles),
                        'risk_analysis': risk_analysis
                    }
                    external_risk_data += f"\n\n**{company_name} External Risk Analysis:**\n{risk_analysis}\n"
                else:
                    print(f"DEBUG: No adverse media found for {company_name}")
                    news_analysis_results[company_name] = {
                        'articles_count': 0,
                        'risk_analysis': f"No recent adverse media or risk-related news found for {company_name}."
                    }
                    external_risk_data += f"\n\n**{company_name} External Risk Analysis:**\nNo recent adverse media or risk-related news found.\n"
                    
            except Exception as e:
                print(f"DEBUG: Error in news analysis for {company_name}: {e}")
                news_analysis_results[company_name] = {
                    'articles_count': 0,
                    'risk_analysis': f"Unable to complete external risk analysis for {company_name}: {str(e)}"
                }
        
        print(f"DEBUG: External risk analysis completed for {len(news_analysis_results)} companies")
        
        # Load prompt template
        prompt_file = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'compare_recommend.md')
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        
        # Format tender content for the prompt
        tender_content = ""
        for tender in tender_data:
            company_name = tender.get('company', tender.get('name', 'Unknown Company'))
            tender_content += f"\n\n**{company_name}:**\n{tender['content']}\n"
        
        # Prepare the complete prompt with external risk data
        prompt = prompt_template.replace("{{rfp_text}}", rfp_text)
        prompt = prompt.replace("{{tender_content}}", tender_content)
        prompt = prompt.replace("{{commercial_text}}", str(financial_data))
        
        # Add external risk analysis data to the prompt
        prompt += f"\n\nExternal Risk Analysis Data (based on recent news and media):\n{external_risk_data}\n"
        
        print(f"DEBUG: Prompt prepared - Total length: {len(prompt)} characters")
        print(f"DEBUG: External risk data included for {len(news_analysis_results)} companies")
        
        # Generate the analysis
        print("DEBUG: Calling LLM for comprehensive analysis...")
        result = llm_function(prompt)
        
        print("DEBUG: ✅ Analysis completed successfully")
        return result
        
    except Exception as e:
        print(f"DEBUG: ❌ Error in compare_and_recommend: {e}")
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        raise e
