
import re

def evaluate_document(parsed_output: dict) -> dict:
    """
    Evaluates the parsed document for legal risks based on predefined rules.
    
    Args:
        parsed_output (dict): The dictionary containing extracted text, entities, and ML insights.
        
    Returns:
        dict: A dictionary containing risk_score, risk_level, reasons, and missing_clauses.
    """
    risk_score = 0
    reasons = []
    missing_clauses = []
    
    # Extract relevant data
    text = parsed_output.get("extracted_text", "").lower()
    entities = parsed_output.get("legal_entities", [])
    ipc_sections = parsed_output.get("identified_bns_sections", [])
    ml_insights = parsed_output.get("ml_insights", {})
    
    # --- RULE 1: Bond Duration > 24 Months (+20 points) ---
    # Try to find duration in text or entities
    duration_months = 0
    # Simple regex for duration
    duration_match = re.search(r'(\d+)\s*(?:months?|years?)', text)
    if duration_match:
        val = int(duration_match.group(1))
        unit = duration_match.group(0).lower()
        if 'year' in unit:
            duration_months = val * 12
        else:
            duration_months = val
            
    if duration_months > 24:
        risk_score += 20
        reasons.append(f"Bond duration ({duration_months} months) exceeds recommended 24 months limit.")

    # --- RULE 2: Penalty Amount Excessive (+25 points) ---
    # Heuristic: If penalty mentioned and seems high (placeholder logic as we don't have salary info)
    # searching for "penalty" and high amounts
    penalty_matches = re.findall(r'penalty\s*(?:of)?\s*(?:rs\.?|inr)?\s*([\d,]+)', text)
    if penalty_matches:
        try:
            amounts = [int(p.replace(',', '')) for p in penalty_matches]
            if any(a > 200000 for a in amounts): # arbitrary threshold for "excessive" without salary context
                risk_score += 25
                reasons.append("High penalty amount detected (> ₹2,00,000).")
        except:
            pass

    # --- RULE 3: Missing Termination Clause (+15 points) ---
    if "termination" not in text and "exit clause" not in text:
        risk_score += 15
        reasons.append("Termination/Exit clause appears to be missing.")
        missing_clauses.append("Termination Clause")

    # --- RULE 4: Missing Dispute Resolution (+10 points) ---
    if "dispute" not in text and "arbitration" not in text and "jurisdiction" not in text:
        risk_score += 10
        reasons.append("Dispute Resolution/Arbitration clause missing.")
        missing_clauses.append("Dispute Resolution Clause")
        
    # --- RULE 5: Outdated IPC Sections (+10 points) ---
    # Check if any identified sections are IPC
    ipc_count = sum(1 for s in ipc_sections if "IPC" in s.get("ipc", "").upper())
    if ipc_count > 0:
        risk_score += 10
        reasons.append(f"Document references {ipc_count} outdated IPC sections (should use BNS).")

    # --- RULE 6: One-sided / Unfair Terms (+15 points) ---
    unfair_terms = ["absolute discretion", "unilateral", "without notice", "forfeit salary"]
    found_terms = [t for t in unfair_terms if t in text]
    if found_terms:
        risk_score += 15
        reasons.append(f"Potential one-sided terms detected: {', '.join(found_terms)}.")

    # --- Final Score & Classification ---
    risk_score = min(risk_score, 100)
    
    if risk_score <= 30:
        risk_level = "Legally Safe"
    elif risk_score <= 60:
        risk_level = "Needs Legal Review"
    else:
        risk_level = "High Legal Risk"
        
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "reasons": reasons,
        "missing_clauses": missing_clauses
    }
