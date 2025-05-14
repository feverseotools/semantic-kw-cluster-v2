"""
Search intent classification module for Semantic Keyword Clustering.

This module contains functions to analyze and classify keywords
into different search intent categories (informational, navigational,
transactional, commercial) and analyze customer journey flow.
"""

import re
import logging
from collections import Counter

# Configure logging
logger = logging.getLogger(__name__)

# Search intent classification patterns
# These are comprehensive patterns based on SEO industry standards
SEARCH_INTENT_PATTERNS = {
    "Informational": {
        "prefixes": [
            "how", "what", "why", "when", "where", "who", "which",
            "can", "does", "is", "are", "will", "should", "do", "did",
            "guide", "tutorial", "learn", "understand", "explain"
        ],
        "suffixes": ["definition", "meaning", "examples", "ideas", "guide", "tutorial"],
        "exact_matches": [
            "guide to", "how-to", "tutorial", "resources", "information", "knowledge",
            "examples of", "definition of", "explanation", "steps to", "learn about",
            "facts about", "history of", "benefits of", "causes of", "types of"
        ],
        "keyword_patterns": [
            r'\bhow\s+to\b', r'\bwhat\s+is\b', r'\bwhy\s+is\b', r'\bwhen\s+to\b', 
            r'\bwhere\s+to\b', r'\bwho\s+is\b', r'\bwhich\b.*\bbest\b',
            r'\bdefinition\b', r'\bmeaning\b', r'\bexamples?\b', r'\btips\b',
            r'\btutorials?\b', r'\bguide\b', r'\blearn\b', r'\bsteps?\b',
            r'\bversus\b', r'\bvs\b', r'\bcompared?\b', r'\bdifference\b'
        ],
        "weight": 1.0
    },
    
    "Navigational": {
        "prefixes": ["go to", "visit", "website", "homepage", "home page", "sign in", "login"],
        "suffixes": ["login", "website", "homepage", "official", "online"],
        "exact_matches": [
            "login", "sign in", "register", "create account", "download", "official website",
            "official site", "homepage", "contact", "support", "customer service", "app"
        ],
        "keyword_patterns": [
            r'\blogin\b', r'\bsign\s+in\b', r'\bwebsite\b', r'\bhomepage\b', r'\bportal\b',
            r'\baccount\b', r'\bofficial\b', r'\bdashboard\b', r'\bdownload\b.*\bfrom\b',
            r'\bcontact\b', r'\baddress\b', r'\blocation\b', r'\bdirections?\b',
            r'\bmap\b', r'\btrack\b.*\border\b', r'\bmy\s+\w+\s+account\b'
        ],
        "brand_indicators": True,  # Presence of brand names indicates navigational intent
        "weight": 1.2  # Navigational intent is often more clear-cut
    },
    
    "Transactional": {
        "prefixes": ["buy", "purchase", "order", "shop", "get"],
        "suffixes": [
            "for sale", "discount", "deal", "coupon", "price", "cost", "cheap", "online", 
            "free", "download", "subscription", "trial"
        ],
        "exact_matches": [
            "buy", "purchase", "order", "shop", "subscribe", "download", "free trial",
            "coupon code", "discount", "deal", "sale", "cheap", "best price", "near me",
            "shipping", "delivery", "in stock", "available", "pay", "checkout"
        ],
        "keyword_patterns": [
            r'\bbuy\b', r'\bpurchase\b', r'\border\b', r'\bshop\b', r'\bstores?\b',
            r'\bprice\b', r'\bcost\b', r'\bcheap\b', r'\bdiscount\b', r'\bdeal\b',
            r'\bsale\b', r'\bcoupon\b', r'\bpromo\b', r'\bfree\s+shipping\b',
            r'\bnear\s+me\b', r'\bshipping\b', r'\bdelivery\b', r'\bcheck\s*out\b',
            r'\bin\s+stock\b', r'\bavailable\b', r'\bsubscribe\b', r'\bdownload\b',
            r'\binstall\b', r'\bfor\s+sale\b', r'\bhire\b', r'\brent\b'
        ],
        "weight": 1.5  # Strong transactional signals are highly valuable
    },
    
    "Commercial": {
        "prefixes": ["best", "top", "review", "compare", "vs", "versus"],
        "suffixes": [
            "review", "reviews", "comparison", "vs", "versus", "alternative", "alternatives", 
            "recommendation", "recommendations", "comparison", "guide"
        ],
        "exact_matches": [
            "best", "top", "vs", "versus", "comparison", "compare", "review", "reviews", 
            "rating", "ratings", "ranked", "recommended", "alternative", "alternatives",
            "pros and cons", "features", "worth it", "should i buy", "is it good"
        ],
        "keyword_patterns": [
            r'\bbest\b', r'\btop\b', r'\breview\b', r'\bcompare\b', r'\bcompari(son|ng)\b', 
            r'\bvs\b', r'\bversus\b', r'\balternatives?\b', r'\brated\b', r'\branking\b',
            r'\bworth\s+it\b', r'\bshould\s+I\s+buy\b', r'\bis\s+it\s+good\b',
            r'\bpros\s+and\s+cons\b', r'\badvantages?\b', r'\bdisadvantages?\b',
            r'\bfeatures\b', r'\bspecifications?\b', r'\bwhich\s+(is\s+)?(the\s+)?best\b'
        ],
        "weight": 1.2  # Commercial intent signals future transactions
    }
}

def extract_features_for_intent(keyword, search_intent_description=""):
    """
    Extract features for search intent classification based on keyword patterns.
    
    Args:
        keyword (str): The keyword to analyze
        search_intent_description (str, optional): Additional intent description
        
    Returns:
        dict: Dictionary of features for classification
    """
    # Features to extract
    features = {
        "keyword_length": len(keyword.split()),
        "keyword_lower": keyword.lower(),
        "has_informational_prefix": False,
        "has_navigational_prefix": False,
        "has_transactional_prefix": False,
        "has_commercial_prefix": False,
        "has_informational_suffix": False,
        "has_navigational_suffix": False,
        "has_transactional_suffix": False,
        "has_commercial_suffix": False,
        "is_informational_exact_match": False,
        "is_navigational_exact_match": False,
        "is_transactional_exact_match": False,
        "is_commercial_exact_match": False,
        "informational_pattern_matches": 0,
        "navigational_pattern_matches": 0,
        "transactional_pattern_matches": 0,
        "commercial_pattern_matches": 0,
        "includes_brand": False,
        "includes_product_modifier": False,
        "includes_price_modifier": False,
        "local_intent": False,
        "modal_verbs": False  # signals a question typically
    }
    
    keyword_lower = keyword.lower()
    
    # Check prefixes
    words = keyword_lower.split()
    if words:
        first_word = words[0]
        for intent_type, patterns in SEARCH_INTENT_PATTERNS.items():
            if any(first_word == prefix.lower() for prefix in patterns["prefixes"]):
                features[f"has_{intent_type.lower()}_prefix"] = True
    
    # Check suffixes 
    if words and len(words) > 1:
        last_word = words[-1]
        for intent_type, patterns in SEARCH_INTENT_PATTERNS.items():
            if any(last_word == suffix.lower() for suffix in patterns["suffixes"]):
                features[f"has_{intent_type.lower()}_suffix"] = True
    
    # Check exact matches
    for intent_type, patterns in SEARCH_INTENT_PATTERNS.items():
        for exact_match in patterns["exact_matches"]:
            if exact_match.lower() in keyword_lower:
                features[f"is_{intent_type.lower()}_exact_match"] = True
                break
    
    # Check pattern matches
    for intent_type, patterns in SEARCH_INTENT_PATTERNS.items():
        match_count = 0
        for pattern in patterns["keyword_patterns"]:
            if re.search(pattern, keyword_lower):
                match_count += 1
        features[f"{intent_type.lower()}_pattern_matches"] = match_count
    
    # Additional features
    features["local_intent"] = any(term in keyword_lower for term in ["near me", "nearby", "in my area", "close to me", "closest", "local"])
    features["modal_verbs"] = any(modal in keyword_lower.split() for modal in ["can", "could", "should", "would", "will", "may", "might"])
    features["includes_price_modifier"] = any(term in keyword_lower for term in ["price", "cost", "cheap", "expensive", "affordable", "discount", "offer", "deal", "coupon"])
    features["includes_product_modifier"] = any(term in keyword_lower for term in ["best", "top", "cheap", "premium", "quality", "new", "used", "refurbished", "alternative"])
    
    return features

def classify_search_intent(keywords, search_intent_description="", cluster_name=""):
    """
    Classify keywords into search intent categories (informational, navigational, 
    transactional, commercial).
    
    Args:
        keywords (list): List of keywords to classify
        search_intent_description (str, optional): Additional intent description
        cluster_name (str, optional): Name of the cluster being analyzed
        
    Returns:
        dict: Classification results with primary intent, scores, and evidence
    """
    if not keywords:
        return {
            "primary_intent": "Unknown",
            "scores": {
                "Informational": 25,
                "Navigational": 25,
                "Transactional": 25,
                "Commercial": 25
            },
            "evidence": {}
        }
    
    # Extract features for all keywords
    all_features = []
    for keyword in keywords[:min(len(keywords), 20)]:  # Limit to first 20 keywords for performance
        features = extract_features_for_intent(keyword, search_intent_description)
        all_features.append(features)
    
    # Aggregate features
    informational_signals = []
    navigational_signals = []
    transactional_signals = []
    commercial_signals = []
    
    # Count pattern matches across all features
    for features in all_features:
        # Informational signals
        if features["has_informational_prefix"]:
            informational_signals.append("Has informational prefix")
        if features["has_informational_suffix"]:
            informational_signals.append("Has informational suffix")
        if features["is_informational_exact_match"]:
            informational_signals.append("Contains informational phrase")
        if features["informational_pattern_matches"] > 0:
            informational_signals.append(f"Matches {features['informational_pattern_matches']} informational patterns")
        if features["modal_verbs"]:
            informational_signals.append("Contains question-like modal verb")
            
        # Navigational signals
        if features["has_navigational_prefix"]:
            navigational_signals.append("Has navigational prefix")
        if features["has_navigational_suffix"]:
            navigational_signals.append("Has navigational suffix")
        if features["is_navigational_exact_match"]:
            navigational_signals.append("Contains navigational phrase")
        if features["navigational_pattern_matches"] > 0:
            navigational_signals.append(f"Matches {features['navigational_pattern_matches']} navigational patterns")
        if features["includes_brand"]:
            navigational_signals.append("Includes brand name")
            
        # Transactional signals
        if features["has_transactional_prefix"]:
            transactional_signals.append("Has transactional prefix")
        if features["has_transactional_suffix"]:
            transactional_signals.append("Has transactional suffix")
        if features["is_transactional_exact_match"]:
            transactional_signals.append("Contains transactional phrase")
        if features["transactional_pattern_matches"] > 0:
            transactional_signals.append(f"Matches {features['transactional_pattern_matches']} transactional patterns")
        if features["includes_price_modifier"]:
            transactional_signals.append("Includes price-related term")
        if features["local_intent"]:
            transactional_signals.append("Shows local intent (near me, etc.)")
            
        # Commercial signals
        if features["has_commercial_prefix"]:
            commercial_signals.append("Has commercial prefix")
        if features["has_commercial_suffix"]:
            commercial_signals.append("Has commercial suffix")
        if features["is_commercial_exact_match"]:
            commercial_signals.append("Contains commercial phrase")
        if features["commercial_pattern_matches"] > 0:
            commercial_signals.append(f"Matches {features['commercial_pattern_matches']} commercial patterns")
        if features["includes_product_modifier"]:
            commercial_signals.append("Includes product comparison term")
    
    # Calculate scores based on unique signals
    info_signals = set(informational_signals)
    nav_signals = set(navigational_signals)
    trans_signals = set(transactional_signals)
    comm_signals = set(commercial_signals)
    
    # Calculate relative proportions (with weighting)
    info_weight = SEARCH_INTENT_PATTERNS["Informational"]["weight"]
    nav_weight = SEARCH_INTENT_PATTERNS["Navigational"]["weight"]
    trans_weight = SEARCH_INTENT_PATTERNS["Transactional"]["weight"]
    comm_weight = SEARCH_INTENT_PATTERNS["Commercial"]["weight"]
    
    info_score = len(info_signals) * info_weight
    nav_score = len(nav_signals) * nav_weight
    trans_score = len(trans_signals) * trans_weight
    comm_score = len(comm_signals) * comm_weight
    
    # Check description for explicit mentions
    if search_intent_description:
        desc_lower = search_intent_description.lower()
        if re.search(r'\binformational\b|\binformation\s+intent\b|\binformation\s+search\b|\bleaning\b|\bquestion\b', desc_lower):
            info_score += 5
        if re.search(r'\bnavigational\b|\bnavigate\b|\bfind\s+\w+\s+website\b|\bfind\s+\w+\s+page\b|\baccess\b', desc_lower):
            nav_score += 5
        if re.search(r'\btransactional\b|\bbuy\b|\bpurchase\b|\bshopping\b|\bsale\b|\btransaction\b', desc_lower):
            trans_score += 5
        if re.search(r'\bcommercial\b|\bcompar(e|ing|ison)\b|\breview\b|\balternative\b|\bbest\b', desc_lower):
            comm_score += 5
    
    # Check cluster name for signals
    if cluster_name:
        name_lower = cluster_name.lower()
        if re.search(r'\bhow\b|\bwhat\b|\bwhy\b|\bwhen\b|\bguide\b|\btutorial\b', name_lower):
            info_score += 3
        if re.search(r'\bwebsite\b|\bofficial\b|\blogin\b|\bportal\b|\bdownload\b', name_lower):
            nav_score += 3
        if re.search(r'\bbuy\b|\bshop\b|\bpurchase\b|\bsale\b|\bdiscount\b|\bcost\b|\bprice\b', name_lower):
            trans_score += 3
        if re.search(r'\bbest\b|\btop\b|\breview\b|\bcompare\b|\bvs\b|\balternative\b', name_lower):
            comm_score += 3
    
    # Normalize to percentages
    total_score = max(1, info_score + nav_score + trans_score + comm_score)
    info_pct = (info_score / total_score) * 100
    nav_pct = (nav_score / total_score) * 100
    trans_pct = (trans_score / total_score) * 100
    comm_pct = (comm_score / total_score) * 100
    
    # Prepare scores
    scores = {
        "Informational": info_pct,
        "Navigational": nav_pct,
        "Transactional": trans_pct,
        "Commercial": comm_pct
    }
    
    # Collect evidence for the primary intent
    evidence = {
        "Informational": list(info_signals),
        "Navigational": list(nav_signals),
        "Transactional": list(trans_signals),
        "Commercial": list(comm_signals)
    }
    
    # Find primary intent (highest score)
    primary_intent = max(scores, key=scores.get)

    # If the highest score is less than 30%, consider mixed intent cases
    max_score = max(scores.values())
    if max_score < 30:
        # Get sorted intent scores
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Case 1: Close competition between top intents (difference < 10%)
        if len(sorted_intents) > 1 and (sorted_intents[0][1] - sorted_intents[1][1] < 10):
            primary_intent = "Mixed Intent"
            # Add information about the mixed intents in the evidence
            evidence["mixed_types"] = [sorted_intents[0][0], sorted_intents[1][0]]
        
        # Case 2: Extremely low confidence (all scores are low)
        elif max_score < 20:
            primary_intent = "Ambiguous Intent"
        
        # For other low confidence cases, we keep the top intent but mark low confidence
        else:
            evidence["low_confidence"] = True
    
    return {
        "primary_intent": primary_intent,
        "scores": scores,
        "evidence": evidence
    }

def analyze_cluster_for_intent_flow(df, cluster_id):
    """
    Analyze the intent distribution within a cluster to map customer journey flow.
    
    Args:
        df (pandas.DataFrame): DataFrame with keywords and cluster IDs
        cluster_id: The ID of the cluster to analyze
        
    Returns:
        dict: Intent flow analysis with distribution, journey phase, etc.
    """
    # Get keywords for this cluster
    cluster_keywords = df[df['cluster_id'] == cluster_id]['keyword'].tolist()
    
    if not cluster_keywords:
        logger.warning(f"No keywords found for cluster {cluster_id}")
        return None
    
    # Classify each keyword individually
    keyword_intents = []
    for keyword in cluster_keywords:
        intent_data = classify_search_intent([keyword])
        keyword_intents.append({
            "keyword": keyword,
            "primary_intent": intent_data["primary_intent"],
            "scores": intent_data["scores"]
        })
    
    # Calculate distribution of intents
    intent_counts = Counter([item["primary_intent"] for item in keyword_intents])
    total = len(keyword_intents)
    
    # Calculate average scores across all keywords
    avg_scores = {
        "Informational": sum(item["scores"]["Informational"] for item in keyword_intents) / total,
        "Navigational": sum(item["scores"]["Navigational"] for item in keyword_intents) / total,
        "Transactional": sum(item["scores"]["Transactional"] for item in keyword_intents) / total,
        "Commercial": sum(item["scores"]["Commercial"] for item in keyword_intents) / total
    }
    
    # Analyze if this represents a customer journey phase
    # Typically, customer journey: Info -> Commercial -> Transactional
    journey_phase = None
    
    # Simple journey phase detection
    info_pct = (intent_counts.get("Informational", 0) / total) * 100
    comm_pct = (intent_counts.get("Commercial", 0) / total) * 100
    trans_pct = (intent_counts.get("Transactional", 0) / total) * 100
    
    if info_pct > 50:
        journey_phase = "Early (Research Phase)"
    elif comm_pct > 50:
        journey_phase = "Middle (Consideration Phase)"
    elif trans_pct > 50:
        journey_phase = "Late (Purchase Phase)"
    elif info_pct > 25 and comm_pct > 25:
        journey_phase = "Research-to-Consideration Transition"
    elif comm_pct > 25 and trans_pct > 25:
        journey_phase = "Consideration-to-Purchase Transition"
    else:
        journey_phase = "Mixed Journey Stages"
    
    return {
        "intent_distribution": {intent: (count / total) * 100 for intent, count in intent_counts.items()},
        "avg_scores": avg_scores,
        "journey_phase": journey_phase,
        "keyword_sample": [{"keyword": k["keyword"], "intent": k["primary_intent"]} for k in keyword_intents[:10]]
    }
