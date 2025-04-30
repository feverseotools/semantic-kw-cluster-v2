import re
import streamlit as st

# Search intent patterns for classification
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
            "facts about", "history of", "benefits of", "causes of", "types of",
            "what is", "how to", "why is"
        ],
        "patterns": [
            r'\bhow\s+to\b', r'\bwhat\s+is\b', r'\bwhy\s+is\b', r'\bwhen\s+to\b',
            r'\bwhere\s+to\b', r'\bwho\s+is\b', r'\bwhich\b.*\bbest\b',
            r'\bdefinition\b', r'\bmeaning\b', r'\bexamples?\b', r'\btips\b',
            r'\btutorials?\b', r'\bguide\b', r'\blearn\b', r'\bsteps?\b',
            r'\bversus\b', r'\bvs\b', r'\bcompared?\b', r'\bdifference\b'
        ],
        "weight": 1.0
    },
    
    "Navigational": {
        "prefixes": ["go to", "visit", "website", "homepage", "home page", "sign in", "login", "access"],
        "suffixes": ["login", "website", "homepage", "official", "online", "app", "portal"],
        "exact_matches": [
            "login", "sign in", "register", "create account", "download", "official website",
            "official site", "homepage", "contact", "support", "customer service", "app",
            "my account", "dashboard", "web portal"
        ],
        "patterns": [
            r'\blogin\b', r'\bsign\s+in\b', r'\bwebsite\b', r'\bhomepage\b', r'\bportal\b',
            r'\baccount\b', r'\bofficial\b', r'\bdashboard\b', r'\bdownload\b',
            r'\bcontact\b', r'\baddress\b', r'\blocation\b', r'\bdirections?\b',
            r'\bmap\b', r'\btrack\b.*\border\b', r'\bmy\s+\w+\s+account\b'
        ],
        "weight": 1.2
    },
    
    "Transactional": {
        "prefixes": ["buy", "purchase", "order", "shop", "get", "subscribe", "download", "install"],
        "suffixes": [
            "for sale", "discount", "deal", "coupon", "price", "cost", "cheap", "online",
            "free", "download", "subscription", "trial", "buy", "shop", "order", "purchase"
        ],
        "exact_matches": [
            "buy", "purchase", "order", "shop", "subscribe", "download", "free trial",
            "coupon code", "discount", "deal", "sale", "cheap", "best price", "near me",
            "shipping", "delivery", "in stock", "available", "pay", "checkout",
            "pricing", "cost of", "where to buy", "get a quote", "sign up"
        ],
        "patterns": [
            r'\bbuy\b', r'\bpurchase\b', r'\border\b', r'\bshop\b', r'\bstores?\b',
            r'\bprice\b', r'\bcost\b', r'\bcheap\b', r'\bdiscount\b', r'\bdeal\b',
            r'\bsale\b', r'\bcoupon\b', r'\bpromo\b', r'\bfree\s+shipping\b',
            r'\bnear\s+me\b', r'\bshipping\b', r'\bdelivery\b', r'\bcheck\s*out\b',
            r'\bin\s+stock\b', r'\bavailable\b', r'\bsubscribe\b', r'\bdownload\b',
            r'\binstall\b', r'\bfor\s+sale\b', r'\bhire\b', r'\brent\b'
        ],
        "weight": 1.5
    },
    
    "Commercial": {
        "prefixes": ["best", "top", "review", "compare", "vs", "versus", "alternative", "find", "choose"],
        "suffixes": [
            "review", "reviews", "comparison", "vs", "versus", "alternative", "alternatives",
            "recommendation", "recommendations", "comparison", "guide", "list", "ranking", "best", "top"
        ],
        "exact_matches": [
            "best", "top", "vs", "versus", "comparison", "compare", "review", "reviews",
            "rating", "ratings", "ranked", "recommended", "alternative", "alternatives",
            "pros and cons", "features", "worth it", "should i buy", "is it good",
            "which is best", "compare prices", "product review", "service review"
        ],
        "patterns": [
            r'\bbest\b', r'\btop\b', r'\breview\b', r'\bcompare\b', r'\bcompari(son|ng)\b',
            r'\bvs\b', r'\bversus\b', r'\balternatives?\b', r'\brated\b', r'\branking\b',
            r'\bworth\s+it\b', r'\bshould\s+I\s+buy\b', r'\bis\s+it\s+good\b',
            r'\bpros\s+and\s+cons\b', r'\badvantages?\b', r'\bdisadvantages?\b',
            r'\bfeatures\b', r'\bspecifications?\b', r'\bwhich\s+(is\s+)?(the\s+)?best\b',
            r'\btop\s+\d+\b', r'\blist\s+of\b', r'\bfind\s+(the\s+)?best\b'
        ],
        "weight": 1.2
    }
}

@st.cache_data
def classify_search_intent(keywords):
    """
    Classify keywords by search intent.
    
    Args:
        keywords (list): List of keywords to classify
        
    Returns:
        dict: Classification results with primary intent, scores, and evidence
    """
    # Initialize counters for each intent type
    intent_counts = {
        "Informational": 0,
        "Navigational": 0,
        "Transactional": 0,
        "Commercial": 0
    }
    
    # Evidence collection for each intent type
    evidence = {
        "Informational": set(),
        "Navigational": set(),
        "Transactional": set(),
        "Commercial": set()
    }
    
    # Analyze each keyword
    for keyword in keywords:
        keyword_lower = keyword.lower()
        
        # Check for prefixes
        first_word = keyword_lower.split()[0] if keyword_lower.split() else ""
        for intent, patterns in SEARCH_INTENT_PATTERNS.items():
            if first_word in patterns["prefixes"]:
                intent_counts[intent] += 1 * patterns["weight"]
                evidence[intent].add(f"Has {intent} prefix: '{first_word}'")
        
        # Check for suffixes
        last_word = keyword_lower.split()[-1] if keyword_lower.split() else ""
        for intent, patterns in SEARCH_INTENT_PATTERNS.items():
            if last_word in patterns["suffixes"]:
                intent_counts[intent] += 1 * patterns["weight"]
                evidence[intent].add(f"Has {intent} suffix: '{last_word}'")
        
        # Check for exact matches
        for intent, patterns in SEARCH_INTENT_PATTERNS.items():
            for exact_match in patterns["exact_matches"]:
                if exact_match in keyword_lower:
                    intent_counts[intent] += 2 * patterns["weight"]  # Higher weight for exact matches
                    evidence[intent].add(f"Contains '{exact_match}'")
        
        # Check for regex patterns
        for intent, patterns in SEARCH_INTENT_PATTERNS.items():
            for pattern in patterns["patterns"]:
                if re.search(pattern, keyword_lower):
                    intent_counts[intent] += 1.5 * patterns["weight"]  # Medium weight for pattern matches
                    evidence[intent].add(f"Matches pattern: '{pattern}'")
        
        # Check for special indicators
        if "near me" in keyword_lower or "nearby" in keyword_lower:
            intent_counts["Transactional"] += 2 * SEARCH_INTENT_PATTERNS["Transactional"]["weight"]
            evidence["Transactional"].add("Contains local intent indicator")
    
    # Calculate proportions
    total_score = sum(intent_counts.values()) or 1  # Avoid division by zero
    intent_scores = {
        intent: (count / total_score) * 100 
        for intent, count in intent_counts.items()
    }
    
    # Determine primary intent
    primary_intent = max(intent_scores, key=intent_scores.get)
    
    # If the highest score is below 40% or multiple intents are close, consider it mixed
    max_score = max(intent_scores.values())
    close_competitors = [intent for intent, score in intent_scores.items() 
                         if max_score - score < 15]  # Within 15 percentage points
    
    if max_score < 40 or len(close_competitors) > 1:
        primary_intent = "Mixed Intent"
    
    # Convert evidence sets to lists for JSON serialization
    evidence = {intent: list(items) for intent, items in evidence.items()}
    
    return {
        "primary_intent": primary_intent,
        "scores": intent_scores,
        "evidence": evidence
    }

@st.cache_data
def analyze_journey_phase(intent_result):
    """
    Map search intent to customer journey phase.
    
    Args:
        intent_result (dict): Search intent classification results
        
    Returns:
        str: Customer journey phase
    """
    primary_intent = intent_result["primary_intent"]
    scores = intent_result["scores"]
    
    # Simple mapping
    if primary_intent == "Informational":
        return "Early (Research/Awareness)"
    elif primary_intent == "Commercial":
        return "Middle (Consideration)"
    elif primary_intent == "Transactional":
        return "Late (Decision/Purchase)"
    elif primary_intent == "Navigational":
        # Navigational is special - can be at any stage
        return "Mixed"
    elif primary_intent == "Mixed Intent":
        # For mixed intent, look at the top two intents
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_two = [intent for intent, _ in sorted_intents[:2]]
        
        # Check common patterns
        if "Informational" in top_two and "Commercial" in top_two:
            return "Early (Research/Awareness)"
        elif "Commercial" in top_two and "Transactional" in top_two:
            return "Middle (Consideration)"
        elif "Informational" in top_two and "Transactional" in top_two:
            return "Mixed"
        else:
            return "Mixed"
    
    return "Mixed"
