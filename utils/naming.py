import json
import re
import time
import streamlit as st

@st.cache_data
def generate_cluster_names(representative_keywords, openai_api_key, model="gpt-3.5-turbo"):
    """
    Generate descriptive names and descriptions for clusters using OpenAI.
    
    Args:
        representative_keywords (dict): Dictionary of cluster_id to list of representative keywords
        openai_api_key (str): OpenAI API key
        model (str): OpenAI model to use (gpt-3.5-turbo or gpt-4)
        
    Returns:
        dict: Dictionary of cluster_id to (name, description) tuples
    """
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=openai_api_key)
        
        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.info("Generating cluster names and descriptions...")
        
        # Process clusters in smaller batches to avoid context limitations
        cluster_ids = list(representative_keywords.keys())
        batch_size = 5  # Process 5 clusters at a time
        
        for batch_start in range(0, len(cluster_ids), batch_size):
            batch_end = min(batch_start + batch_size, len(cluster_ids))
            batch_cluster_ids = cluster_ids[batch_start:batch_end]
            
            # Create a prompt for this batch
            batch_prompt = """
            You are an SEO and content marketing expert. For each cluster of keywords, provide:
            1. A short, descriptive name (3-5 words)
            2. A brief description explaining the semantic theme (1-2 sentences)
            
            Format your response as JSON with this structure:
            {
              "clusters": [
                {
                  "cluster_id": 1,
                  "name": "Running Shoes and Footwear",
                  "description": "Keywords related to running shoes, athletic footwear, and running equipment."
                }
              ]
            }
            
            Here are the clusters:
            """
            
            for cluster_id in batch_cluster_ids:
                keywords = representative_keywords[cluster_id][:10]  # Use up to 10 keywords per cluster
                batch_prompt += f"\nCluster {cluster_id}: {', '.join(keywords)}"
            
            # Try up to 3 times if we hit errors
            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": batch_prompt}],
                        temperature=0.5,
                        response_format={"type": "json_object"} if model in ["gpt-4", "gpt-3.5-turbo"] else None
                    )
                    
                    content = response.choices[0].message.content.strip()
                    
                    # Extract JSON
                    try:
                        # Try to parse JSON directly
                        data = json.loads(content)
                        
                        if "clusters" in data and isinstance(data["clusters"], list):
                            # Process each cluster
                            for cluster_info in data["clusters"]:
                                if "cluster_id" in cluster_info and "name" in cluster_info and "description" in cluster_info:
                                    cluster_id = int(cluster_info["cluster_id"])
                                    name = cluster_info["name"]
                                    description = cluster_info["description"]
                                    
                                    if cluster_id in batch_cluster_ids:
                                        results[cluster_id] = (name, description)
                            
                            # If we've processed all clusters in this batch, break the retry loop
                            if all(cluster_id in results for cluster_id in batch_cluster_ids):
                                break
                    
                    except json.JSONDecodeError:
                        # If JSON parsing fails, try to extract with regex
                        for cluster_id in batch_cluster_ids:
                            pattern = rf'"cluster_id"\s*:\s*{cluster_id}\s*,\s*"name"\s*:\s*"([^"]+)"\s*,\s*"description"\s*:\s*"([^"]+)"'
                            matches = re.search(pattern, content)
                            
                            if matches:
                                name = matches.group(1)
                                description = matches.group(2)
                                results[cluster_id] = (name, description)
                    
                    except Exception as json_error:
                        st.warning(f"Error parsing JSON response: {str(json_error)}")
                
                except Exception as api_error:
                    st.warning(f"OpenAI API error on attempt {attempt + 1}: {str(api_error)}")
                    time.sleep(2)  # Wait before retrying
            
            # Update progress
            progress = min(1.0, batch_end / len(cluster_ids))
            progress_bar.progress(progress)
            status_text.info(f"Processed {batch_end}/{len(cluster_ids)} clusters...")
        
        # Fallback for any clusters that didn't get names
        for cluster_id in representative_keywords.keys():
            if cluster_id not in results:
                keywords = representative_keywords[cluster_id]
                name = f"Cluster {cluster_id}: {keywords[0] if keywords else ''}"
                description = f"Keywords related to {', '.join(keywords[:3]) if keywords else 'various topics'}"
                results[cluster_id] = (name, description)
        
        progress_bar.progress(1.0)
        status_text.success("✅ Cluster naming complete!")
        
        return results
    
    except Exception as e:
        st.error(f"Error generating cluster names: {str(e)}")
        
        # Fallback to generic names
        results = {}
        for cluster_id, keywords in representative_keywords.items():
            name = f"Cluster {cluster_id}"
            if keywords:
                name += f": {keywords[0]}"
                description = f"Keywords related to {', '.join(keywords[:3])}"
            else:
                description = f"Group of related keywords (cluster {cluster_id})"
            
            results[cluster_id] = (name, description)
        
        return results
