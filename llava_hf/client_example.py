#!/usr/bin/env python3
"""
Simple client to test the /fetch_material endpoint
"""
import requests
import json
import sys

def fetch_material_client(query, host='localhost', port=5000):
    """
    Send a request to the fetch_material API endpoint
    
    Args:
        query: Text description of the material you want to fetch
        host: Server host (default: localhost)
        port: Server port (default: 5000)
    
    Returns:
        Response from the server
    """
    url = f"http://{host}:{port}/fetch_material"
    
    payload = {
        'query': query
    }
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    try:
        print(f"Sending request to {url}")
        print(f"Query: {query}")
        print("-" * 50)
        
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\nSuccess! Response:")
            print(json.dumps(result, indent=2))
            return result
        else:
            error = response.json()
            print(f"\nError: {error}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to {url}")
        print("Make sure the server is running!")
        return None
    except requests.exceptions.Timeout:
        print("Error: Request timed out")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        # Default example query
        query = "wooden table material"
    
    result = fetch_material_client(query)
    
    if result:
        print("\n" + "=" * 50)
        print("Material fetched successfully!")
    else:
        print("\n" + "=" * 50)
        print("Failed to fetch material")
        sys.exit(1)
