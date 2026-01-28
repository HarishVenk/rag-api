import requests

def test_kubernetes_query():
    # Send q as JSON in the request body
    response = requests.post(
        "http://127.0.0.1:8000/query",
        json={"q": "What is Kubernetes?"}  # <-- JSON here
    )
    
    if response.status_code != 200:
        raise Exception(f"Server returned {response.status_code}: {response.text}")
    
    answer = response.json()["answer"]

    # Check for key concepts
    assert "orchestration" in answer.lower(), "Missing 'orchestration' keyword"
    assert "container" in answer.lower(), "Missing 'container' keyword"

    print("✅ Kubernetes query test passed")

if __name__ == "__main__":
    test_kubernetes_query()
    print("All semantic tests passed!")

