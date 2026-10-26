from config import Settings

def test_ollama_endpoints_accept_root_or_v1_suffix():
    root=Settings(provider="ollama",ollama_url="http://localhost:11434")
    suffixed=Settings(provider="ollama",ollama_url="http://localhost:11434/v1")
    assert root.effective_base_url == "http://localhost:11434/v1"
    assert root.ollama_server_url == "http://localhost:11434"
    assert suffixed.effective_base_url == "http://localhost:11434/v1"
    assert suffixed.ollama_server_url == "http://localhost:11434"
    assert root.effective_api_key == "ollama"
