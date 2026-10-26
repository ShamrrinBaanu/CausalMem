"""Environment-backed application configuration."""
from dataclasses import dataclass
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@dataclass(frozen=True)
class Settings:
    provider: str = os.getenv("LLM_PROVIDER", "openai").lower()
    model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    base_url: str = os.getenv("LLM_BASE_URL", "")
    ollama_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    database_path: str = os.getenv("CAUSALMEM_DB", "data/causalmem.sqlite3")
    mem0_path: str = os.getenv("MEM0_PATH", "data/mem0")
    default_user: str = os.getenv("CAUSALMEM_USER", "default")
    @property
    def ollama_server_url(self): return self.ollama_url.rstrip("/").removesuffix("/v1")
    @property
    def effective_base_url(self):
        if self.base_url: return self.base_url
        if self.provider != "ollama": return ""
        root=self.ollama_url.rstrip("/")
        return root if root.endswith("/v1") else root+"/v1"
    @property
    def effective_api_key(self): return "ollama" if self.provider == "ollama" else self.api_key

settings = Settings()
Path(settings.database_path).parent.mkdir(parents=True, exist_ok=True)
Path(settings.mem0_path).mkdir(parents=True, exist_ok=True)
# Mem0 OSS creates a small telemetry/user config directory during import.
# Keep that side effect within the project data directory for portability.
os.environ.setdefault("MEM0_DIR", str(Path(settings.mem0_path).parent / "mem0-config"))
