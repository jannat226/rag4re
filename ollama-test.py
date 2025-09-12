print("flag-0")
from llama_index.llms.ollama import Ollama
print("flag-1")
llm = Ollama(
    model="llama3.1:latest",
    request_timeout=120.0,
    # Manually set the context window to limit memory usage
    context_window=8000,
)
print("flag-2")
resp = llm.complete("Who is Paul Graham?")
print("flag-3")
print(resp)