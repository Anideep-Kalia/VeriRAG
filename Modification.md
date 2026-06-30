Context:
I am a Software Engineer at Oracle. I have already built VeriRAG — a Self-Correcting RAG system with the following features:

Query Rewriting
Hybrid Retrieval (Vector + BM25)
Cross Encoder Reranking
Context Compression
Faithfulness Verification
Regeneration Loops
Abstention Detection

What I want to do now:
Upgrade VeriRAG into a production-grade LLMOps / AI Platform project suitable for AI Platform Engineer and LLMOps Engineer roles.
What to add (decided and finalized):

RAGAS Evaluation Pipeline — automatically measure Faithfulness, Context Recall, Context Precision, Answer Relevance, Hallucination Rate
MLflow Experiment Tracking — store every evaluation run, compare versions, track metric history
GitHub Actions CI/CD with Evaluation-Gated Deployment — on every push, run eval benchmark, compare against baseline, block deployment if metrics regress
Prometheus + Grafana — track latency, token usage, cost per query, failure rate in production

What to skip (decided):

Kubernetes and Canary Deployments — too complex to demo solo, document in README instead
Full microservices split — Docker Compose is sufficient

Key architectural decision:
Kubernetes + canary will be documented in README as "designed for Kubernetes, currently running on Docker Compose" — to demonstrate architectural knowledge without implementation overhead.
Start from here: Help me upgrade VeriRAG phase by phase starting with the RAGAS Evaluation Pipeline.