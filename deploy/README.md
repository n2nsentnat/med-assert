# Deployment

## Docker

Build and run the HTTP API (FastAPI on port 8000):

```bash
docker build -t article-miner:latest .
docker run --rm -p 8000:8000 --env-file .env article-miner:latest
```

The image installs optional **`[specter]`** extras (SPECTER 2 + PyTorch) so API/CLI dedup can use `--specter` / `enable_specter_faiss` in containers.

Use `.env` from the project root (see `.env.example`). The API listens on `0.0.0.0:8000`.

## Docker Compose

```bash
cp .env.example .env   # edit keys
docker compose up --build
```

Open [http://localhost:8000/docs](http://localhost:8000/docs). File-mode API outputs go to the named volume under `/app/article_miner_output` inside the container.

## Kubernetes

Requires a cluster with `kubectl` configured. Image must be available to your cluster (push to your registry and update the Deployment image, or use `kind load docker-image` for local testing).

```bash
kubectl apply -k deploy/kubernetes
```

Optional: create credentials for NCBI / LLM (keys become `env` inside the pod):

```bash
kubectl apply -f deploy/kubernetes/secret.example.yaml   # edit values first
# or
kubectl create secret generic article-miner-env -n article-miner \
  --from-literal=NCBI_API_KEY=... \
  --from-literal=OPENAI_API_KEY=...
```

Optional Ingress: edit `deploy/kubernetes/ingress.yaml` (host, TLS), uncomment the resource in `deploy/kubernetes/kustomization.yaml`, then `kubectl apply -k deploy/kubernetes`.

Port forward for a quick test:

```bash
kubectl port-forward -n article-miner svc/article-miner-api 8080:80
curl http://127.0.0.1:8080/health
```
