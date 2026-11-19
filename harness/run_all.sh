#!/bin/zsh
# run_all.sh — reproduce every run in results/ from task files already built by prep.py.
#   MODELS=/path/to/models  OSS20B=/path/to/gpt-oss-20b-MXFP4.gguf  harness/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
M="${MODELS:?set MODELS to the folder holding the model files}"
H=harness

$H/serve.sh "$M/nomic-embed-text-v1.5.f16.gguf" 8791 embed 2048
for t in reg iss; do
  for m in random bm25; do python3 $H/rank.py --task private/tasks/$t.json --method $m; done
  for m in embed hybrid; do python3 $H/rank.py --task private/tasks/$t.json --method $m --embed-port 8791 --tag nomic137m; done
  python3 $H/judge.py --task private/tasks/$t.json --tag always-yes --always-yes
done
python3 $H/judge_trained.py --train private/tasks/reg.json --test private/tasks/iss.json --embed-port 8791
python3 $H/judge_trained.py --train private/tasks/iss.json --test private/tasks/reg.json --embed-port 8791

for spec in "OLMo-2-0425-1B-Instruct-F16.gguf olmo2-1b" "olmo-2-1124-7B-instruct-F16.gguf olmo2-7b"; do
  f=${spec%% *}; tag=${spec##* }
  [[ -f "$M/$f" ]] || { echo "skip $tag (no model file)"; continue; }
  $H/serve.sh "$M/$f" 8792 chat 8192
  for t in reg iss; do
    python3 $H/judge.py --task private/tasks/$t.json --chat-port 8792 --tag $tag
    [[ $tag == olmo2-1b ]] && python3 $H/judge.py --task private/tasks/$t.json --chat-port 8792 --tag $tag-fewshot --fewshot
    python3 $H/rank.py  --task private/tasks/$t.json --method judge --embed-port 8791 --chat-port 8792 --k 5 --tag $tag
  done
  $H/serve.sh stop 8792
done

if [[ -n "${OSS20B:-}" ]]; then
  $H/serve.sh "$OSS20B" 8793 chat 8192
  for t in reg iss; do python3 $H/judge.py --task private/tasks/$t.json --chat-port 8793 --tag gptoss20b-local-ref --generate; done
  $H/serve.sh stop 8793
fi
$H/serve.sh stop 8791
python3 $H/summarize.py private/runs results/summary-v0.2.json
