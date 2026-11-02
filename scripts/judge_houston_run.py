"""LLM-judge proxy (RQ5) on Houston with real Food.com diners. Prompt v1, seed 7,
24 groups, k = 5, both diner orders. Resumable: skips every (model, group, selector,
repeat, order) already in the log and appends. Judges: gpt-4o-mini (OpenAI API),
qwen3.8-27b and allam-2-7b (Groq)."""
import os, sys, json, time, threading
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R); os.chdir(R)
import nbio, openai
from openai import OpenAI
SEED, N, K, TEMP = 7, 24, 5, 0.0
SELECTORS = ["average", "least_misery", "most_pleasure", "fair"]
LOG = f"runs/judge_{nbio.JUDGE_PROMPT_VERSION}_seed{SEED}_houston-tx-foodcom.jsonl"

class Retrying:
    def __init__(self, c): self._c = c; self.chat = self; self.completions = self
    def create(self, **kw):
        for attempt in range(60):
            try: return self._c.chat.completions.create(**kw)
            except openai.RateLimitError as e:
                wait = 15
                try: wait = float(e.response.headers.get("retry-after", 15)) + 1
                except Exception: pass
                time.sleep(min(wait, 120))
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError):
                time.sleep(10)
        raise RuntimeError("gave up after 60 retries")

groq = Retrying(OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1", timeout=60))
oai = Retrying(OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=60))
MODELS = {"gpt-4o-mini": oai, "qwen/qwen3.8-27b": groq, "allam-2-7b": groq}
LIMIT = int(os.environ.get("JUDGE_LIMIT", "0"))   # >0: smoke test, that many calls per model

done_keys = set()
for l in open(LOG):
    r = json.loads(l)
    done_keys.add((r["model"], r["group_id"], r["selector"], r["repeat"], r["reverse_order"]))

cat = nbio.load_catalog()
groups = nbio.generate_groups(N, seed=SEED, size_range=(3, 4))
items = []
for g in groups:
    for sel in SELECTORS:
        res = nbio.run_group(g, cat, method=sel)
        if res is None: continue
        opt = next(o for o in res["_F"] if o["option_id"] == res["pick_id"])
        items.append((g, sel, res, opt, res["_S"][res["pick_id"]]))
lock = threading.Lock(); fh = open(LOG, "a"); done = {m: 0 for m in MODELS}
todo = {m: sum(1 for g, s, *_ in items for rep in range(K) for rev in (False, True)
               if (m, g["group_id"], s, rep, rev) not in done_keys) for m in MODELS}
print(f"{len(items)} items; calls still to do: {todo}", flush=True)
def worker(model):
    client = MODELS[model]
    for g, sel, res, opt, sats in items:
        for rep in range(K):
            for rev in (False, True):
                if (model, g["group_id"], sel, rep, rev) in done_keys: continue
                if LIMIT and done[model] >= LIMIT: return
                r = nbio.judge_recommendation(g, opt, sats, client=client, model=model, temperature=TEMP, reverse_order=rev)
                r.update(group_id=g["group_id"], option_id=opt.get("option_id"), repeat=rep, selector=sel,
                         min_sat=res["min_sat"], jain=res["jain"], mean_sat=res["mean_sat"], option_name=opt["name"])
                with lock:
                    fh.write(json.dumps(r) + "\n"); fh.flush(); done[model] += 1
                    if done[model] % 100 == 0: print(f"  {model}: {done[model]} new calls", flush=True)
ts = [threading.Thread(target=worker, args=(m,)) for m in MODELS]
t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]; fh.close()
print(f"finished {sum(done.values())} new calls in {(time.time()-t0)/60:.1f} min -> {LOG}", flush=True)
