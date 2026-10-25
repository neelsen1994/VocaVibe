# VocaVibe — German Learning Backend

FastAPI + LangGraph backend for a German learning app. It is built around three ideas that set it
apart from Anki-style flashcard apps:

1. **Context-isolated flashcards.** A word with several meanings becomes one card *per sense*. A mistake
   on *eintreten = to occur* triggers drills on that sense only, never on *eintreten = to enter*.
2. **Multi-tense conjugation matrix.** Each verb sense comes with a single anchor sentence shown in
   Präsens, Präteritum and Perfekt. Only the verb changes, so the learner sees conjugation inside a fixed
   meaning, including auxiliary switches (*ist eingetreten* vs. *hat die Tür eingetreten*).
3. **Voice-to-confirmation fuzzy loop.** A misheard or mispronounced word is never silently auto-corrected
   or rejected. The app ranks what the learner probably meant (Levenshtein + Kölner Phonetik) and asks
   them to confirm, showing exactly what was off (*"drop 't'"*).

The persistence layer is in-memory for now. Its interfaces mirror the AWS services it will move to
(DynamoDB and OpenSearch Serverless).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

uvicorn app.main:app --reload     # http://localhost:8000/docs
pytest                            # runs fully offline
```

### API keys

Put keys in `german_learning_app/.env`. The file is git-ignored; copy `.env.example` if it is missing.
Real environment variables override it.

```env
ANTHROPIC_API_KEY=sk-ant-...
```

With an Anthropic key, every agent node calls Claude through the official `anthropic` SDK
(`beta.messages.parse` with Pydantic structured outputs and server-side refusal fallback). Without any
key the app runs in **offline mode**: heuristic intent detection, a curated lexicon (`eintreten` with
5 senses, `Bank` with 2 homonyms), a list of commonly confused lemmas for the candidate search, and a
deterministic hashing embedder. Anthropic has no embeddings API, so a Claude-only setup keeps the hashing
embedder.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `auto` | `auto` (Anthropic → OpenAI → offline), or force `anthropic` / `openai` / `offline` |
| `ANTHROPIC_API_KEY` | – | Turns on Claude |
| `ANTHROPIC_MODEL` | `claude-opus-5-5` | `claude-sonnet-5-5` or `claude-haiku-5-5` cost less per token |
| `ANTHROPIC_EFFORT` | `medium` | `low` … `max`: how much Claude thinks (and bills) per call |
| `LLM_TIMEOUT_SECONDS` | `120` | Per-request timeout |
| `OPENAI_API_KEY` | – | Alternative chat provider, and OpenAI embeddings |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model when the OpenAI provider is active |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Vector store embeddings |
| `MAX_CRITIQUE_RETRIES` | `2` | How many times the critic can send a draft back for repair |
| `FUZZY_MATCH_THRESHOLD` | `0.85` | Minimum similarity for a correct answer |
| `FLASHCARD_TTL_DAYS` / `HISTORY_TTL_DAYS` | `180` / `365` | DynamoDB TTL |
| `CORS_ORIGINS` | `http://localhost:5173` | Vite dev server |

## Layout

```
app/
├── main.py                  # app factory, routes, DI container, error mapping
├── config.py                # env-driven settings
├── agents/
│   ├── state.py             # AgentState TypedDict
│   ├── graph.py             # intent (+ confirmation gate) -> extraction <-> critique
│   └── validation.py        # deterministic checks: conjugation, auxiliary, tense matrix
├── services/
│   ├── llm_service.py       # LLM gateway, offline fallback, agent tools, embedder
│   ├── fuzzy_service.py     # Levenshtein, Kölner Phonetik, candidate ranking, answer scoring
│   ├── lexicon.py           # curated entries + candidate vocabulary
│   ├── storage_service.py   # MockDynamoDB, FlashcardRepository, MockVectorStore
│   └── practice_service.py  # sense-isolated RAG practice + server-side scoring
└── schemas/
    ├── dictionary.py        # strict LLM output contracts
    └── api.py               # HTTP request/response models
frontend/                    # placeholder for the React/Vite client
tests/                       # offline end-to-end tests, grouped by USP
```

## How each USP is implemented

### 1. Context-isolated flashcards

- `POST /api/flashcard/save` stores **one** `WordContext` per card. The card id is
  `sha256(word | english_gloss)`, so saving the same sense twice returns **409** and different senses
  never merge.
- Every failed attempt is written as an *error memory*: a DynamoDB history event plus an embedded summary
  tagged with the card id.
- `GET /api/flashcard/practice` searches the vector store for **error memories only**
  (`kind = "error"`). Each hit therefore points to the exact sense that was missed. The senses are ranked
  by accumulated similarity, and each one gets **freshly generated sentences** (`PracticeSentenceSet`,
  prompted to use that sense unambiguously). The generated sentences go through the same deterministic
  checks as dictionary entries.
- Other senses of a word under drill are **excluded** from the session. Any review cards added to fill
  the session come from other words.

### 2. Multi-tense conjugation matrix

- `WordContext.tense_matrix` holds one row per tense for verbs (`null` for other parts of speech).
- `agents/validation.py` enforces the matrix on every draft and sends violations back to the extractor:
  - exactly one row per tense;
  - each `used_conjugation` appears in its sentence and is a valid form of that tense. Perfekt is checked
    as auxiliary + participle, so senses that switch between *haben* and *sein* pass;
  - the rest of the sentence must stay the same across rows: at least 60% word overlap (Jaccard) once
    the verb is removed. A row that changes the context is rejected.
- Practice turns the matrix into `tense_transform` drills (*"Say this in the Perfekt: Der Ernstfall tritt
  ein."*). The drill uses the tense the learner most recently got wrong.

### 3. Voice-to-confirmation fuzzy loop

```
POST /api/analyze {text: "eintretten", source: "voice"}
  -> status: "needs_confirmation"
     candidates: [ {word: "eintreten",  edit_distance: 1, phonetic_match: true,  correction: "drop 't'"},
                   {word: "eintreffen", edit_distance: 2, phonetic_match: false, correction: "'tt' -> 'ff'"}, ... ]
POST /api/analyze {text: "eintreten", source: "voice", confirmed: true}
  -> status: "validated", result: {...}
```

- Candidates are ranked by `0.7 × Levenshtein similarity + 0.3 × Kölner Phonetik similarity`, after
  folding umlauts (ä/ae, ß/ss). Phonetics catches mispronunciations that sound right but are spelled far
  from the target. In online mode the LLM's `candidate_words` are added to the pool, but the same
  deterministic ranker orders them.
- Voice input that exactly matches a known form (e.g. *"tritt ein"*) goes straight to analysis. Typed input
  is never gated.
- The loop is **stateless**: two requests, no checkpointer, no session affinity. This keeps the API
  horizontally scalable.
- Answer scoring: an attempt counts as correct only if the **drilled form is present** (short words such
  as *ist* must match exactly) **and** the overall similarity reaches the threshold. A cloze can be
  answered with the missing words alone.

## Agent workflow

```
START -> intent_analysis --invalid----------------> END (HTTP 422)
              |        --voice, no exact match-----> END (needs_confirmation)
              v
   dictionary_extraction <------- errors and retries left
              |                         |
              v                         |
      grammar_critique -----------------+
              |
              v
             END   status = validated | unverified
```

The critic runs the cheap deterministic checks first. The LLM critic is only called on drafts that
pass them. The workflow allows 1 initial extraction plus `MAX_CRITIQUE_RETRIES` repairs. If problems
remain after that, the draft is returned as `unverified` with its `critique_errors`.

## API

| Method | Path | Description |
|---|---|---|
| POST | `/api/analyze` | `{text, source, confirmed?, requested_tense?}` → entry, or `needs_confirmation` + candidates |
| POST | `/api/flashcard/save` | Saves one sense (`context` + optional `conjugations`) as a card. 409 if that sense is already saved |
| GET | `/api/flashcard/practice?user_id=&limit=` | Builds and stores a practice set: `error_targeted`, `review` or `empty` |
| POST | `/api/flashcard/attempt` | `{user_id, exercise_id, transcript}` → scored server-side. Expected answers never reach the client |
| GET | `/health` | Liveness probe |

> `user_id` is sent in the request body for now. In production it will come from the Cognito JWT `sub`
> claim, checked by an API Gateway authorizer.

## AWS migration map (SAA-C03)

| Local component | AWS target | Notes |
|---|---|---|
| `MockDynamoDB` + `FlashcardRepository` | DynamoDB, on-demand, single table | `PK=USER#id`, `SK=CARD#…` / `EXERCISE#…` / `HISTORY#<iso-ts>#…`. TTL on `expires_at` (exercises: 24 h). `TransactWriteItems` for card + audit event |
| `MockVectorStore` | OpenSearch Serverless (vector search collection) | k-NN with `user_id` and `kind` filters. Error memories indexed through **DynamoDB Streams → Lambda** |
| `rank_candidates` / lexicon | Lemma list in S3 or DynamoDB; OpenSearch `fuzzy` queries at scale | Swap the pure-Python Levenshtein for `rapidfuzz` for large vocabularies |
| `HashingEmbedder` / OpenAI | Bedrock Titan Embeddings or OpenAI | Swap the `Embeddings` implementation in `LLMService.build_embedder` |
| `LLMService` | Bedrock (`ChatBedrockConverse`) or OpenAI | Pass any `BaseChatModel` to the constructor |
| FastAPI process | ECS Fargate behind an ALB (or Lambda + API Gateway) | Stateless, including the confirmation loop. Scale across multiple AZs |
| Speech-to-text (client) | Amazon Transcribe (streaming, `de-DE`) or the browser Web Speech API | The transcript is posted with `source: "voice"` |
| `OPENAI_API_KEY` env var | Secrets Manager | Injected through the task definition |
| `frontend/` | S3 + CloudFront with OAC | `/api/*` routed to the ALB origin |
| Python `logging` | CloudWatch Logs; X-Ray / ADOT for traces | |
