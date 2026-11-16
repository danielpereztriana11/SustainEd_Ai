# SustainEd AI

A sustainability tutor for employees and small-business owners. A Kubeflow pipeline, run on Vertex AI, fine-tunes `microsoft/Phi-3-mini-4k-instruct` with LoRA so that it answers every question in four parts: a definition, an example in the learner's business, a practical action and a knowledge-check question. The pipeline scores the fine-tuned model and the base model on a held-out test set. The fine-tuned model is then served on a Vertex AI endpoint, with a Chainlit chat app in front of it and Langfuse monitoring.

```
User:  I run a small coffee shop. What are Scope 3 emissions? Explain for a beginner.

Model: Definition: Scope 3 emissions are the indirect emissions in your value chain that are not covered by Scope 2. ...
       Example: For your coffee shop, they include growing and shipping the coffee beans and making the cups you buy.
       Action: List your five biggest purchases and ask each supplier what they know about their emissions.
       Check: Is the electricity you buy for the shop normally Scope 2 or Scope 3?
```

This is an educational assistant. It does not give legal or regulatory advice.

## The four parts of the project

| Part | What it covers | Main files |
|---|---|---|
| 1. Training pipeline | Dataset, data preparation, LoRA fine-tuning, inference, evaluation | `scripts/build_dataset.py`, `src/pipeline_components/`, `src/pipelines/` |
| 2. Deployment | Model Registry, endpoint | `scripts/register_model_with_custom_handler.py`, `src/handler.py` |
| 3. Chat interface | Chainlit app calling the endpoint | `src/app/main.py` |
| 4. Monitoring | Langfuse traces, LLM-as-a-judge, dashboard | `src/app/main.py`, this README |

## Setup

Requirements: Python 3.11.6, [`uv`](https://docs.astral.sh/uv/), the `gcloud` CLI, and a GCP project with billing enabled.

```bash
uv sync
cp .env.example .env    # fill in GCP_PROJECT_ID, GCP_REGION and GCP_BUCKET_NAME
```

Without `uv`, install the pinned versions with `pip` in a Python 3.11 virtual environment, then drop the `uv run` prefix from the commands below:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Connecting to Google Cloud

```bash
# 1. Sign in and select the project
gcloud auth login
gcloud config set project <GCP_PROJECT_ID>

# 2. Credentials used by the Python scripts (Application Default Credentials)
gcloud auth application-default login
gcloud auth application-default set-quota-project <GCP_PROJECT_ID>

# 3. Enable Vertex AI and Cloud Storage (once per project)
gcloud services enable aiplatform.googleapis.com storage.googleapis.com

# 4. Create the bucket (skip this if you reuse an existing bucket)
gcloud storage buckets create gs://<GCP_BUCKET_NAME> --location=<GCP_REGION>
```

The pipeline steps run as the project's Compute Engine default service account. It needs to read and write the bucket. On a new project, grant it the role once:

```bash
gcloud projects add-iam-policy-binding <GCP_PROJECT_ID> \
    --member="serviceAccount:<GCP_PROJECT_NUMBER>-compute@developer.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"
```

> **GPU quota:** the fine-tuning and inference steps each request one `NVIDIA_TESLA_T4`, one at a time. `GCP_REGION` must be a region where your project has T4 quota for Vertex AI custom training.

## Part 1: dataset and training pipeline

### 1.1 Build the dataset

The model learns the four-part format from examples written in that format. The Hugging Face dataset [`nessa01macias/sustainability_qa`](https://huggingface.co/datasets/nessa01macias/sustainability_qa) has plain questions and answers, most of them about one specific paper, and its card states no license. It is therefore used as a source of question seeds, not as training text.

```bash
# 1. Download the Hugging Face dataset and keep the general questions (data/hf_candidates.csv)
uv run python -m scripts.download_hf_dataset

# 2. Pilot: draft about 50 examples and read them
uv run python -m scripts.build_dataset --limit-concepts 10

# 3. Full dataset: 120 concepts x 5 variants = 600 examples (480 train / 60 validation / 60 test)
uv run python -m scripts.build_dataset

# 4. After the review, copy the CSV to the bucket
gcloud storage cp data/sustained_dataset.csv gs://<GCP_BUCKET_NAME>/data/sustained_dataset.csv
```

How `build_dataset` works:

- **Concepts:** six themes of 20 concepts each, listed in [src/taxonomy.py](src/taxonomy.py). With `data/hf_candidates.csv` present, five concepts per theme are replaced with Hugging Face questions (`--hf-per-theme`, 0 to turn this off).
- **Variants:** each concept gets five examples, one per combination of learner level (beginner, intermediate, advanced) and business profile (coffee shop, online clothing store, furniture workshop, IT consulting firm). The 12 combinations are used equally often.
- **Drafting:** Gemini on Vertex AI writes the question and the four parts. It uses your `gcloud` credentials, so it needs no extra API key. `GEMINI_MODEL` in `.env` selects the model. Drafts are cached in `data/drafts/`, so an interrupted run restarts where it stopped.
- **Split:** all the variants of a concept go to the same split, so no paraphrase of a training question ends up in the test set.

The CSV has the record fields of the project brief, plus `concept`:

`id, topic, concept, learner_level, business_sector, instruction, target_response, factual_sources, license_status, reviewer_status, dataset_split`

**Review before training.** Every record starts with `reviewer_status = draft`. Gemini can write things that are wrong or repetitive: read the examples, correct `target_response` against the reference material (GHG Protocol, US EPA small-business guide, European Commission VSME Q&A), then set `reviewer_status` to `reviewed`. To train on reviewed examples only, set `require_reviewed=True` on `prepare_sustained_data`. `build_dataset` refuses to overwrite a CSV that contains reviewed records unless you add `--overwrite`.

### 1.2 Pipeline

```
prepare_sustained_data
 ├─ train, validation ─▶ fine_tuning_component (T4 GPU)
 │                             └─ model ─┐
 ├─ test ────────────────────────────────┴─▶ inference_component (T4 GPU) ─▶ evaluation_component
 └─ test ──────────────────────────────────▶ base_inference_component (T4 GPU) ─▶ evaluation_component
```

| Step | What it does | Outputs |
|---|---|---|
| `prepare_sustained_data` | Normalizes whitespace, drops empty rows, answers without the four parts and duplicate questions, converts each row to chat messages, drops conversations longer than 512 tokens (base model tokenizer) and writes the three splits | `train_output_path`, `validation_output_path`, `test_output_path` (CSV), `metrics` (row counts) |
| `fine_tuning_component` | Loads Phi-3 in 4 bits, adds LoRA adapters and trains with `trl`'s `SFTTrainer`, with the validation set evaluated at each epoch | `model` (LoRA adapters + tokenizer), `metrics` (losses, hyperparameters, TensorBoard logs) |
| `inference_component` | Generates an answer for each test question with the fine-tuned model | `predictions` (CSV) |
| `base_inference_component` | Same with the base model, as the point of comparison | `predictions` (CSV) |
| `evaluation_component` | Scores each prediction (run once per model) | `evaluation_results` (per-sample CSV), `metrics` (means) |

Evaluation metrics, all computed without an LLM:

| Metric | Meaning |
|---|---|
| `rouge_score`, `bleu_score` | Overlap with the reference answer (`ragas`) |
| `format_adherence` | 1 if the four parts are present, not empty and in order. Its mean is the share of answers with all four sections |
| `has_definition`, `has_example`, `has_action`, `has_check` | Presence of each part |
| `check_is_question` | The Check part ends with a question mark |
| `sector_in_example` | The Example part names the learner's business |
| `reading_grade` | Flesch-Kincaid grade level. `reading_grade_<level>_mean` shows whether advanced answers are harder to read than beginner ones |
| `word_count` | Length of the answer |

These metrics measure format, relevance to the business and level. They do not measure factual accuracy: review a sample of the `evaluation_results` CSV by hand, or use the LLM judge described in the monitoring section.

Fine-tuning hyperparameters (LoRA rank, learning rate, epochs and so on) are parameters of `fine_tuning_component`, and their defaults are in [fine_tuning_component.py](src/pipeline_components/fine_tuning_component.py).

### 1.3 Run it

Run all commands from the root of the repository.

```bash
# Run the unit tests
uv run pytest

# Run the data preparation locally (writes data/train_sustained.csv, validation and test)
uv run python -m scripts.process_data_locally

# Compile the pipeline to sustained_pipeline.yaml and submit it to Vertex AI
uv run python -m scripts.pipeline_runner
```

The runner prints a link to the run in the Vertex Pipelines console.

### 1.4 Check the results

Each step's artifacts are stored under `gs://<GCP_BUCKET_NAME>/pipeline_root/`. You can find the exact URI of each one in the Vertex Pipelines console.

- **Metrics**: the training losses appear on the metrics artifact of the fine-tuning step. Compare the metrics artifacts of `evaluation-fine-tuned` and `evaluation-base` to see what the fine-tuning changed.
- **Training curves**: download the TensorBoard logs and open them locally:
  ```bash
  gcloud storage cp -r "gs://<bucket>/pipeline_root/.../fine-tuning-component/metrics" logs/
  uv run tensorboard --logdir logs/
  ```
- **Predictions and scores**: download the `predictions` and `evaluation_results` CSVs of both models and read the answers side by side.
- **Model**: [notebooks/inference.ipynb](notebooks/inference.ipynb) downloads the `model` artifact and generates a few predictions locally.

## Part 2: serving the model

### 2.1 Register the model

In the Vertex Pipelines console, click the `model` output of `fine-tuning-component` in a successful run and copy its URI. Then:

```bash
uv run python -m scripts.register_model_with_custom_handler \
    --model-uri gs://<bucket>/pipeline_root/<project_number>/<run_id>/fine-tuning-component_<id>/model \
    --model-name sustained-phi3
```

The script uploads [src/handler.py](src/handler.py) next to the model files, then registers the model with the pre-built Hugging Face serving container. The container loads Phi-3 plus the LoRA adapters and calls the handler for each request. Add `--parent-model <model_id>` to register a new version of an existing model.

### 2.2 Deploy it to an endpoint

In the Vertex AI Model Registry, open the model's menu and select **Deploy to endpoint**. Use one `n1-standard-8` node with one `NVIDIA_TESLA_T4`. Deployment takes 15 to 30 minutes.

Then set `GCP_PROJECT_NUMBER` and `GCP_ENDPOINT_ID` in `.env`. The project number is in the GCP project settings, and the endpoint ID is on the endpoint's page.

> **Cost:** a deployed endpoint is billed every hour it exists, even when idle. Undeploy the model and delete the endpoint when you're done.

### 2.3 Test the endpoint

The endpoint takes prompts that already have the Phi-3 chat template applied, plus generation parameters. It returns the full generated text, prompt included. See [scripts/sample_request.json](scripts/sample_request.json):

```bash
source .env
curl -X POST \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "Content-Type: application/json" \
    "https://${GCP_REGION}-aiplatform.googleapis.com/v1/projects/${GCP_PROJECT_NUMBER}/locations/${GCP_REGION}/endpoints/${GCP_ENDPOINT_ID}:predict" \
    -d "@${INPUT_DATA_FILE}"
```

An answer is about 150 tokens long and takes around ten seconds on a T4. Send one question per request to stay well under the 60-second limit of Vertex AI online predictions.

## Part 3: chat interface

```bash
uv run chainlit run src/app/main.py
```

In the app at http://localhost:8000, the learner chooses a level and a business in the chat settings, then asks a question. The app builds the same sentence as in the training data (`I run a small coffee shop. <question> Explain for a beginner.`), applies the chat template, calls the endpoint with your `gcloud` credentials and shows the four parts. It uses greedy decoding and up to 256 new tokens, the same settings as the pipeline's evaluation.

## Part 4: monitoring with Langfuse

### 4.1 Run Langfuse locally

Langfuse runs with Docker Compose from a folder outside this repository, following the [self-hosting guide](https://langfuse.com/self-hosting/deployment/docker-compose):

```bash
mkdir -p ~/langfuse && cd ~/langfuse
curl -fLO https://raw.githubusercontent.com/langfuse/langfuse/main/docker-compose.yml
docker compose up -d langfuse-web langfuse-worker
```

Starting only `langfuse-web` and `langfuse-worker` brings up the services they depend on and leaves out the optional AI gateway, which saves memory. Langfuse needs about 3 GB of Docker memory, so stop other Compose stacks first.

The UI is at http://localhost:3000. Create an organisation, a project and an API key (**Settings → API Keys**), then add them to `.env`:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

Use the Langfuse Python SDK v4 (`langfuse>=4.18,<5`), which matches the v4 server.

### 4.2 What the app logs

Each message sent in the Chainlit app creates one trace:

| Observation | Input | Output | Also logged |
|---|---|---|---|
| `ask-sustained` (span) | the full question sent to the model: business, question and level | the answer shown to the user | learner level, business and raw question as metadata; Chainlit session ID, so a conversation's messages are grouped in **Sessions** |
| `vertex-endpoint-predict` (generation, inside the span) | the request sent to the endpoint: prompt with the chat template, plus generation parameters | the model's raw output, special tokens included | model name and version, prompt and output token counts, endpoint and deployed model IDs, latency, `ERROR` level if the call fails |

Chainlit runs all the messages of a chat inside a single OpenTelemetry context, so without a fix the whole chat would end up in one trace. `ask_sustained_in_new_trace()` starts each message in an empty context. Each message is its own trace, and the session ID groups the chat.

A failed endpoint call is recorded too: the span and the generation are marked `ERROR`, with the HTTP error as their status message, and the user sees the error in the chat.

### 4.3 Teaching-quality judge (LLM-as-a-judge)

1. **LLM connection:** in **Settings → LLM Connections**, add a Gemini API key (from [Google AI Studio](https://aistudio.google.com/apikey)) or an OpenAI key.
2. **Default model:** in **LLM-as-a-Judge**, set the default evaluation model at the top right to a model from that connection.
3. **Custom evaluator:** select **Set up evaluator → Create custom evaluator**, name it `teaching-quality`, and paste this prompt:

   ```text
   You are a sustainability expert who reviews the answers of a tutor for employees and
   small-business owners. Rate the RESPONSE to the LEARNER MESSAGE.

   The learner message gives a business, a question and a level (beginner, intermediate or
   advanced). A good response:
   - Has four labelled parts in this order: Definition, Example, Action, Check.
   - Is factually correct and consistent with the GHG Protocol and common sustainability
     standards. It invents no statistics, dates or legal obligations.
   - Gives an example and an action that fit the learner's business.
   - Matches the level: plain words for a beginner, precise technical terms for an advanced
     learner.
   - Ends with a Check question that the Definition and the Example allow the learner to answer.

   Examples:
   LEARNER MESSAGE: I run a small coffee shop. What are Scope 3 emissions? Explain for a beginner.
   RESPONSE: Definition: Scope 3 emissions are the indirect emissions in your value chain that
   are not covered by Scope 2. Example: For your coffee shop, they include growing the coffee
   beans and making the cups you buy. Action: List your five biggest purchases. Check: Is the
   electricity you buy for the shop normally Scope 2 or Scope 3?
   SCORE: 1.0 (four parts, correct, fits a coffee shop, plain words)

   LEARNER MESSAGE: I run a small coffee shop. What are Scope 3 emissions? Explain for a beginner.
   RESPONSE: Definition: Scope 3 emissions come from the electricity you buy. Example: The
   lights of your coffee shop. Action: Switch to LED bulbs. Check: What is Scope 3?
   SCORE: 0.3 (four parts, but the definition describes Scope 2, so the content is wrong)

   LEARNER MESSAGE: I run a small furniture workshop. What is the circular economy? Explain for
   an advanced learner.
   RESPONSE: The circular economy is an economic model that keeps materials in use.
   SCORE: 0.2 (correct but no example, action or check, and not at an advanced level)

   LEARNER MESSAGE: {{input}}
   RESPONSE: {{output}}
   ```

   - **Score:** a number between 0 and 1, where 1 means correct, complete, fitted to the business and at the right level, 0.5 means correct but with a part missing or a poor fit, and 0 means wrong or unrelated.
   - **Reasoning:** one or two sentences on the accuracy, the four parts and the fit to the learner.
4. **Run it on the `ask-sustained` observations:** back in **LLM-as-a-Judge**, select **Set up evaluator** and choose `teaching-quality`. Langfuse 4 runs evaluators on observations, so add a filter on the observation name `ask-sustained`. Map `{{input}}` to the observation's input and `{{output}}` to its output. Enable it on new and existing observations, so messages recorded while the endpoint was deployed get scored too.

Scores appear on each `ask-sustained` observation, in its trace and on the **Scores** page.

### 4.4 Dashboard

In **Dashboards → New dashboard**, add these widgets:

- **Teaching quality over time:** average of the `teaching-quality` score, by day.
- **Endpoint latency:** p50 and p95 latency of the `vertex-endpoint-predict` observations.
- **Traffic:** number of `ask-sustained` traces over time.
- **Tokens:** output tokens per generation.

## Local exploration

The notebooks in [notebooks/](notebooks/) are the local prototypes of each step:

- [fine_tuning.ipynb](notebooks/fine_tuning.ipynb): builds the LoRA and SFT configurations, loads the model and launches a short training run.
- [inference.ipynb](notebooks/inference.ipynb): builds prompts with the chat template, generates responses and extracts the answer.
- [evaluation.ipynb](notebooks/evaluation.ipynb): computes the ragas metrics and the format adherence on a predictions CSV.

`bitsandbytes` doesn't install on macOS, so on a Mac the fine-tuning notebook runs without 4-bit quantization. Running the full model locally needs around 8 GB of memory.

## Repository structure

```
├── src/
│   ├── app/
│   │   └── main.py                           # Chainlit chat app calling the endpoint
│   ├── constants.py                          # config loaded from .env
│   ├── taxonomy.py                           # themes, levels, businesses and answer format
│   ├── handler.py                            # EndpointHandler for the serving container
│   ├── pipeline_components/
│   │   ├── data_transformation_component.py  # clean, format and split the dataset
│   │   ├── fine_tuning_component.py          # LoRA fine-tuning (GPU)
│   │   ├── inference_component.py            # predictions of the fine-tuned and base models (GPU)
│   │   └── evaluation_component.py           # ragas metrics and format checks
│   └── pipelines/
│       └── model_training_pipeline.py        # wires the steps together
├── scripts/
│   ├── download_hf_dataset.py                # downloads and filters the Hugging Face dataset
│   ├── build_dataset.py                      # drafts the training dataset with Gemini
│   ├── process_data_locally.py               # runs the data step without Vertex AI
│   ├── pipeline_runner.py                    # compiles and submits the pipeline
│   ├── register_model_with_custom_handler.py # uploads the handler and registers the model
│   └── sample_request.json                   # example request for the endpoint
├── notebooks/                                # local prototypes of each step
└── tests/                                    # unit tests for the data step, the evaluation and the taxonomy
```
