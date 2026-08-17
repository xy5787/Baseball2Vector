# LM API integration guide

Phase 5 deliberately makes no network request. It produces versioned prompts,
tool definitions, request envelopes, and validated Prediction objects. Add a
provider only in a later experiment phase so API behavior cannot silently
change the benchmark definition.

## 1. Generate and audit local requests

```bash
cd LM_experiments
python scripts/05_lm_arm_stub.py --arm b \
  --record-id "Juan Soto__2021_2022" \
  --output outputs/requests/soto_b_v1.json
```

Repeat with `--arm c` and `--arm d`. Arms B and D must contain no player name,
team, or calendar season. The script fails closed if name or season leaks.
The envelope's `record_id` is local metadata: never send it to the model for B
or D.

## 2. Pin the experiment before spending money

Record the provider, exact model ID, model/version date, prompt version,
tool-schema version, temperature or reasoning setting, maximum output tokens,
retry policy, and run timestamp. Use the same model and generation settings for
B, C, and D. Run each record in a fresh conversation and do not share tool
results between records.

Store API keys only in the environment or a secret manager. Do not put keys in
`config.yaml`, request JSON, logs, or shell history. Use the environment-variable
name recommended by the selected provider SDK.

## 3. Implement one provider adapter

Implement `LMProviderAdapter.predict(request)` from
`src/lm_experiments/lm_arms/client.py` in a new provider-specific module.
The adapter should:

1. Send only `request["prompt"]` and converted `request["tools"]`.
2. Convert the repository's provider-neutral `{name, description, input_schema}`
   tools into the selected API's function/tool representation.
3. Force or require a final `submit_prediction` tool call for Arms B and C.
4. For Arm D, execute only `ToolBox.dispatch()` calls locally, return each tool
   result to the same model turn, and stop after a configured iteration cap.
5. Never expose `record_id`, ground-truth deltas, `player_name`, or season to B/D.
6. Pass the final tool payload to `validate_prediction(payload, record_id)`.

Minimal adapter shape:

```python
class ProviderAdapter:
    def predict(self, request: dict) -> dict:
        provider_tools = convert_tools(request["tools"])
        raw_payload = provider_tool_loop(
            prompt=request["prompt"],
            tools=provider_tools,
            local_dispatch=toolbox.dispatch if request["arm"] == "d" else None,
        )
        return validate_prediction(raw_payload, request["record_id"])
```

`provider_tool_loop` and `convert_tools` are intentionally not implemented in
Phase 5; their exact code depends on the chosen provider SDK.

## 4. Prevent test leakage

`ToolBox` fits regression models and comparable pools on the training windows
only. `get_player_vector` is restricted to 2021–2024 for the current 2024→2025
holdout, so the held-out 2025 vector cannot be queried. Preserve that boundary
in any adapter. Do not add web search or external baseball databases to Arm D
without defining a new arm, because that changes the measured treatment.

## 5. Persist resumably

Use one output row per `(record_id, arm, provider, model, prompt_version)` and
reject duplicate keys. Write each validated response immediately to a
checkpoint file so a failed batch can resume. Keep raw provider response IDs,
latency, token usage, and error metadata in a separate audit log; keep the
scoring Prediction table limited to the Prediction v1 fields plus run metadata.

Start with 5 records per arm, inspect masking and schema validity, then run a
stratified 200–300-record subset. Estimate cost from the pilot's actual token
usage before launching a full batch.

## 6. Score without provider-specific code

```python
import pandas as pd
from lm_experiments.scoring import score

ground_truth = pd.read_csv("outputs/tables/transitions_stratified.csv")
predictions = pd.read_csv("outputs/tables/lm_predictions.csv")
report = score(predictions, ground_truth, reference_comparables=knn_reference)
```

Always inspect `by_stratum` and `by_window`. A named-arm advantage confined to
older transitions is a contamination or memorization warning, not automatically
better vector reasoning.
