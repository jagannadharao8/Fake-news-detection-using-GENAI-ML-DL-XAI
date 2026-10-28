# Misinformation evidence assistant
Portfolio project by Jalla Jagannadharao using FastAPI, a local text model, search snippets, cloud model integrations and text attribution.

## Decision policy
The API and evidence card use the same policy in src/decision.py. Missing, mixed, weak evidence or an available image/text mismatch returns Uncertain. Strong retrieved support or contradiction can produce Real/Fake. These are uncalibrated model-assisted assessments, not guarantees of truth. Local linguistic probabilities alone do not establish factuality. Text attributions explain the local text classifier, not the evidence policy.

Legacy XGBoost fusion modules remain for experiments; they do not select the live API's top-level verdict. No current benchmark score or automatic image-manipulation detection is claimed.

## Run
Python 3.11:
```sh
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8000
python -m unittest discover -s tests -v
```
Model artifacts are downloaded from the existing Hugging Face Space during setup. Configure GROQ_API_KEY, GEMINI_API_KEY and SERPER_API_KEY privately. Model and account availability must be checked on the deployment.

## Supported inputs and privacy
Paste text or upload an image/PDF up to 10 MB. URL fetching is disabled; paste article text instead. Shared public history is disabled and new submissions are not persisted by this API. Existing history.db files are not deleted by this change. Delete historical data separately through the hosting service if required.

## Verification and limits
[Live demonstration](https://jagannadharao8-fake-news-detection.hf.space/).
On 28 October 2026, the deployed API returned HTTP 200 with matching Uncertain verdict and evidence-card label for a synthetic unsupported claim, with no confidence percentage. Empty input, URL fetching and a malformed PDF returned HTTP 400; public history returned HTTP 410. [Live check evidence](https://github.com/jagannadharao8/jagannadharao8.github.io/actions/runs/40711305705) (its separate research-app check failed; the fake-news checks passed).

Fifteen offline policy/API tests cover decision consistency and mocked provider failures. A text-path smoke is not full multimodal validation: image interpretation, provider-specific failures and attribution quality still need review. Retrieval uses snippets and may miss or repeat misinformation. Model output requires manual source review.

## Deployment
GitHub Actions uploads the runtime server, src and static files to the existing Hugging Face Space. It adds the Docker SDK metadata to the remote README and preserves remote model artifacts and private settings. The root app.py, browser extension and fusion modules are legacy experiments; server.py is the deployed application entry point.

MIT licensed educational application.
