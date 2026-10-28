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

## Verification still needed
Offline tests cover the decision policy and card agreement. Test the full deployed API with real artifacts and provider credentials before merging. Check document/image handling, provider failures and attribution rendering. Retrieval uses snippets and may miss or repeat misinformation. Model output requires manual source review.

MIT licensed educational application.
