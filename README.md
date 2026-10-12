# Tourism purchase prediction

- Live app: https://pramod-tourism-prediction.streamlit.app/
- Successful pipeline run: https://github.com/pramod-collab/tourism-purchase-prediction/actions/runs/38189076630
- Deployment and a live prediction were checked on 12 October 2026.

This course project predicts past package purchases using details available before a sales call.
The data does not separately identify Wellness package purchases. Applying it to the new
Wellness package is an assumption that needs a small campaign pilot.

## Files

- `Tourism_Project.ipynb`: editable notebook with observations after every code cell.
- `Tourism_Project.html`: executed notebook for submission.
- `tourism_project/data/tourism.csv`: supplied data.
- `tourism_project/model_building/`: data validation, preparation and model training.
- `tourism_project/deployment/`: Streamlit app, model, metadata and app dependencies.
- `tourism_project/reports/`: experiment comparison and evaluation results.
- `.github/workflows/pipeline.yml`: three dependent GitHub Actions jobs.
- `tests/`: model, split and app checks.

## Run locally

Use Python 3.13 to match the tested environment. Run commands from this repository folder.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r tourism_project/requirements.txt
python -m tourism_project.model_building.data_register
python -m tourism_project.model_building.prep
python -m tourism_project.model_building.train
python -m pytest tests -q
python -m streamlit run tourism_project/deployment/app.py
```

Open the notebook from this folder and run its cells in order. The notebook also accepts
`tourism.csv` beside it when starting in a new folder. It writes the project source files
and runs the same scripts as the workflow. Training is reproducible with seed 42, though
small platform differences are possible.

MLflow logs each parameter combination and its cross-validation results in `mlruns`.
The notebook reads those records directly. For an optional visual tracking UI, install
`mlflow==3.0.1` in the same environment and run `mlflow ui --backend-store-uri ./mlruns`.
No ngrok account or public tracking server is needed.

## Publish on GitHub and Streamlit

1. Create a public GitHub repository named `tourism-purchase-prediction`.
2. Upload the repository contents, including the hidden `.github` folder. Do not upload
   virtual environments, secrets, or temporary files. GitHub Desktop is a simple way to
   publish the folder while preserving all paths.
3. Keep `main` as the default branch. The workflow needs permission to write repository
   contents. Organization rules or protected branches may prevent the final bot push;
   resolve that repository setting before claiming successful automation.
4. In Actions, run **Tourism Project Pipeline** and confirm all three jobs pass. Its
   training job downloads the split artifact, tunes models and commits the model to main.
   A newer source revision makes the revision check fail rather than mixing old splits
   with new code. Run the workflow again on the latest main in that case.
5. Sign in at https://share.streamlit.io and choose Create app. Select this repository,
   branch `main`, and `tourism_project/deployment/app.py`. In advanced settings select
   Python 3.13. Deploy and verify that a prediction works. Make the app public.
6. Capture the GitHub folder structure, successful Actions run and deployed app with
   its public address visible. Save these real screenshots in `evidence/`.
7. Fill `deployment_evidence.json` with the real URLs and screenshot paths, rerun the
   notebook evidence cell and export the notebook to HTML again.

```bash
python -m nbconvert --to html --embed-images Tourism_Project.ipynb
```

The public deployment and GitHub run are verified. The notebook includes the real
links and screenshots. The steps above explain how to reproduce the deployment.

## Method and limits

The model excludes customer IDs, exported row numbers and current contact/pitch fields.
It assumes the remaining profile and travel preferences are already held in the customer
database. Identical cleaned profiles stay together in the holdout and CV splits.
Three model families are compared using average precision. The contact threshold is
chosen using training-fold F2 scores, because missing buyers is the main concern.
No contact costs or campaign capacity were supplied, so this threshold is provisional.

The holdout results estimate performance on similar historical customers. They do not
prove future campaign profit, fairness or performance for the new Wellness package.
Review gender and marital-status use before real deployment; compare a model without
them and check errors across groups. Revisit the model using new outcomes, record data
versions and review major changes before a real campaign.

## References

- Course Business Context, Rubric and Learner Template supplied with this assignment.
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
- https://docs.github.com/en/actions/tutorials/store-and-share-data
- https://docs.github.com/en/actions/concepts/security/github_token
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://mlflow.org/docs/3.0.1/tracking.html
