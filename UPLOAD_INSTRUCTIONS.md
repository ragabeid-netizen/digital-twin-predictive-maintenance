# How to publish this package (GitHub + Zenodo DOI)

A Zenodo DOI is the citable, permanent archive that Q1 journals expect. The usual
flow is: push the code to a **GitHub** repo, then connect that repo to **Zenodo**,
which mints a **DOI** for each release.

## 1. Put it on GitHub
1. Create a free account at https://github.com and a new **public** repository, e.g.
   `digital-twin-predictive-maintenance`.
2. Upload this whole folder. Easiest (no git needed): on the new repo page click
   **"uploading an existing file"** and drag in the contents of this folder.
   Or with git:
   ```bash
   cd DigitalTwin_PdM_Reproducibility
   git init && git add . && git commit -m "Initial release: reproducibility package"
   git branch -M main
   git remote add origin https://github.com/<your-user>/<your-repo>.git
   git push -u origin main
   ```

## 2. Get a DOI from Zenodo
1. Sign in at https://zenodo.org with your GitHub account.
2. Go to **Settings → GitHub**, find your repository in the list and toggle it **ON**.
3. Back on GitHub, create a **Release** (Releases → "Draft a new release" →
   tag `v1.0.0` → Publish). Zenodo automatically archives that release and assigns a
   **DOI**.
4. Copy the DOI badge / number Zenodo shows (looks like `10.5281/zenodo.XXXXXXX`).

## 3. Put the links in the paper and in this package
- Paste the GitHub URL and the Zenodo DOI into the paper's
  **"Code and Data Availability"** section (a placeholder is already there).
- Also fill them into `CITATION.cff` (the commented `doi:` and `repository-code:` lines).

That's it — the manuscript now points to a permanent, citable code+data archive,
which directly answers the reviewer's request.
