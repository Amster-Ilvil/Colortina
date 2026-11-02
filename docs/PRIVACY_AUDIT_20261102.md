# Colortina privacy audit — 2026-11-02

## Outcome

No recognizable API keys, passwords, private keys, personal user-home paths, personal project files or model weights were confirmed in the accessible source and Git history. The scanned Windows v5.20.0 release ZIP also passed.

This is a bounded verification, not a claim of zero risk: legacy encoded payloads cannot all be reconstructed from the available archived fragments. The old v5.13.27 binaries and all macOS/Linux release archives were not independently extracted and content-scanned in this audit.

## Scope and evidence

- Current main: 145 source/text/config files checked against common credential patterns, user-home paths and credential assignments. No confirmed private data. Third-party copyright email addresses are upstream notices.
- Complete fetched Git refs: GitHub Actions scanned 485 reachable Git blobs. No positive detections in available plaintext blobs after resolving historical README path placeholder false positives.
- Windows v5.20.0 ZIP: 33,037 archive entries inventoried, 160 first-party sources inspected. No flagged secrets, personal project assets or bundled model checkpoints.
- Two initial PEM filename matches were publicly distributed certifi CA bundles in Python dependencies and were explicitly allowlisted.
- Commit author email records use GitHub-generated noreply addresses.
- v5.20.0 and v5.13.27 release attachment names and sizes were reviewed.

Verification workflow: https://github.com/Amster-Ilvil/Colortina/actions/runs/41455752024

Windows release: https://github.com/Amster-Ilvil/Colortina/releases/tag/v5.20.0

## Historical caveat

Earlier source-sync commits retained 41 Base64 (.b64) fragments in Git history, including Base64/XZ source transport data. Three attempted archive reconstructions failed. Therefore the complete encoded interiors could not be verified. Historical Git objects remain reachable in the public repository even after working-tree removal.

Seven historical README path detections were confirmed to be literal Windows examples using generic Chinese placeholders, not personal Windows account paths.

Do not erase shared Git history or tags blindly. If a real credential is ever found, revoke/rotate it first, then coordinate any git-filter-repo rewrite and downstream cache/fork cleanup.

## Preventive changes

- .gitignore now blocks user project formats, imported/output media, archives, Base64 payloads, model checkpoints and local runtime folders; only explicitly named public branding assets are exempt.
- scripts/privacy_audit.py blocks those file categories even if someone force-adds a Git-ignored file; scans its own source as well.
- scripts/audit_git_history.py scans all reachable blobs and attempts reconstruction of historical archives. CI outputs only category counts and source locations, not secret values.
- .github/workflows/full-privacy-audit.yml provides repeatable source/history and Windows ZIP privacy checks.

## Coverage limits / recommended owner checks

- The repository Security > Secret scanning alert list and push-protection settings were not accessible through the connected API. Review the GitHub repository Security panel; GitHub may detect token types outside this audit's patterns.
- Historical v5.13.27 release binaries and macOS/Linux distribution binaries were not independently unpacked and scanned. Their names/metadata and packaging workflows were reviewed.
- Third-party model files downloaded by end users and source-sync archives hosted on third-party file-sharing sites are outside this repository audit.
- Static scanning cannot prove the complete absence of personal information in binary data or content not covered by patterns.

No password, key or token values are included in this report.
