# Security, Privacy, and Data Governance

## 1. Processing model

The standalone Surgiplot package performs metric computation locally. The tracked core contains no telemetry client, cloud uploader, account system, segmentation-weight downloader, or remote inference service.

Installing dependencies contacts the configured package index, and cloning or updating contacts the configured Git remote. Those installation operations are separate from analysis-data processing.

## 2. Data classification

Coordinate files, screenshots, rendered scenes, navigation exports, and filenames can contain protected or identifying information even when the numerical table appears minimal. Treat as sensitive:

- names, dates, medical-record identifiers, accession numbers, or device IDs;
- facial or cranial surface geometry;
- DICOM headers and embedded metadata;
- original filesystem paths containing patient or operator names;
- screenshots containing application title bars or clinical annotations;
- free-text notes and provenance fields;
- source-control history, not only the current working tree.

## 3. Repository policy

This software repository is for source, documentation, synthetic tests, and non-identifying assets. Do not commit clinical datasets, derived patient geometry, screenshots from clinical cases, model checkpoints, virtual environments, caches, or local workstation configuration.

The root `.gitignore` is a guardrail, not a privacy control. Git can still track a file that was added before an ignore rule, and archives can retain previously committed content.

## 4. Pseudonymization

Use a study-specific coded identifier instead of a direct identifier. Keep the re-identification key in an approved, access-controlled system separate from the software checkout.

Pseudonymization is not anonymization. Detailed geometry and longitudinal combinations may remain linkable. Apply institutional policy, data-use agreements, and applicable law.

## 5. Safe workflow

1. copy only the minimum required data into an approved analysis location;
2. remove direct identifiers and unnecessary metadata;
3. verify coordinate-system and units without retaining clinical filenames in examples;
4. run Surgiplot from an isolated environment;
5. export results to the approved study-data location;
6. do not export inputs or results into the Git repository;
7. inspect staged files before every commit;
8. securely remove temporary data according to institutional policy.

## 6. Pre-commit inspection

Review the exact index, not only the working tree:

```bash
git status --short
git diff --cached --name-status
git diff --cached
git ls-files
```

Search tracked text for workstation paths and identifier vocabulary:

```bash
git grep -n -I -E '/Users/|/home/|[A-Za-z]:\\Users\\'
git grep -n -I -E 'patient|mrn|accession|date_of_birth|dob'
```

These searches produce false positives in policy documentation and cannot establish de-identification. They are triage tools.

Use an approved secret scanner for tokens and credentials. If a credential is found, revoke or rotate it before rewriting history; deleting the text alone does not invalidate the secret.

## 7. File inventory

Audit suspicious extensions and large files:

```bash
git ls-files | grep -E '\.(dcm|nii|nii\.gz|nrrd|mha|mhd|mrb|mrml|xlsx?|csv|tsv|zip)$'
git count-objects -vH
```

Also inspect Git LFS pointers if LFS is in use. Large-file absence does not prove PHI absence; identifiers can exist in tiny JSON, text, or source files.

## 8. History versus current release tree

Removing a file in a new commit removes it from the branch tip but not from earlier commits. If prohibited data or an active secret entered published history:

1. stop further distribution;
2. revoke credentials where applicable;
3. identify every affected ref and clone;
4. obtain repository-owner approval for history rewriting;
5. rewrite and force-update all affected refs;
6. invalidate caches or releases when possible;
7. notify collaborators to re-clone;
8. follow institutional incident-response requirements.

History rewriting is disruptive and must not be performed as routine cleanup without explicit authorization.

## 9. Logs and exceptions

Errors may display input paths or point labels. Before sharing terminal output or opening a public issue:

- replace clinical paths with neutral placeholders;
- remove dataset excerpts and screenshots;
- retain the exception type and a synthetic reproduction;
- verify that copied debug dictionaries do not contain scene metadata.

## 10. Dependency and supply-chain controls

For controlled deployments:

- install in a dedicated environment;
- use trusted package indexes;
- record dependency versions and hashes where required;
- review dependency advisories;
- build from a signed/tagged revision when available;
- avoid running unreviewed notebooks or scripts in the analysis environment;
- separate optional GUI/VTK dependencies from headless compute nodes.

The repository does not currently provide a cryptographically locked dependency set. Capture the resolved environment for each study.

## 11. Threat model boundary

Surgiplot parses local files and therefore treats their contents as untrusted input. Validation limits malformed numerical structures, but the project has not undergone a formal penetration test. Do not expose the CLI or GUI as a public file-processing service without an additional sandbox, upload restrictions, resource limits, and security review.

The software does not provide access control, encryption at rest, secure deletion, audit logging, or a clinical data repository. Those controls belong to the host environment and institutional workflow.

## 12. Reporting a security or privacy issue

Do not place sensitive evidence in a public issue. Contact the repository owner privately with a minimal description, affected version or commit, and a sanitized reproduction. If patient data may be involved, use the applicable institutional incident channel before sending files.

## 13. Release privacy checklist

- [ ] staged file list reviewed;
- [ ] no clinical or participant files;
- [ ] no absolute workstation paths;
- [ ] no caches, `.DS_Store`, virtual environments, or checkpoints;
- [ ] no embedded secrets or private keys;
- [ ] no identifying screenshots or document metadata;
- [ ] package archive contents inspected;
- [ ] current Git history implications understood;
- [ ] release notes contain no identifiers.
