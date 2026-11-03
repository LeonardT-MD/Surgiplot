# Surgiplot Documentation Index

This index covers the standalone Python, command-line, and desktop Surgiplot software. Documentation for the 3D Slicer extension is maintained in the separate [SurgiplotToolkit repository](https://github.com/LeonardT-MD/SurgiplotToolkit).

## Start here

| Reader | Recommended path |
|---|---|
| New user | [README](../README.md) → [Getting Started](GETTING_STARTED.md) → [CLI Reference](CLI_REFERENCE.md) or [GUI Guide](GUI_GUIDE.md) |
| Python user | [Installation](INSTALLATION.md) → [Data Formats](DATA_FORMATS.md) → [Python API](PYTHON_API.md) |
| Researcher defining a protocol | [Metric Reference](METRICS_REFERENCE.md) → [Validation](VALIDATION.md) → [Publication Guide](SURGIPLOT_PUBLICATION_GUIDE.md) |
| Software reviewer | [Architecture](ARCHITECTURE.md) → [Metric Reference](METRICS_REFERENCE.md) → [Development](DEVELOPMENT.md) |
| Data-governance reviewer | [Security and Privacy](SECURITY_AND_PRIVACY.md) → [Data Formats](DATA_FORMATS.md) |

## Operational documentation

- [Getting Started](GETTING_STARTED.md) — release installation, terminal syntax, a synthetic end-to-end tutorial, all metric families, GUI steps, Python use, expected outputs, and common failures.
- [Installation and Deployment](INSTALLATION.md) — core, GUI, rendered-scene, Conda, platform, upgrade, and uninstall procedures.
- [Data Formats and Coordinate Semantics](DATA_FORMATS.md) — normalized schemas, vendor imports, aliases, coordinate systems, units, and round trips.
- [CLI Reference](CLI_REFERENCE.md) — every command, option family, range syntax, JSON output, and automation pattern.
- [Python API](PYTHON_API.md) — dataset construction, metric calls, result dataclasses, debug geometry, provenance, and examples.
- [Desktop GUI Guide](GUI_GUIDE.md) — dataset curation, rendered-scene acquisition, plotting, and failure recovery.

## Technical and scientific documentation

- [Architecture and Execution Model](ARCHITECTURE.md) — package boundaries, dependency graph, execution paths, and optional components.
- [Metric Reference](METRICS_REFERENCE.md) — exact inputs, formulas, result fields, validation rules, invariants, and limitations.
- [Validation and Quality Assurance](VALIDATION.md) — automated tests, analytical fixtures, tolerances, and release acceptance criteria.
- [Development and Release Guide](DEVELOPMENT.md) — local setup, style, tests, packaging, Git hygiene, and release sequence.
- [Security, Privacy, and Data Governance](SECURITY_AND_PRIVACY.md) — local processing model, PHI exclusions, repository scanning, and incident response.
- [Changelog](../CHANGELOG.md) — release-specific corrections, additions, and compatibility notes.
- [Citation metadata](../CITATION.cff) — software author, version, release date, and repository citation record.

## Methods and publication material

- [Extended Publication Guide](SURGIPLOT_PUBLICATION_GUIDE.md) — end-to-end research workflow and reporting recommendations.
- [Ecosystem Methods Summary](SURGIPLOT_ECOSYSTEM_METHODS_SUMMARY.md) — mathematical and interface summary.
- [Supplementary Methods](SURGIPLOT_SUPPLEMENTARY_METHODS.md) — concise implementation-oriented methods text.
- [Manuscript Methods Draft](SURGIPLOT_MANUSCRIPT_METHODS_DRAFT.md) — editable manuscript-language starting point.
- [Technical Implementation Note](SURGIPLOT_TECHNICAL_IMPLEMENTATION_NOTE.md) — reviewer-oriented implementation summary.

## Source-of-truth hierarchy

When descriptions differ, use this order:

1. tagged source code and automated tests for the installed release;
2. [Metric Reference](METRICS_REFERENCE.md) for the explicit software contract;
3. [Data Formats](DATA_FORMATS.md), [Python API](PYTHON_API.md), and [CLI Reference](CLI_REFERENCE.md) for interface behavior;
4. publication-oriented narrative documents.

Scientific papers provide the methodological context and canonical names of published metrics. They do not automatically imply that every software implementation detail is identical to every historical protocol. Release-specific deviations and simplifications must be stated in study methods.
