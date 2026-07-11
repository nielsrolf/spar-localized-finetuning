# Engineering Guidelines

## Temporary Scripts

- Put one-off or highly scenario-specific temporary scripts in a `tmp/` folder under the most relevant project area instead of leaving them in the main repo tree.
- The `tmp/` folder does not need to be a single master repo-level folder; prefer local scratch folders like `scripts/layerfreeze/eval/tmp/` when that is where the work belongs.
- Use this for scripts that are unlikely to be reused, such as quick data checks, migration helpers, scratch analysis, or narrow repair utilities.
- Prefer keeping reusable project scripts in their appropriate package or `scripts/` location with clear names and documentation.

## Structured Data Models

- When creating or reshaping structured Python records, prefer explicit datamodel classes over inline hard-coded JSON/dict blocks.
- Use small dataclasses or existing project model patterns to name fields, document the record shape, and keep serialization in methods like `to_jsonl_record()`, `to_csv_row()`, or API-specific payload builders.
- Keep raw dict literals acceptable for tiny local log events or one-off glue, but introduce a model when the structure crosses function boundaries, is written to disk, is sent to an API, or is reused in multiple places.

## Entrypoint Orchestration

- Keep `main()` and other entrypoint functions focused on orchestration: load inputs, call clearly named stages in order, and handle top-level setup/teardown.
- Move implementation details such as serialization, upload/download mechanics, logging payload construction, data shaping, and per-stage bookkeeping into helper functions with descriptive names.
- A reader should be able to skim `main()` and understand the workflow without parsing low-level JSON, CSV, API, or logging details.
