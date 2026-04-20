# TODO

1. Improve LLM system and user prompts to include definition of each metric.

2. List all validation criteria and provide users with command line options to choose and relax rejection/human_review criteria.

3. Add more metrics.

4. LOW_PRIORITY: Current flow is a 3-stage pipeline (`fetch_and_store_data` → `cleanse_data` → `run_findings`) that can be executed as a single workflow. Analyse possibility of a single analysis tool (`fetch_from_pubmed` → `analyse_for_search_term`). **Note:** This requires maintaining a visited article hash to avoid duplicates, and an article ID–findings map to audit ground truths.
