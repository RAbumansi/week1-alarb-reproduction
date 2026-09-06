# Safe agentic direction for the later project

The parent paper supports an Arabic legal-reasoning project. It does not contain criminal-record data and cannot verify whether a named person has a criminal record.

## Recommended research scope

Build an auditable Arabic legal-case reasoning agent over anonymized cases:

1. A case-intake component validates that the input is an anonymized research case.
2. A retrieval component finds potentially applicable statutory articles.
3. A reasoning component produces a structured, citation-grounded analysis.
4. A verdict component predicts an outcome for benchmark evaluation only.
5. A verification component checks every cited article and flags unsupported conclusions.
6. A human-review gate approves any output before it leaves the research environment.

## If the topic must remain criminal-record verification

The agent should only coordinate consent-based checks against authorized official sources. It must not infer criminal history from names, social media, commercial-case text, nationality, or demographic attributes. Required controls include identity disambiguation, purpose limitation, access logs, data retention limits, an appeal/correction process, and mandatory human review. This would require a different parent paper and a lawful dataset.

