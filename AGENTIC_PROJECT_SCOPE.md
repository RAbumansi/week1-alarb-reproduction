# Safe agentic direction for the later project

The parent paper supports an Arabic legal-reasoning project. It does not contain criminal-record data and cannot verify whether a named person has a criminal record.

## Recommended research scope

Build an auditable Arabic legal-case reasoning agent over research cases, with the residual-identification limitations below:

1. A case-intake component validates that the input is an anonymized research case.
2. A retrieval component finds potentially applicable statutory articles.
3. A reasoning component produces a structured, citation-grounded analysis.
4. A verdict component predicts an outcome for benchmark evaluation only.
5. A verification component checks every cited article and flags unsupported conclusions.
6. A human-review gate approves any output before it leaves the research environment.

## Implemented prototype as of October 25 2026

`alarb/agentic.py` now implements structured fact intake, training-only retrieval,
model drafting, citation/schema verification, bounded correction and an explicit
human-review flag. Its hybrid retriever combines article-text TF-IDF with cited
provisions from similar training cases. It fits only on the 9,339 training cases,
not the development test or final validation cases. Held-out case verdicts,
reasoning and legal references are not passed to the drafting component.

A 50-case offline fixture run exercises the orchestration and evaluates
retrieval separately. The fixture deliberately abstains and is not evidence of
legal reasoning quality. A real-model experiment and human evaluation remain
necessary. Citation checks confirm that an article was retrieved; they do not
verify legal entailment or guarantee correct verdicts. Outputs always require
human review, and the prototype does not automatically publish or approve them.

## The published data still contains personal names

Section 3.1 of the paper states that the facts and reasoning "often contained
identifiable information about plaintiffs and defendants, which needed to be
removed", and describes an LLM anonymization step verified on random samples.

That step did not catch everything. While reviewing model output we found a
case whose `case_facts` and `court_reasoning` both name a natural person, and
the model repeated the name in its predicted verdict. Probing the 12,012
development cases for a short list of common Saudi given names as whole words
matches **2,223 cases (18.5%)**.

Read that as an upper bound, not a count of exposed individuals. Saudi
commercial entities are routinely named after their founders, so many hits are
company names rather than private parties — identifying, but differently so.
The confirmed case above is enough to establish that residual identifiers exist
in the release.

Consequences for this project:

- Do not treat the dataset as reliably anonymized, and do not quote case text
  verbatim in write-ups or slides.
- Model outputs can echo these names, so run artifacts under `runs/` inherit
  the problem. They are gitignored; keep them that way.
- A measured first step would be a proper named-entity sweep rather than the
  keyword probe above, which is deliberately crude.
- This is worth reporting upstream to the dataset authors.

## If the topic must remain criminal-record verification

The agent should only coordinate consent-based checks against authorized official sources. It must not infer criminal history from names, social media, commercial-case text, nationality, or demographic attributes. Required controls include identity disambiguation, purpose limitation, access logs, data retention limits, an appeal/correction process, and mandatory human review. This would require a different parent paper and a lawful dataset.

