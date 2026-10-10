"""Prompt templates for the ALARB tasks.

The four templates marked PAPER are transcribed from the appendices of
arXiv:2510.00694 and are reproduced as printed, including the unusual closing
tags ([\\REASONING], [\\VERDICT]) and the inconsistent "Begin"/"Begin!" endings.
Templates marked DERIVED have no published counterpart; the paper describes
those tasks in prose only. Every departure is listed in DEVIATIONS.md.
"""

from __future__ import annotations


# PAPER -- Appendix B.1, Figure 7.
JUDGE = r"""You are a legal assistant. You will be given a judge's verdict from a legal case in Saudi Arabia, and a prediction of the verdict from another legal assistant.
Your task is to evaluate how well the prediction matches the judge's verdict.
The evaluation should be based on the content of the verdicts and how well they align with each other.
A prediction is correct if it is similar to the judge's verdict and captures the essence of the decision. It does not have to be identical, but it should reflect the same outcome and reasoning.
It's acceptable for the prediction to be shorter or more concise than the judge's verdict, or the other way around, as long as the core message is the same. Ignore any noise or irrelevant tokens in the verdicts.
Before you output your evaluation, think about how well the prediction matches the judge's verdict.
Output one of the following for the evaluation:
- "CORRECT" if the prediction matches the judge's verdict.
- "INCORRECT" if the prediction does not match the judge's verdict.
- "PARTIALLY CORRECT" if the prediction is partially correct but does not fully match the judge's verdict.
Follow this format:
[THINK]
"Your reasoning here"
[EVALUATION]
"Evaluation here (CORRECT, INCORRECT, or PARTIALLY CORRECT)"

Judge's verdict:
{judge_verdict}

Predicted verdict:
{predicted_verdict}

Begin!"""


# PAPER -- Appendix B.2, Figure 8. Task 1: verdict from facts alone.
VERDICT_FROM_FACTS = r"""You are a legal assistant specialized in Saudi Arabian law. Your task is to predict the verdict of a legal case from Saudi Arabia.
The cases involve trade and finance and commercial laws.
You will be given a set of facts from the case, and you MUST provide BOTH:
1. A reasoning section analyzing the facts
2. A verdict prediction section stating what you think the court will decide
The verdict should be based only on the facts provided without personal opinions or biases.
Think carefully about the facts and how they relate to the laws in Saudi Arabia.
Your verdict and reasoning should be strictly in {language}
The verdict should be short and direct.
Follow the format below:
[REASONING]
"Your reasoning and analysis here"
[\REASONING]
[VERDICT]
"Your verdict here"
[\VERDICT]
Do not output anything else outside these two sections.
Here are the facts of the case:
{case_facts}
Begin!"""


# PAPER -- Appendix B.2, Figure 10. Task 2: verdict from facts and applicable articles.
# Figure 10 prints "strictly in language" without braces, unlike Figures 8 and 9.
# Read as a typesetting slip and restored to {language}; see DEVIATIONS.md.
VERDICT_FROM_FACTS_AND_LAWS = r"""You are a legal assistant. Your task is to predict the verdict of a legal case from Saudi Arabia.
The cases involve trade and finance and commercial laws.
You will be given a set of facts from the case, and the laws and regulations applicable to this case, and you MUST provide BOTH:
1. A reasoning section analyzing the facts
2. A verdict prediction section stating what you think the court will decide
You should provide a verdict based on the facts and the given laws.
The verdict is a sentence that summarizes the outcome of the case showing what do you think the court will decide.
The verdict should be based on the facts and laws provided and should not include any personal opinions or biases.
Your verdict and reasoning should be strictly in {language}.
Think about the case facts and how they relate to the given laws.
Follow the format below:
[REASONING]
"Your reasoning and analysis here"
[\REASONING]
[VERDICT]
"Your verdict here"
[\VERDICT]
Do not output anything else.
Here are the facts of the case:
{case_facts}
Here are the laws related to this case:
{case_laws}
Begin!"""


# PAPER -- Appendix B.2, Figure 9. Task 3: verdict from facts and the court's reasoning.
# This one asks for a verdict only, with no reasoning section, and ends "Begin".
VERDICT_FROM_FACTS_AND_REASONING = r"""You are a legal assistant. Your task is to predict the verdict of a legal case from Saudi Arabia.
The cases involve trade and finance and commercial laws.
You will be given a set of facts from the case, and the reasoning of court on these facts.
You should provide a verdict based on the facts and the reasoning of the court.
The verdict is a sentence that summarizes the outcome of the case showing what do you think the court will decide.
The verdict should be based on the facts and reasoning provided and should not include any personal opinions or biases.
Your verdict should be strictly in {language}.
Your output should only be a direct and short verdict, do not output anything else.
Make sure to label the start and end of the verdict properly.
Follow the format below:
[VERDICT]
"Your verdict here"
[\VERDICT]
Do not output anything else.
Here are the facts of the case:
{case_facts}
Here is the reasoning of the court:
{case_reasoning}
Begin"""


# DERIVED -- Task 4, argument completion. Section 4.1 describes it as the
# intermediate between Figures 8 and 10: the model receives the opening steps of
# the court's reasoning and must finish the chain and reach a verdict. No prompt
# was published, so this follows Figure 9's wording as closely as the changed
# task allows.
ARGUMENT_COMPLETION = r"""You are a legal assistant. Your task is to predict the verdict of a legal case from Saudi Arabia.
The cases involve trade and finance and commercial laws.
You will be given a set of facts from the case, and the first steps of the reasoning of the court on these facts.
The reasoning of the court is incomplete. You MUST provide BOTH:
1. A reasoning section that completes the remaining steps of the court's reasoning
2. A verdict prediction section stating what you think the court will decide
The verdict should be based on the facts and the reasoning provided and should not include any personal opinions or biases.
Your verdict and reasoning should be strictly in {language}.
Follow the format below:
[REASONING]
"Your reasoning and analysis here"
[\REASONING]
[VERDICT]
"Your verdict here"
[\VERDICT]
Do not output anything else.
Here are the facts of the case:
{case_facts}
Here are the first steps of the reasoning of the court:
{case_reasoning}
Begin!"""


# DERIVED -- Article identification. Section 4.2 describes the task and Figure 11
# shows a rendered item, but no prompt was published.
ARTICLE_MCQ = r"""You are a legal assistant specialized in Saudi Arabian law.
You will be given the facts of a commercial court case from Saudi Arabia, and four articles from Saudi statutes and regulations.
Exactly one of the four articles is the one cited by the court in its reasoning on this case.
Your task is to choose the most applicable article based on the facts of the case.
Answer with a single letter: A, B, C, or D.
Follow the format below:
[ANSWER]
"A, B, C, or D"
[\ANSWER]
Do not output anything else.
Here are the facts of the case:
{case_facts}
Here are the four articles:
{choices}
Begin!"""
