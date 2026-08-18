# AGENTS.md

## OnKith project agent instructions

This repository is for OnKith (formerly "Home AI"). Keep all work simple, clear, and easy for the team to understand.

## 1. Plan First, Then Wait for Approval

Before making code changes, always:

1. Read the task carefully.
2. Inspect the repository to understand the current structure and existing work.
3. Check whether the requested feature is already implemented, partially implemented, or missing.
4. Create a short plan explaining:

   * What you found.
   * What files may need to change.
   * What steps you will take.
   * Any assumptions or risks.

After writing the plan, stop and wait for approval before editing files or running major changes.

Accept approval phrases such as:

* `approved`
* `approve`
* `yes`
* `go ahead`
* `continue`

Do not start implementation until approval is given.

## 2. Keep It Simple

Follow Occam’s razor: the simplest correct solution is preferred.

Avoid unnecessary complexity, over-engineering, large abstractions, or complicated designs unless the task clearly requires them.

Prefer:

* Clear code over clever code.
* Small changes over large rewrites.
* Existing patterns over new frameworks.
* Simple explanations over long theory.
* Practical working prototypes over perfect architecture.

## 3. Understand Before Changing

Before editing, inspect the relevant files and understand how the repository currently works.

Do not assume the project structure. Look first.

If something is unclear, state the uncertainty in the plan instead of guessing silently.

## 4. Notebook-First AI Workflow

For most AI/ML work, prefer Jupyter Notebook files (`.ipynb`) instead of plain Python scripts (`.py`), unless the task specifically asks for a Python script.

Notebook work should be readable for students and teammates.

Structure notebooks as alternating explanation and code sections:

1. Markdown text block explaining the next step.
2. Code block implementing that step.
3. Markdown text block explaining the next step.
4. Code block implementing that step.

Example notebook flow:

* Markdown: Explain that the dataset will be loaded and briefly describe what it contains.
* Code: Load the dataset into a dataframe.
* Markdown: Explain what cleaning is needed, such as removing inconsistent formatting, brackets, commas, missing values, or invalid labels.
* Code: Clean the dataset.
* Markdown: Explain feature extraction or preprocessing.
* Code: Apply preprocessing.
* Markdown: Explain model training.
* Code: Train the model.
* Markdown: Explain evaluation.
* Code: Evaluate the model.

Keep markdown explanations short and useful. The goal is for someone reading the notebook to quickly understand where they are and what the code is doing.

## 5. Notebook Style

When writing notebooks:

* Use clear section titles.
* Keep each code cell focused on one purpose.
* Avoid very long code cells when possible.
* Use readable variable names.
* Add comments only when they clarify the code.
* Show important outputs such as dataframe shapes, sample rows, metrics, and plots when useful.
* Avoid unnecessary decoration or complicated visualizations.

## 6. Code Quality

Keep code clean and maintainable.

Prefer:

* Simple functions.
* Minimal dependencies.
* Clear file organization.
* Reproducible steps.
* Small commits or small change sets.
* Comments for important reasoning, not obvious syntax.

Avoid:

* Unrelated refactors.
* Hidden hardcoded paths.
* Large changes outside the task scope.
* Adding dependencies without explaining why.
* Complex solutions when a simple one works.

## 7. Final Response After Changes

After completing approved work, summarize:

1. What changed.
2. Which files were modified.
3. How to run or test it.
4. Any known limitations or next steps.

Keep the summary concise and practical.
