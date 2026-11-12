# Prepare the models and GitHub submission

The hand-in is one **GitHub repository** containing your report, code and supporting
files, with the model link and upload receipt. Follow the steps below to prepare it.

Use one **public Hugging Face model repository** for the baseline and modified model.
Upload directly from GSC, where your final training runs are stored.

## 1. Open the course environment on GSC

Connect to GSC as described in [setup.md](../setup.md#4-connect-to-gsc1).
Open your checkout's `practicals/` directory and activate its environment:

```sh
cd /cl_tmp/$USER/llm-course/YOUR_REPOSITORY/practicals
source .venv/bin/activate
hf auth whoami
```

Replace `YOUR_REPOSITORY` with your checkout's folder name. `whoami` should show
your Hugging Face username. If you have not logged in on GSC yet, follow
[account creation and login](../setup.md#hugging-face-account-and-login).

Run the following commands in this GSC terminal. Packaging and uploading use the
login node; they do not need a GPU job.

## 2. Prepare the submission

For both runs, check that the last line of `metrics.jsonl` shows `step: 30518`
and `tokens: 500006912`. Then package the models:

```sh
python assignment1/code/tools.py submission package \
  --baseline runs/a1-baseline --variant runs/a1-variant \
  --tests assignment1/tests/test_student.py --out submissions/a1
```

The script checks and packages both models, their tokenizer, settings, logs,
saved source and your tests. It loads each model and generates a short sample
on CPU. Add other test files after `--tests` if needed.

Inspect `submissions/a1/` before uploading: its contents will be public.
The exported checkpoints contain what is needed to load and use the models.

Check your packaged tests against the modified model's saved source:

```sh
PYTHONPATH=submissions/a1/variant/source python -B -m pytest \
  submissions/a1/tests -q -p no:cacheprovider
```

## 3. Upload

Replace `YOUR_USERNAME` with the username printed by `hf auth whoami`:

```sh
python assignment1/code/tools.py submission upload submissions/a1 \
  --repo-id YOUR_USERNAME/llm-assignment1 \
  --receipt submissions/a1-receipt.json
```

This creates a new public repository. Choose a name you have not used before.
The receipt records its link, the uploaded revision and the package checksum.
If upload fails after creating the repository, use a new repository name to retry.

## 4. Verify the uploaded models

```sh
python assignment1/code/tools.py submission check \
  --receipt submissions/a1-receipt.json \
  --out submissions/a1-downloaded --trust-code
```

This downloads the recorded version, checks its files and loads both models on
CPU. It runs the saved Python source, so use it only for your own submission.

To copy the receipt to your laptop, open a **local terminal** and run:

```sh
scp YOUR_UNIGRAZONLINE_USERNAME@l1.gsc1.uni-graz.at:/cl_tmp/YOUR_UNIGRAZONLINE_USERNAME/llm-course/YOUR_REPOSITORY/practicals/submissions/a1-receipt.json .
```

Replace the username and repository folder as in setup. The final `.` saves the
file in your local terminal's current directory.

## 5. Assemble the GitHub submission

Hand in one **GitHub repository** containing:

- Your report of at most two pages.
- Source code, configuration files and commands to reproduce the work.
- Your tests, explanations, bug demonstration and its outputs.
- The upload receipt, `a1-receipt.json`.

In the repository README, explain how to run the code and include the **Hugging
Face repository link and full revision ID** from the receipt. One Hugging Face
repository contains both models, in `baseline/` and `variant/` directories; there
is no need to create two repositories. Keep the checkpoint files on Hugging Face
rather than committing them to GitHub.

Email the GitHub link to **lorenz.prattes@uni-graz.at** and
**ruggero.lazzaroni@uni-graz.at** by **Sunday 13 December 2026, 23:59 (Graz time)**.
Use the subject **final submission: assignment 1**. Ensure both instructors can
access the repository. Keep your local model package until grading is complete.
