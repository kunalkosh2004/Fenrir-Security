# Static Evaluation Results

## 1. Create a new Git branch and switch to it

**🔹 Reference:**
```
To create a new Git branch and switch to it:
1. Use `git checkout -b <branch-name>` to create and switch in one command
2. Or use `git branch <branch-name>` then `git checkout <branch-name>`
3. Example: `git checkout -b feature-login`
4. Verify with `git branch` to see current branch marked with *
```

**🔸 Base Response:** `Create a new git branch and switch`

**🔸 Finetuned Response:** `Create a git branch`

**📊 Metrics:**

- Base BLEU: `0.000142`
- Finetuned BLEU: `0.000000`
- Base ROUGE-L: `0.229508`
- Finetuned ROUGE-L: `0.137931`
- Base Word Overlap: `0.205882`
- Finetuned Word Overlap: `0.117647`

## 2. Compress the folder reports into reports.tar.gz

**🔹 Reference:**
```
To compress the folder reports into reports.tar.gz:
1. Use `tar -czf reports.tar.gz reports/`
2. The -c flag creates archive, -z compresses with gzip, -f specifies filename
3. Verify compression: `ls -lh reports.tar.gz`
4. Alternative: `tar -czvf reports.tar.gz reports/` (with verbose output)
```

**🔸 Base Response:** `How do you report bugs?`

**🔸 Finetuned Response:** `int : report`

**📊 Metrics:**

- Base BLEU: `0.000000`
- Finetuned BLEU: `0.000000`
- Base ROUGE-L: `0.037037`
- Finetuned ROUGE-L: `0.039216`
- Base Word Overlap: `0.000000`
- Finetuned Word Overlap: `0.025641`

## 3. List all Python files in the current directory recursively

**🔹 Reference:**
```
To list all Python files recursively:
1. Use `find . -name '*.py'` to find all .py files
2. Use `find . -type f -name '*.py'` to ensure only files, not directories
3. Alternative: `ls -la **/*.py` (if shell supports globstar)
4. With details: `find . -name '*.py' -exec ls -l {} \;`
```

**🔸 Base Response:** `Python : a program that makes you go to the same place twice`

**🔸 Finetuned Response:** `Python :`

**📊 Metrics:**

- Base BLEU: `0.000132`
- Finetuned BLEU: `0.000000`
- Base ROUGE-L: `0.067797`
- Finetuned ROUGE-L: `0.041667`
- Base Word Overlap: `0.068182`
- Finetuned Word Overlap: `0.045455`

## 4. Set up a virtual environment and install requests

**🔹 Reference:**
```
To set up virtual environment and install requests:
1. Create venv: `python -m venv myenv` or `python3 -m venv myenv`
2. Activate: `source myenv/bin/activate` (Linux/Mac) or `myenv\Scripts\activate` (Windows)
3. Install requests: `pip install requests`
4. Verify installation: `pip show requests`
5. Deactivate when done: `deactivate`
```

**🔸 Base Response:** `Response : No.`

**🔸 Finetuned Response:** `Hello, I would like to have your computer running on a server with the same IP as the computer you are on.`

**📊 Metrics:**

- Base BLEU: `0.000000`
- Finetuned BLEU: `0.001718`
- Base ROUGE-L: `0.000000`
- Finetuned ROUGE-L: `0.027778`
- Base Word Overlap: `0.027027`
- Finetuned Word Overlap: `0.027027`

## 5. Fetch only the first ten lines of a file named output.log

**🔹 Reference:**
```
To fetch the first ten lines of output.log:
1. Use `head -10 output.log` or `head -n 10 output.log`
2. Default head shows first 10 lines: `head output.log`
3. Alternative with sed: `sed -n '1,10p' output.log`
4. Check file exists first: `ls -la output.log`
```

**🔸 Base Response:** `Fetch only the first line of a file named output.log`

**🔸 Finetuned Response:** `Fetch only the first one line of a file named output.log`

**📊 Metrics:**

- Base BLEU: `0.000558`
- Finetuned BLEU: `0.000870`
- Base ROUGE-L: `0.262295`
- Finetuned ROUGE-L: `0.258065`
- Base Word Overlap: `0.187500`
- Finetuned Word Overlap: `0.187500`

## 6. Find and kill all processes containing 'python' in their name, but exclude the current shell session

**🔹 Reference:**
```
To find and kill Python processes safely:
1. List processes: `ps aux | grep python | grep -v grep`
2. Get PIDs: `pgrep -f python`
3. Exclude current shell: `ps aux | grep python | grep -v grep | grep -v $$`
4. Kill specific processes: `pkill -f 'python script.py'`
5. Force kill if needed: `pkill -9 -f python`
6. Always verify before killing: check process details with `ps -p <PID>`
```

**🔸 Base Response:** `Not found`

**🔸 Finetuned Response:** `How do you stop the Python?`

**📊 Metrics:**

- Base BLEU: `0.000000`
- Finetuned BLEU: `0.000000`
- Base ROUGE-L: `0.000000`
- Finetuned ROUGE-L: `0.027778`
- Base Word Overlap: `0.000000`
- Finetuned Word Overlap: `0.020000`

## 7. Recover uncommitted changes after accidentally running 'git reset --hard' in a repository with unstaged modifications

**🔹 Reference:**
```
To recover from accidental git reset --hard:
1. Check reflog immediately: `git reflog`
2. Find the commit before reset: look for HEAD@{1} or similar
3. Reset to previous state: `git reset --hard HEAD@{1}`
4. If files were untracked, they may be lost permanently
5. Check git fsck for dangling commits: `git fsck --lost-found`
6. For future: use `git stash` before risky operations
7. Note: Unstaged changes are usually unrecoverable after hard reset
```

**🔸 Base Response:** `Relevant username`

**🔸 Finetuned Response:** `git reset... hard reset`

**📊 Metrics:**

- Base BLEU: `0.000000`
- Finetuned BLEU: `0.000000`
- Base ROUGE-L: `0.000000`
- Finetuned ROUGE-L: `0.101266`
- Base Word Overlap: `0.000000`
- Finetuned Word Overlap: `0.049180`
