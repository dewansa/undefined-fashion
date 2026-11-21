# Fashion-MNIST class leaderboard

Train a model that looks at a small picture of clothing and says what it is:
a shirt, a sneaker, a bag, and so on. Then see how you rank against your classmates.

**Leaderboard:** https://dewansa.github.io/undefined-fashion/

Every model is scored on 10,000 test images it has never seen. The leaderboard shows
these numbers for each submission:

- **Test accuracy:** how many test images the model labels correctly. Higher is better.
- **Parameters:** how many numbers the model learns by gradient descent. Smaller is neater.
  Values stored with `register_buffer` (for example, class averages) are not counted, so a
  model can have 0.
- **Size:** all the memory the trained model holds on to, in bytes. Parameters count, and
  so do buffers and any other tensor the model keeps, in an attribute, a list or a dict.
  A model that stores all the training images is as big as those images (about 188 MB).
- **Compute:** how much arithmetic your `train()` function did, counted in FLOPs
  (floating-point operations). One multiply or add is one FLOP, so 225 GFLOP is
  225 billion. It counts the matrix multiplies and convolutions of every forward and
  backward pass, and also linear algebra (`torch.linalg.solve`, `lstsq`, `eigh`, `svd`,
  ...), distances (`torch.cdist`) and sorting. It skips cheap steps such as ReLU. Unlike time, it is the same on
  every computer. Your code is not changed to count it: the counter only watches.
- **Compute per image:** the same count for labelling one test image with the trained
  model. A model can be cheap to train and costly to use, like one that compares each
  test image with every training image.

  A compute count marked **\*** may be too low. Your code ran operations the counter has
  no formula for, such as `torch.linalg.matrix_exp`. Point at the number on
  the leaderboard to see which ones. `uv run score ... --flops` lists them too.
- **Training time:** how long your `train()` function ran on GitHub's computer.

The chart on the leaderboard draws a line through the **best trade-offs**: submissions
that no other submission beats on both accuracy and cost (compute, time or parameters).
A small, cheap model can be on that line too.

## What you need

- [uv](https://docs.astral.sh/uv/getting-started/installation/), which installs Python and everything else for you.
- A GitHub account.

## How to submit

1. **Fork** this repo on GitHub, then clone your fork and set it up:

   ```sh
   git clone https://github.com/<your-github-name>/undefined-fashion
   cd undefined-fashion
   uv sync
   ```

2. **Make your folder** by copying the baseline. Use your GitHub name as the folder name:

   ```sh
   cp -r submissions/baseline submissions/<your-github-name>
   ```

3. **Improve the model.** Edit `submissions/<your-github-name>/model.py`. You own the
   whole training loop: the model, the optimizer, the number of epochs, everything.

4. **Score it on your computer:**

   ```sh
   uv run score submissions/<your-github-name>
   ```

   The first run downloads the images (about 30 MB) into `data/`.
   To also see your compute, add `--flops`. That trains your model a second time,
   and counting makes training a few times slower:

   ```sh
   uv run score submissions/<your-github-name> --flops
   ```

5. **Open a pull request** from your fork. GitHub checks your code, then trains and
   scores your model. Fix anything the check complains about. Before you push, you can
   run the same check yourself:

   ```sh
   uv run ruff format .
   uv run ruff check .
   uv run pytest
   ```

6. **See your score.** A few minutes later, a bot comments your score on the pull request,
   and your row appears on the leaderboard marked *not reviewed*. Each new push to the
   pull request is scored again and replaces that row. (The first time you open a pull
   request, the teacher may need to approve the run before it starts.)

7. **Wait for the teacher to merge it.** After the merge, GitHub scores your model once
   more, counts its compute, and your row becomes *reviewed*. Rows that are not reviewed
   show no compute yet. Want to try again? Open a new pull request.
   Every merged submission keeps its own row, so your earlier scores stay.

## The rules

Your `model.py` must define this function:

```python
def train(train_set: Dataset) -> nn.Module:
    ...
    return model
```

- `train_set` holds the 60,000 training images. Each item is `(image, label)`. The image is
  a `1x28x28` tensor with values from 0 to 1, and the label is a number from 0 to 9.
- Return your trained model. We test it on images in the same format, on the CPU. For a
  batch of images it must return one row of 10 class scores per image, shape `(batch, 10)`.
  The class with the highest score is its answer.
- **Train only on `train_set`.** Do not download or read the test images in any way.
- **Training must finish in 10 minutes** on GitHub's computer, which has 4 CPU cores and no
  GPU. The baseline takes under 2 minutes there. Your laptop may be faster or slower, so
  the time GitHub reports on your pull request is the one that counts.
- If you change the input (for example, normalize it), do it inside your model, because
  the test images arrive unchanged.
- Put all your code in `model.py` and only change files inside your own folder.

## Ideas to try

- Use the `Adam` optimizer instead of `SGD`, or a bigger learning rate.
- Train for more epochs and watch the valid loss, so you stop before the model overfits.
- Add more layers, or use a small convolutional network (`nn.Conv2d`).
- Normalize the input: subtract the mean (0.286) and divide by the standard deviation (0.353).
- Training is slow mostly because each image is converted one at a time. Load all images
  into one tensor once at the start, and every epoch gets much faster.

## Project layout

```
src/fashion/data.py           loads the images
src/fashion/score.py          trains a submission and measures it (the `score` command)
submissions/<name>/model.py   one folder per student
src/fashion/leaderboard.py    keeps the leaderboard rows and checks pull request scores
results/                      one small JSON file per merged (reviewed) submission
results/pending/              one file per open pull request (not reviewed)
site/index.html               the leaderboard page
tests/                        checks for the scoring code and all submissions
.github/workflows/check.yml          checks and scores every pull request (read-only)
.github/workflows/pr-leaderboard.yml puts pull request scores on the leaderboard
.github/workflows/publish.yml        scores merged submissions and updates the leaderboard
```

## For the teacher

**First-time setup:** in the repo's Settings > Pages, set Source to "GitHub Actions".

**Pull request scores:** every pull request is checked and scored right away, and its
score goes on the leaderboard marked *not reviewed*. Nobody has read that code yet, so
treat those numbers as a preview. The job that runs student code can only read the repo.
A separate job that never runs student code checks the score file and adds the row.
Rows of closed pull requests are removed. You can hide unreviewed rows with the checkbox
on the leaderboard page.

**Reviewing a pull request:** read the student's `model.py` before you merge. Look for
anything that reads the test images (`train=False`), downloads files, or changes files
outside the student's folder. Merging is what makes a score *reviewed*.

**After a merge:** the "Score and publish" workflow trains every submission that changed
again, then trains it once more inside a FLOP counter to measure its compute. Counting
slows training down 2 to 6 times, so it is a separate run with its own 30-minute limit,
and the training time on the leaderboard comes from the first run. If counting does not
finish, the row has no compute. The workflow saves a JSON file per submission in `results/`, removes the pull request's
*not reviewed* row, and updates the leaderboard. If a
submission fails, the run shows a red cross and the log says why. The other submissions
are still published.

**Scoring again:** in the Actions tab, open "Score and publish", click "Run workflow" and
type a folder name, or `all` to score every submission again. Each run adds new rows.
Rows from before compute was counted show "—" for it: score them again, then delete
the old rows' files if you want only the new ones.

**Removing a row:** delete its file from `results/` and push. That push updates the website.

**Branch rules:** the workflows push new results straight to `main`. If you protect `main`,
allow GitHub Actions to push to it.

**First-time contributors:** GitHub may ask you to approve the first pull request run of a
new student. You can change this in Settings > Actions > General.
