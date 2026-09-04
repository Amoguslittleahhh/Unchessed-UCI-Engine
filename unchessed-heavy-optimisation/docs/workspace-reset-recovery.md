# Recovering from a workspace reset

## Symptom

At the start of a session the checkout is not where the last session left it:

```
$ git log --oneline -1
7a988bf Merge remote-tracking branch 'origin/main' into arena/01a0175d-...
```

`git status` then shows a long list of modified and untracked files. Those are
**not** new edits — they are previously committed work that the reset has
spilled back into the working tree, because the branch pointer moved backwards
to an unrelated commit while the files on disk stayed put.

This has recurred several times. It is not caused by anything in the repo.

## Do not panic-commit

The instinct is to `git add -A && git commit` the dirty tree. **Don't.** That
recommits work that is already on the remote, usually on top of the wrong
base, and produces a duplicate-content commit with a misleading diff.

Check the remote first — the work is almost always already there:

```sh
git ls-remote origin arena/<session-branch>
```

## Recovery

If the remote SHA is ahead of local `HEAD`, the remote is the source of truth:

```sh
git fetch origin arena/<session-branch>
git reset --soft FETCH_HEAD   # move the branch pointer back to the real tip
git reset                     # unstage; working tree now matches that commit
git status --short            # expect empty
```

`--soft` then a plain `reset` is deliberate: it moves the pointer and unstages
without touching file contents, so nothing on disk is destroyed even if the
diagnosis was wrong. Avoid `--hard` unless the tree has been confirmed
redundant.

If `git status` is empty afterwards, the tree matches the pushed commit and
nothing was lost.

## Prevention

Commit and push after every task, not just at the end of a round, and verify
with `git ls-remote` rather than trusting local state. The remote is the only
thing that survives a reset — anything uncommitted when one hits has to be
redone from scratch.

Always re-verify `HEAD` at the start of a session before making edits, since
building on the wrong base is much harder to unpick than fixing the pointer
up front.
