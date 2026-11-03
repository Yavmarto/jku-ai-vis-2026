# CI: hosted checks and the Akida runner

Two workflows in `.github/workflows/`:

| Workflow | Runs on | Triggers | Does |
|---|---|---|---|
| `ci.yml` | GitHub-hosted Ubuntu | push, pull request (forks included) | `ruff check`, `pytest -m "not hw"`. No chip, no secrets. |
| `hw.yml` | self-hosted `[self-hosted, linux, akida]` | push to this repo, manual run | preflight, `pytest -m hw` on the AKD1000, uploads `outputs/` as an artifact for 14 days |

Pull requests from forks never reach the server: `hw.yml` has no `pull_request` trigger. Checks from a
push to a branch appear on that branch's PR.

## Rules for the group

- Push to a branch in this repo (not a fork), open a PR, wait for `ci` and `hw`.
- Label results `hardware` or `simulator`. Never mix the two in one table.
- Do not put secrets, course PDFs or lecture transcripts in the repo.

## One card, one job at a time

Register **one** runner with the `akida` label. A single runner takes one job at a time, so jobs queue
and never share the card. We do not use a `concurrency:` group, because GitHub keeps only one pending run
per group and cancels older pending ones, which would silently drop pushes from other members.

The test step runs under `flock "${AKIDA_LOCK:-/tmp/akida.lock}"`. If you use the card by hand on the
server, take the same lock: `flock /tmp/akida.lock python your_script.py`.

## Setting up the runner (once, on the server)

1. Create an unprivileged user `gh-runner` with no sudo. Give it access to the Akida device only. Do
   not run the runner as your own account or as root, and keep no other secrets on that account.
2. In GitHub: repo Settings > Actions > Runners > New self-hosted runner (Linux). Follow the shown
   commands and add the labels `self-hosted,linux,akida`. The registration token is shown to you on that
   page. Do not paste it into chat, files or issues.
3. Install it as a service (`./svc.sh install && ./svc.sh start`) so it survives reboots.
4. Run `bash scripts/hw_preflight.sh` as `gh-runner` once.
5. Settings > Actions > General: set "Require approval for all outside collaborators" for fork PRs.
   Keep collaborators to the four group members.
6. Settings > Branches: protect `main`, require the `ci` check. Require `hw` once it is stable.

## Security note

A self-hosted runner executes the code in each push as `gh-runner` on your server, and GitHub advises
against self-hosted runners on public repos. Anyone who can push to the repo can run code there. Keep
the repo private until the course needs it public, or keep collaborators to the group. Optionally
create an Environment `akida-hw` with you as required reviewer and set `environment: akida-hw` on the
`chip` job to approve each run.

## Known gaps

- `requirements-akida.txt` is unpinned. Pin the versions and the Python version after the first
  hardware run.
- The first green `hw` run is the hardware check. The akida API calls in
  `tests/hw/test_akida_smoke.py` follow the docs from memory and may need fixing.
- The one-image model test is skipped until `AKIDA_SMOKE_MODEL` points to a `.fbz` file.
- The venv is rebuilt on every run, because checkout cleans ignored files.
- The hosted job does not run the Akida simulator. Add it after checking that `akida` installs on hosted
  runners.
- The full fog/noise/low-contrast evaluation workflow is not written yet.
