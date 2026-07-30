# Coding-agent instructions

## Required start

1. Read `PROJECT_STATUS.md`.
2. Read `README.md`.
3. Read `Desktop_Runtime_依赖与集成.md` for cross-repository work.
4. Read only the active section of `AI_Infra_LLM_Agent_待做任务清单.md`.

`PROJECT_STATUS.md` owns sequencing and the single active objective. The task
checklist owns detailed requirements. Do not create another competing tracker.

## Scope

- Keep one flagship project and four depth Labs.
- Complete one vertical MVP loop before expanding modalities or scenarios.
- Treat planned work as planned until code, tests, metrics, and evidence exist.
- Runtime changes must be made in
  `C:\Users\Alienware\guarded-desktop-agent` under that repository's own
  `PROJECT_STATUS.md`.
- Models may propose actions but never bypass Runtime policy, approval, WAL,
  grounding, budgets, or the sole desktop boundary.

## End-of-session update

Report outcome, modified files, exact validation results, limitations, and the
single next objective. Update `PROJECT_STATUS.md` only for real progress.

## GitHub publishing

- When publishing through a pull request, wait for required checks and confirm
  there are no blocking review findings or merge conflicts.
- If the pull request is clean and all required checks pass, merge it directly
  without requesting another confirmation.
- After merge, delete the merged feature branch from both the remote and the
  local checkout. Never merge a failing, blocked, or unresolved pull request.
