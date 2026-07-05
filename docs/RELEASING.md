# Releasing to PyPI

Publishing is automated via GitHub Actions using **Trusted Publishing** (OIDC) — no API
tokens are created or stored. A push of a version tag builds, checks, and uploads the
release. There is a **one-time setup** on PyPI, then every release is one tag.

## One-time setup (do this once)

1. **Create the project's trusted publisher on PyPI.** Because the project doesn't exist on
   PyPI yet, add it as a *pending* publisher:
   - Log in at <https://pypi.org> → **Your account → Publishing** →
     *"Add a new pending publisher"*.
   - Fill in:
     - **PyPI Project Name:** `cite-or-abstain`
     - **Owner:** `cl-poehl`
     - **Repository name:** `cite-or-abstain`
     - **Workflow name:** `publish.yml`
     - **Environment name:** *(leave blank)*
   - Save. PyPI will now accept an upload from this repo's `publish.yml` workflow and create
     the project on first publish.

   (Optional hardening: create a GitHub Environment named `pypi` with required reviewers,
   add `environment: pypi` under the `publish` job, and set the same environment name in the
   PyPI publisher. Not required for a first release.)

## Each release

1. Pick a version. For the **first PyPI release**, a clean `0.7.0` reads better than
   continuing the `0.6.x` patch line. Bump it in **two** places:
   - `pyproject.toml` → `version = "0.7.0"`
   - `coa/__init__.py` → `__version__ = "0.7.0"`
2. Commit and push to `main`. Wait for CI to go green.
3. Tag and push the tag — this triggers the publish workflow:
   ```bash
   git tag v0.7.0
   git push origin v0.7.0
   ```
4. Watch the **Publish to PyPI** action. When it's green, `pip install cite-or-abstain`
   works, and you can drop the "not yet on PyPI" note from the README install section.
5. (Optional) create a GitHub Release from the tag with notes drawn from the README Status
   section.

## Verifying locally before you tag

```bash
python -m build          # builds dist/*.whl and dist/*.tar.gz
python -m twine check dist/*   # validates metadata + README rendering
```

Both should pass before you push a tag.

## After the first publish

- Add a version badge to the README:
  `[![PyPI](https://img.shields.io/pypi/v/cite-or-abstain)](https://pypi.org/project/cite-or-abstain/)`
- Change the install instructions back to the simple `pip install cite-or-abstain`.
