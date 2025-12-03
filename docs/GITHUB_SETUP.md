# Setting up GitHub Authentication

Since the GitHub CLI (`gh`) is not installed, you need to authenticate manually to push changes.

## Option 1: Personal Access Token (Recommended for HTTPS)
If you are using HTTPS (which you are: `https://github.com/vaishnavak2001/ControlGym-mass-spring-damper.git`), you need a Personal Access Token (PAT) to use as your password.

1.  **Generate a Token**:
    *   Go to [GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)](https://github.com/settings/tokens).
    *   Click **Generate new token (classic)**.
    *   Give it a name (e.g., "ControlGym PC").
    *   Select the **repo** scope (this is required to push).
    *   Click **Generate token**.
    *   **Copy the token** immediately (you won't see it again).

2.  **Authenticate**:
    *   Run `git push` in your terminal.
    *   When asked for **Username**, enter your GitHub username: `vaishnavak2001`.
    *   When asked for **Password**, paste the **Personal Access Token** you just copied.
        *   *Note: On Windows, the Git Credential Manager might pop up. Paste the token there.*

## Option 2: SSH Keys (Advanced)
If you prefer SSH, you must generate keys and add them to GitHub.

1.  **Generate Key**:
    ```bash
    ssh-keygen -t ed25519 -C "your_email@example.com"
    ```
    (Press Enter to accept defaults).

2.  **Add to GitHub**:
    *   Copy the public key: `type %USERPROFILE%\.ssh\id_ed25519.pub`
    *   Go to [GitHub SSH Keys](https://github.com/settings/keys).
    *   Click **New SSH key**, give it a title, and paste the key.

3.  **Switch Remote to SSH**:
    ```bash
    git remote set-url origin git@github.com:vaishnavak2001/ControlGym-mass-spring-damper.git
    ```

## Verification
After completing one of the above, try pushing again:
```bash
git push
```
