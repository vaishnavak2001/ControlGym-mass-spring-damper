# Browser Setup & Troubleshooting Guide

## 1. Running the Project in Your Browser
The project uses **Streamlit** to create a web-based dashboard.

### Status
The dashboard is currently **running** on your machine.
- **URL**: [http://localhost:8501](http://localhost:8501)

### How to Run Manually
If you stop the current process, you can restart it with:
```bash
streamlit run dashboard/app.py
```

## 2. Fixing "Chrome Not Found" for AI Recording
If you want the AI to record the simulation, **Google Chrome** must be installed and accessible.

### Issue
The AI agent could not find a valid Chrome installation to launch the browser tool.

### Solution
1.  **Install Google Chrome**:
    *   Download from [google.com/chrome](https://www.google.com/chrome/).
    *   Install it in the default location.

2.  **Verify Installation**:
    *   Ensure `chrome.exe` exists at one of these locations:
        *   `C:\Program Files\Google\Chrome\Application\chrome.exe`
        *   `C:\Program Files (x86)\Google\Chrome\Application\chrome.exe`
        *   `%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe`

3.  **Restart**:
    *   After installing, try the request again.
