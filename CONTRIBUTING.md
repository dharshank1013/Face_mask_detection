# Contributing to Real-Time Face Mask Detection System

Thank you for your interest in contributing to the **Face Mask Detection System**! We welcome bug fixes, documentation improvements, new features, and feedback.

---

## 🛠️ Getting Started

1. **Fork the repository** on GitHub:
   ```bash
   git clone https://github.com/dharshank1013/Face_mask_detection.git
   cd Face_mask_detection
   ```

2. **Set up a virtual environment**:
   ```powershell
   py -3.10 -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify your hardware acceleration**:
   ```powershell
   python verify_hardware.py
   ```

---

## 🧪 Testing Your Changes

Before submitting a Pull Request, ensure that all unit and integration tests pass cleanly:

```powershell
pytest tests/test_system.py -v
```

And verify code syntax:

```powershell
python -m compileall .
```

---

## 📝 Pull Request Guidelines

1. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Commit your changes**:
   Write clear, concise commit messages explaining *what* and *why*.
3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
4. **Open a Pull Request**:
   Fill out the PR template checklist completely.

---

## ⚖️ Code of Conduct

Please be respectful and constructive in all discussions, issues, and pull request reviews.
