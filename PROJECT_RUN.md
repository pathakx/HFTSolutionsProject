
# Running the Project

## 1. Create Virtual Environment

### Windows PowerShell

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
````

### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Run Public Tests

```bash
pytest tests/ -v
```

---

## 4. Run Sanity Check

```bash
python sanity_check.py
```

---

## 5. Run Main Demonstration

```bash
python main.py
```

This runs:

* Standard parameter optimization
* Additional strategy analytics
* Walk-forward validation
* Performance benchmark

---

## 6. Run Benchmark Only

```bash
python benchmark.py
```

---

## 7. Run Walk-Forward Validation Only

```bash
python walk_forward.py
```

```
```
