## Requirements

Make sure you have **Python 3.8+** installed.

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---


Make sure all image files are placed in the same directory as `gesture-tracker.py`.

---

## How to Run

Run the main script:
```bash
python gesture-tracker.py
```

Controls:
- Press **q** or **ESC** → Quit the program.

---


## Customization

You can add more gestures by:
1. Creating a new image file
2. Adding it to the `IMAGE_PATHS` dictionary
3. Implementing logic in `classify_gesture()` to detect that gesture
