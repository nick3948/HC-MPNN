# 🧠 HC-MPNN Relational Link Prediction UI

A web-based tool to run and evaluate the [HC-MPNN](https://github.com/HxyScotthuang/HC-MPNN) model for relational link prediction on hypergraphs. Built using **Django**, this interface supports interactive runs, custom settings, downloadable summaries, and real-time model feedback.

---

## 🚀 Features

- Run HC-MPNN on supported datasets (e.g., `FB-AUTO`, `WP-IND`)
- Configure number of epochs and evaluation type (`Raw`, `Filtered`, `Both`)
- Track and download full model summaries as `.pdf`
- Auto-generates system runtime metadata (e.g., hostname, Python version)
- Bootstrap-powered responsive UI with loading animation
- Optional notes input for each run
- PDF export includes full metrics and system details

---

## 📸 Screenshot

![image](https://github.com/user-attachments/assets/61f3f999-e7e6-4680-915b-d245c562f75b)

![image](https://github.com/user-attachments/assets/c36cb5bf-83cd-409e-9406-adbb69a9cc1f)

---

## 📦 Requirements

- Python 3.9+
- Django 4.2+
- ReportLab
- pytz

Install dependencies:
```bash
pip install -r requirements.txt
```

## ⚙️ Setup

1. Clone the repo:
```bash
git clone https://github.com/your-username/HC-MPNN-UI.git
cd HC-MPNN-UI
```
2. Create and activate a virtual environment:
```bash
python3 -m venv hc-mpnn-env
source hc-mpnn-env/bin/activate
```
3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Apply database migrations:
```bash
python manage.py migrate
```

5. Start the Django development server:
```bash
python manage.py runserver
```

6. Open the app in your browser:
```
http://127.0.0.1:8000/
```
