## AgriSense

An easy-to-use platform for farmers to:

- Detect plant leaf diseases from lab or mobile photos
- Visualize affected areas on the leaf with a Grad-CAM heatmap (red = most affected)
- Get disease name, key symptoms, and practical remedies
- Receive data-driven crop recommendations using NPK, pH, temperature, and humidity inputs, aligned with meteorological context

### Features

- **Leaf disease detection**: Upload a leaf image; the system identifies the disease and overlays a Grad-CAM heatmap to highlight the most affected regions.
- **Actionable guidance**: Shows the detected disease’s symptoms and recommended remedies.
- **Crop recommendation**: Enter NPK, pH, temperature, and humidity; the model suggests crops that can improve yield under current conditions.

### Tech Stack

- **Frontend**: Next.js (TypeScript)
- **Backend**: Django
- **ML Models**: Pretrained models for disease classification and crop recommendation (stored under `backend/models/`)

---

## Getting Started

### Prerequisites

- Node.js and npm
- Python 3.x

### 1) Frontend

From the `frontend` directory:

```bash
cd frontend
npm install
npm run dev
```

App runs at `http://localhost:3000`.

### 2) Backend

From the `backend` directory, create and activate a virtual environment, install requirements, and start the server:

```bash
cd backend
python -m venv venv
venv\Scripts\activate  # On Windows PowerShell
pip install -r requirements.txt
python manage.py runserver
```

API runs at `http://127.0.0.1:8000`.

---

## How It Works (High Level)

- **Disease Detection**: The uploaded leaf image is passed through a plant disease classifier. Grad-CAM highlights the regions most responsible for the prediction, coloring highly affected areas in red. The app then displays disease details, symptoms, and remedies.
- **Crop Recommendation**: Given N, P, K, pH, temperature, and humidity, the model recommends crops that are likely to perform well considering meteorological conditions.

---

## Project Structure

- `frontend/`: Next.js UI
- `backend/`: Django project and API
- `backend/models/`: Pretrained ML models used by the API
