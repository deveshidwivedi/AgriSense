# API Documentation

## Crop Recommendation

### Endpoint:
**POST** `http://127.0.0.1:8000/api/crop-recommendation/`

### Description:
This endpoint accepts soil and environmental parameters as input and returns recommended crops based on the provided data.

### Request Body:
The request body should be a JSON object with the following fields:

| Field       | Type   | Description                                  |
|------------|--------|----------------------------------------------|
| soil_type  | string | The type of soil (e.g., sandy, clay).        |
| ph         | string | The pH level of the soil.                    |
| nitrogen   | string | The nitrogen level in the soil.              |
| phosphorus | string | The phosphorus level in the soil.            |
| potassium  | string | The potassium level in the soil.             |
| temperature| string | The temperature of the environment.          |
| humidity   | string | The humidity level of the environment.       |
| rainfall   | string | The amount of rainfall in the area.          |

### Response:
The response will be a JSON object with the following field:

| Field              | Type   | Description                                   |
|-------------------|--------|-----------------------------------------------|
| recommended_crops | string | A list or description of recommended crops.   |

---

## Disease Detection

### Endpoint:
**POST** `http://127.0.0.1:8000/api/disease-detection/`

### Description:
This endpoint accepts an image of a plant leaf and returns the detected disease along with possible remedies.

### Request Body:
The request body should be a `multipart/form-data` object with the following field:

| Field  | Type  | Description                          |
|--------|-------|--------------------------------------|
| image  | file  | An image file of the plant leaf.    |

### Response:
The response will be a JSON object with the following fields:

| Field   | Type   | Description                              |
|---------|--------|------------------------------------------|
| disease | string | The name of the detected disease.       |
| remedies| string | Suggested remedies for the disease.     |
