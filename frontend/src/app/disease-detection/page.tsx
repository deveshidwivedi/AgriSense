"use client";

import { useState } from "react";
import axios from "axios";

export default function DiseaseDetection() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<{
    predicted_disease: string;
    confidence: number;
  } | null>(null);
  const [error, setError] = useState<string>("");

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPrediction(null);
      setError("");
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/api/detect-disease/",
        formData
      );
      setPrediction(response.data);
      setError("");
    } catch (err) {
      setPrediction(null);
      setError("Error detecting disease.");
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-center bg-gradient-to-br from-green-50 to-green-200 p-6">
      <h1 className="text-4xl font-extrabold text-green-800 drop-shadow-lg">
        Plant Disease Detection 🌿
      </h1>

      <div className="mt-6 bg-white p-6 rounded-xl shadow-lg w-full max-w-md break-words">
        <input
          type="file"
          onChange={handleFileChange}
          className="block w-full text-gray-700 border border-gray-300 rounded-lg cursor-pointer p-2"
        />

        {previewUrl && (
          <div className="mt-4">
            <p className="text-sm text-gray-600 mb-1">Image Preview:</p>
            <img
              src={previewUrl}
              alt="Preview"
              className="w-full rounded-lg object-cover max-h-64 border"
            />
          </div>
        )}

        <button
          onClick={handleUpload}
          className="mt-4 w-full bg-green-600 text-white px-6 py-3 font-semibold rounded-xl shadow-md hover:bg-green-700 transition-all"
        >
          Upload & Detect
        </button>

        {prediction && (
          <div className="mt-4 text-green-900 break-words">
            <p className="text-lg font-semibold break-words">
              Detected Disease:
              <span className="font-bold ml-1 break-words">
                {prediction.predicted_disease.replace(/_/g, " ")}
              </span>
            </p>
            <p className="text-sm">
              Confidence: {(prediction.confidence * 100).toFixed(2)}%
            </p>
          </div>
        )}

        {error && (
          <p className="mt-4 text-red-600 font-semibold">{error}</p>
        )}
      </div>
    </div>
  );
}
