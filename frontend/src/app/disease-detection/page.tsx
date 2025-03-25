"use client";
import { useState } from "react";
import axios from "axios";

export default function DiseaseDetection() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<string>("");

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setSelectedFile(event.target.files[0]);
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
      setResult(response.data.disease);
    } catch (error) {
      setResult("Error detecting disease.");
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-center bg-gradient-to-br from-green-50 to-green-200 p-6">
      <h1 className="text-4xl font-extrabold text-green-800 drop-shadow-lg">
        Plant Disease Detection 🌿
      </h1>
      <div className="mt-6 bg-white p-6 rounded-xl shadow-lg w-full max-w-md">
        <input
          type="file"
          onChange={handleFileChange}
          className="block w-full text-gray-700 border border-gray-300 rounded-lg cursor-pointer p-2"
        />
        <button
          onClick={handleUpload}
          className="mt-4 w-full bg-green-600 text-white px-6 py-3 font-semibold rounded-xl shadow-md hover:bg-green-700 transition-all"
        >
          Upload & Detect
        </button>
        {result && (
          <p className="mt-4 text-lg text-green-900 font-medium">
            Detected Disease: <span className="font-bold">{result}</span>
          </p>
        )}
      </div>
    </div>
  );
}
