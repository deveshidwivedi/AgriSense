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
      const response = await axios.post("http://127.0.0.1:8000/api/detect-disease/", formData);
      setResult(response.data.disease);
    } catch (error) {
      setResult("Error detecting disease.");
    }
  };

  return (
    <div className="p-6 text-center">
      <h1 className="text-2xl font-bold">Plant Disease Detection</h1>
      <input type="file" onChange={handleFileChange} className="mt-4" />
      <button onClick={handleUpload} className="bg-green-600 text-white px-4 py-2 mt-4">Upload & Detect</button>
      {result && <p className="mt-4 text-lg">Detected Disease: {result}</p>}
    </div>
  );
}
