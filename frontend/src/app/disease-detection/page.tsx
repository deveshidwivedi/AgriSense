"use client";

import { useState } from "react";
import Link from "next/link";

interface DetectionResult {
  predicted_disease: string;
  confidence: number;
  symptoms: string[];
  remedies: string[];
  gradcam_image?: string;
  gradcam_generated?: boolean;
  gradcam_error?: string;
  visualization_type?: string;
  last_conv_layer?: string;
  reason?: string;
}

export default function DiseaseDetection() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<DetectionResult | null>(null);
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);

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

    setLoading(true);
    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/api/detect-disease/",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();
      console.log("Response data:", data); // Debug log
      setPrediction(data);
      setError("");
    } catch (err) {
      setPrediction(null);
      setError("Error detecting disease. Please try again.");
      console.error("Detection error:", err);
    } finally {
      setLoading(false);
    }
  };

  const getDiseaseStatus = (disease: string) => {
    return disease.toLowerCase().includes("healthy")
      ? "Healthy"
      : "Disease Detected";
  };

  const getStatusColor = (disease: string) => {
    return disease.toLowerCase().includes("healthy")
      ? "text-green-600"
      : "text-red-600";
  };

  const isHealthy = prediction?.predicted_disease
    .toLowerCase()
    .includes("healthy");

  return (
    <div className="flex flex-col items-center min-h-screen bg-gradient-to-br from-green-50 to-green-200 p-6 relative">
      <h1 className="text-4xl font-extrabold text-green-800 drop-shadow-lg mb-8">
        Plant Disease Detection 🌿
      </h1>

      <div className="w-full max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left Column: Upload and Results */}
        <div className="bg-white p-6 rounded-xl shadow-lg">
          {/* File Upload Section */}
          <div className="mb-6">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="block w-full text-gray-700 border border-gray-300 rounded-lg cursor-pointer p-2 mb-4"
            />
            <button
              onClick={handleUpload}
              disabled={!selectedFile || loading}
              className="w-full bg-green-600 text-white px-6 py-3 font-semibold rounded-xl shadow-md hover:bg-green-700 transition-all disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              {loading ? "Analyzing..." : "Upload & Detect"}
            </button>
          </div>

          {/* Image Previews */}
          {(previewUrl || prediction?.gradcam_image) && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              {previewUrl && (
                <div>
                  <h3 className="text-lg font-semibold text-gray-800 mb-2">
                    Original Image
                  </h3>
                  <div className="border rounded-lg overflow-hidden shadow-md">
                    <img
                      src={previewUrl}
                      alt="Original plant"
                      className="w-full h-64 object-cover"
                    />
                  </div>
                </div>
              )}
              {prediction?.gradcam_image && (
                <div>
                  <h3 className="text-lg font-semibold text-gray-800 mb-2">
                    {isHealthy ? "Health Analysis" : "Disease Heatmap"}
                  </h3>
                  <div className="border rounded-lg overflow-hidden shadow-md">
                    <img
                      src={prediction.gradcam_image}
                      alt="Analysis visualization"
                      className="w-full h-64 object-cover"
                    />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Prediction Summary */}
          {prediction && (
            <div className="p-4 bg-gray-50 rounded-lg">
              <h4 className="font-semibold text-gray-800 mb-2">
                Classification Result
              </h4>
              <p
                className={`text-xl font-bold ${getStatusColor(
                  prediction.predicted_disease
                )}`}
              >
                Status: {getDiseaseStatus(prediction.predicted_disease)}
              </p>
              <p className="text-gray-700">
                Prediction:{" "}
                <span className="font-medium">
                  {prediction.predicted_disease.replace(/_/g, " ")}
                </span>
              </p>
              <p className="text-gray-700">
                Confidence:{" "}
                <span className="font-medium">
                  {(prediction.confidence * 100).toFixed(2)}%
                </span>
              </p>
            </div>
          )}

          {error && (
            <div className="mt-4 p-3 bg-red-100 border border-red-300 rounded-lg">
              <p className="text-red-600 font-semibold">{error}</p>
            </div>
          )}
        </div>

        {/* Right Column: Symptoms and Remedies */}
        {prediction && !isHealthy && (
          <div className="bg-white p-6 rounded-xl shadow-lg">
            <h2 className="text-2xl font-bold text-gray-800 mb-4">
              Disease Information
            </h2>

            {/* Symptoms Section */}
            <div className="mb-6">
              <h3 className="text-xl font-semibold text-red-700 mb-2">
                Symptoms
              </h3>
              <ul className="list-disc list-inside space-y-1 text-gray-700">
                {prediction.symptoms.map((symptom, index) => (
                  <li key={index}>{symptom}</li>
                ))}
              </ul>
            </div>

            {/* Remedies Section */}
            <div>
              <h3 className="text-xl font-semibold text-green-700 mb-2">
                Recommended Remedies
              </h3>
              <ul className="list-disc list-inside space-y-1 text-gray-700">
                {prediction.remedies.map((remedy, index) => (
                  <li key={index}>{remedy}</li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {prediction && isHealthy && (
          <div className="bg-white p-6 rounded-xl shadow-lg flex flex-col items-center justify-center">
            <h2 className="text-2xl font-bold text-green-700 mb-4">
              Plant is Healthy!
            </h2>
            <p className="text-gray-600 text-center">
              No disease was detected. Continue with good care practices to
              maintain plant health.
            </p>
          </div>
        )}
      </div>

      {/* Navigation Buttons */}
      <Link href="/crop-recommendation">
        <button className="fixed bottom-6 right-6 bg-emerald-600 text-white px-4 py-3 rounded-full shadow-lg hover:bg-emerald-700 transition-all flex items-center space-x-2 text-sm font-medium border border-emerald-500">
          <span>🌾</span>
          <span>Get Crop Recommendation</span>
        </button>
      </Link>
      <Link href="/">
        <button className="fixed bottom-6 left-6 bg-green-700 text-white px-4 py-3 rounded-full shadow-lg hover:bg-green-800 transition-all flex items-center space-x-2 text-sm font-medium border border-green-600">
          <span>Home</span>
        </button>
      </Link>
    </div>
  );
}
