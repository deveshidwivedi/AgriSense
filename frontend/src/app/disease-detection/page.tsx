"use client";

import { useState } from "react";
import Link from "next/link";

interface DetectionResult {
  predicted_disease: string;
  confidence: number;
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

  const getVisualizationTitle = (prediction: DetectionResult) => {
    if (!prediction.gradcam_generated) {
      if (prediction.reason === "healthy_plant") {
        return "Plant Analysis (Healthy)";
      } else if (prediction.reason === "low_confidence") {
        return "Plant Analysis (Low Confidence)";
      } else {
        return "Plant Analysis";
      }
    }
    return "Disease Heat Map";
  };

  const getVisualizationDescription = (prediction: DetectionResult) => {
    if (!prediction.gradcam_generated) {
      if (prediction.reason === "healthy_plant") {
        return "No heat map generated - plant appears healthy";
      } else if (prediction.reason === "low_confidence") {
        return "No heat map generated - prediction confidence too low";
      } else if (prediction.gradcam_error) {
        return `Heat map generation failed: ${prediction.gradcam_error}`;
      }
      return "Original image displayed";
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-center bg-gradient-to-br from-green-50 to-green-200 p-6 relative">
      <h1 className="text-4xl font-extrabold text-green-800 drop-shadow-lg mb-8">
        Plant Disease Detection 🌿
      </h1>

      <div className="bg-white p-6 rounded-xl shadow-lg w-full max-w-4xl">
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

        {/* Results Section */}
        {(previewUrl || prediction?.gradcam_image) && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Original Image */}
            {previewUrl && (
              <div>
                <h3 className="text-lg font-semibold text-gray-800 mb-2">
                  Original Image
                </h3>
                <div className="border rounded-lg overflow-hidden shadow-md">
                  <img
                    src={previewUrl}
                    alt="Original plant image"
                    className="w-full h-64 object-cover"
                  />
                </div>
              </div>
            )}

            {/* GradCAM Visualization */}
            {prediction?.gradcam_image && (
              <div>
                <h3 className="text-lg font-semibold text-gray-800 mb-2">
                  {getVisualizationTitle(prediction)}
                </h3>
                <div className="border rounded-lg overflow-hidden shadow-md">
                  <img
                    src={prediction.gradcam_image}
                    alt="Disease analysis visualization"
                    className="w-full h-64 object-cover"
                  />
                </div>
                <p className="text-sm text-gray-600 mt-2">
                  {getVisualizationDescription(prediction)}
                </p>
                {/* {prediction.gradcam_generated && (
                  <div className="mt-2 p-2 bg-blue-50 rounded text-xs text-gray-700">
                    <strong>Technical Info:</strong> Visualization generated
                    using layer: {prediction.last_conv_layer}
                  </div>
                )} */}
              </div>
            )}
          </div>
        )}

        {/* Prediction Results */}
        {prediction && (
          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-semibold text-gray-800 mb-2">
                  Classification Results
                </h4>
                <p
                  className={`text-lg font-bold ${getStatusColor(
                    prediction.predicted_disease
                  )}`}
                >
                  Status: {getDiseaseStatus(prediction.predicted_disease)}
                </p>
                <p className="text-gray-700">
                  Disease:{" "}
                  <span className="font-medium">
                    {prediction.predicted_disease.replace(/_/g, " ")}
                  </span>
                </p>
                {/* <p className="text-gray-700">
                  Confidence:{" "}
                  <span className="font-medium">
                    {(prediction.confidence * 100).toFixed(2)}%
                  </span>
                </p> */}
              </div>

              <div>
                <h4 className="font-semibold text-gray-800 mb-2">
                  Visualization Details
                </h4>
                <p className="text-gray-700">
                  Heat Map:{" "}
                  <span
                    className={`font-medium ${
                      prediction.gradcam_generated
                        ? "text-green-600"
                        : "text-gray-600"
                    }`}
                  >
                    {prediction.gradcam_generated
                      ? "Generated"
                      : "Not Generated"}
                  </span>
                </p>
                {/* {prediction.gradcam_generated && (
                  <p className="text-sm text-gray-600 mt-1">
                    Heat map highlights areas that influenced the disease
                    classification decision.
                  </p>
                )} */}
                {prediction.gradcam_error && (
                  <p className="text-sm text-red-600 mt-1">
                    Error: {prediction.gradcam_error}
                  </p>
                )}
              </div>
            </div>

            {/* Health Status Banner */}
            <div
              className={`mt-4 p-3 rounded-lg ${
                prediction.predicted_disease.toLowerCase().includes("healthy")
                  ? "bg-green-100 border border-green-300"
                  : "bg-red-100 border border-red-300"
              }`}
            >
              <p
                className={`font-semibold ${getStatusColor(
                  prediction.predicted_disease
                )}`}
              >
                {prediction.predicted_disease.toLowerCase().includes("healthy")
                  ? "✅ Plant appears healthy!"
                  : `⚠️ Disease detected: ${prediction.predicted_disease.replace(
                      /_/g,
                      " "
                    )}`}
              </p>
              {prediction.gradcam_generated && (
                <p className="text-sm text-gray-700 mt-1">
                  The heat map visualization shows which parts of the leaf
                  contributed most to the disease classification. Warmer colors
                  (red/orange) indicate areas of highest attention by the AI
                  model.
                </p>
              )}
            </div>

            {/* Legend for GradCAM */}
            {/* {prediction.gradcam_generated && (
              <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                <h5 className="font-semibold text-gray-800 mb-2">
                  Heat Map Legend
                </h5>
                <div className="flex items-center space-x-4 text-sm">
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-red-500 rounded mr-2"></div>
                    <span>High attention (disease indicators)</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-yellow-400 rounded mr-2"></div>
                    <span>Medium attention</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-blue-500 rounded mr-2"></div>
                    <span>Low attention</span>
                  </div>
                </div>
              </div>
            )} */}
          </div>
        )}

        {error && (
          <div className="mt-4 p-3 bg-red-100 border border-red-300 rounded-lg">
            <p className="text-red-600 font-semibold">{error}</p>
          </div>
        )}
      </div>

      {/* Crop Recommendation Button - Bottom Right */}
      <Link href="/crop-recommendation">
        <button className="fixed bottom-6 right-6 bg-emerald-600 text-white px-4 py-3 rounded-full shadow-lg hover:bg-emerald-700 transition-all flex items-center space-x-2 text-sm font-medium border border-emerald-500">
          <span>🌾</span>
          <span>Get Crop Recommendation</span>
        </button>
      </Link>
    </div>
  );
}
