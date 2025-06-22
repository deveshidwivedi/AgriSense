"use client";

import { useState } from "react";

interface DetectionResult {
  predicted_disease: string;
  confidence: number;
  annotated_image?: string;
  detected_objects?: Array<{
    bbox: [number, number, number, number];
    confidence: number;
    class_id: number;
  }>;
  has_detections?: boolean;
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
    return disease.toLowerCase() === "healthy" ? "Healthy" : "Disease Detected";
  };

  const getStatusColor = (disease: string) => {
    return disease.toLowerCase() === "healthy"
      ? "text-green-600"
      : "text-red-600";
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-center bg-gradient-to-br from-green-50 to-green-200 p-6">
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
        {(previewUrl || prediction?.annotated_image) && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Original Image */}
            {previewUrl && (
              <div>
                <h3 className="text-lg font-semibold text-gray-800 mb-2">
                  Original Image
                </h3>
                <div className="border rounded-lg overflow-hidden">
                  <img
                    src={previewUrl}
                    alt="Original"
                    className="w-full h-64 object-cover"
                  />
                </div>
              </div>
            )}

            {/* Annotated Image with Bounding Boxes */}
            {prediction?.annotated_image && (
              <div>
                <h3 className="text-lg font-semibold text-gray-800 mb-2">
                  {prediction.has_detections
                    ? "Detected Disease Areas"
                    : "Analysis Result"}
                </h3>
                <div className="border rounded-lg overflow-hidden">
                  <img
                    src={prediction.annotated_image}
                    alt="Analysis Result"
                    className="w-full h-64 object-cover"
                  />
                </div>
                {prediction.has_detections && (
                  <p className="text-sm text-gray-600 mt-2">
                    Red boxes show detected disease areas
                  </p>
                )}
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
                <p className="text-gray-700">
                  Confidence:{" "}
                  <span className="font-medium">
                    {(prediction.confidence * 100).toFixed(2)}%
                  </span>
                </p>
              </div>

              {prediction.detected_objects &&
                prediction.detected_objects.length > 0 && (
                  <div>
                    <h4 className="font-semibold text-gray-800 mb-2">
                      Detection Details
                    </h4>
                    <p className="text-gray-700">
                      Affected Areas:{" "}
                      <span className="font-medium text-red-600">
                        {prediction.detected_objects.length} region(s) detected
                      </span>
                    </p>
                    <div className="mt-2 max-h-32 overflow-y-auto">
                      {prediction.detected_objects.map((obj, index) => (
                        <div key={index} className="text-sm text-gray-600 py-1">
                          Area {index + 1}: {(obj.confidence * 100).toFixed(1)}%
                          confidence
                        </div>
                      ))}
                    </div>
                  </div>
                )}
            </div>

            {/* Health Status Banner */}
            <div
              className={`mt-4 p-3 rounded-lg ${
                prediction.predicted_disease.toLowerCase() === "healthy"
                  ? "bg-green-100 border border-green-300"
                  : "bg-red-100 border border-red-300"
              }`}
            >
              <p
                className={`font-semibold ${getStatusColor(
                  prediction.predicted_disease
                )}`}
              >
                {prediction.predicted_disease.toLowerCase() === "healthy"
                  ? "✅ Plant appears healthy!"
                  : `⚠️ Disease detected: ${prediction.predicted_disease.replace(
                      /_/g,
                      " "
                    )}`}
              </p>
              {prediction.has_detections && (
                <p className="text-sm text-gray-700 mt-1">
                  Red bounding boxes highlight areas of concern that may require
                  treatment.
                </p>
              )}
            </div>
          </div>
        )}

        {error && (
          <div className="mt-4 p-3 bg-red-100 border border-red-300 rounded-lg">
            <p className="text-red-600 font-semibold">{error}</p>
          </div>
        )}
      </div>
    </div>
  );
}
