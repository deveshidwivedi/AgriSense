"use client";
import { useState } from "react";
import axios from "axios";

export default function CropRecommendation() {
  const [ph, setPh] = useState("");
  const [nitrogen, setNitrogen] = useState("");
  const [phosphorus, setPhosphorus] = useState("");
  const [potassium, setPotassium] = useState("");
  const [temperature, setTemperature] = useState("");
  const [humidity, setHumidity] = useState("");
  const [rainfall, setRainfall] = useState("");
  const [recommendations, setRecommendations] = useState<string>("");
  const [loading, setLoading] = useState(false);

  const handleRecommend = async () => {
    if (!ph || !nitrogen || !phosphorus || !potassium || !temperature || !humidity || !rainfall) {
      setRecommendations("Please enter all fields.");
      return;
    }
    setLoading(true);
    try {
      const response = await axios.post("http://127.0.0.1:8000/api/crop-recommendation/", {
        ph,
        N: nitrogen,
        P: phosphorus,
        K: potassium,
        temperature,
        humidity,
        rainfall,
      });
      setRecommendations(response.data.recommended_crop);
    } catch (error) {
      setRecommendations("Error fetching recommendations. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-center bg-gradient-to-br from-green-50 to-green-200 p-6">
      <h1 className="text-4xl font-extrabold text-green-800 drop-shadow-lg">
        Crop Recommendation 🌾
      </h1>
      <div className="mt-6 bg-white p-6 rounded-xl shadow-lg w-full max-w-lg">
        <div className="grid grid-cols-2 gap-4">
          <input type="text" value={ph} onChange={(e) => setPh(e.target.value)} placeholder="pH Level" className="border p-3 rounded-lg shadow-sm w-full" />
          <input type="text" value={nitrogen} onChange={(e) => setNitrogen(e.target.value)} placeholder="Nitrogen Level (N)" className="border p-3 rounded-lg shadow-sm w-full" />
          <input type="text" value={phosphorus} onChange={(e) => setPhosphorus(e.target.value)} placeholder="Phosphorus Level (P)" className="border p-3 rounded-lg shadow-sm w-full" />
          <input type="text" value={potassium} onChange={(e) => setPotassium(e.target.value)} placeholder="Potassium Level (K)" className="border p-3 rounded-lg shadow-sm w-full" />
          <input type="text" value={temperature} onChange={(e) => setTemperature(e.target.value)} placeholder="Temperature (°C)" className="border p-3 rounded-lg shadow-sm w-full" />
          <input type="text" value={humidity} onChange={(e) => setHumidity(e.target.value)} placeholder="Humidity (%)" className="border p-3 rounded-lg shadow-sm w-full" />
          <input type="text" value={rainfall} onChange={(e) => setRainfall(e.target.value)} placeholder="Rainfall (mm)" className="border p-3 rounded-lg shadow-sm w-full" />
        </div>
        <button 
          onClick={handleRecommend} 
          className="w-full bg-green-600 text-white px-6 py-3 mt-6 font-semibold rounded-xl shadow-md hover:bg-green-700 transition-all disabled:bg-gray-400 disabled:cursor-not-allowed"
          disabled={loading}
        >
          {loading ? "Fetching..." : "Get Recommendation"}
        </button>
        {recommendations && (
          <p className="mt-4 text-lg text-green-900 font-medium">
            Recommended Crop: <span className="font-bold">{recommendations}</span>
          </p>
        )}
      </div>
    </div>
  );
}
