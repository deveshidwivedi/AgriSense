"use client";
import { useState } from "react";
import axios from "axios";

export default function CropRecommendation() {
  const [soilType, setSoilType] = useState("");
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
    if (!soilType || !ph || !nitrogen || !phosphorus || !potassium || !temperature || !humidity || !rainfall) {
      setRecommendations("Please enter all fields.");
      return;
    }
    setLoading(true);
    try {
      const response = await axios.post("http://127.0.0.1:8000/api/crop-recommendation/", {
        soil_type: soilType,
        ph: ph,
        nitrogen: nitrogen,
        phosphorus: phosphorus,
        potassium: potassium,
        temperature: temperature,
        humidity: humidity,
        rainfall: rainfall,
      });
      setRecommendations(response.data.recommended_crops);
    } catch (error) {
      setRecommendations("Error fetching recommendations. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8 text-center max-w-lg mx-auto bg-white shadow-xl rounded-xl border border-gray-200">
      <h1 className="text-3xl font-bold mb-6 text-gray-800">Crop Recommendation</h1>
      <div className="grid grid-cols-2 gap-4">
        <input type="text" value={soilType} onChange={(e) => setSoilType(e.target.value)} placeholder="Soil Type" className="border p-3 rounded-md shadow-sm w-full" />
        <input type="text" value={ph} onChange={(e) => setPh(e.target.value)} placeholder="pH Level" className="border p-3 rounded-md shadow-sm w-full" />
        <input type="text" value={nitrogen} onChange={(e) => setNitrogen(e.target.value)} placeholder="Nitrogen Level" className="border p-3 rounded-md shadow-sm w-full" />
        <input type="text" value={phosphorus} onChange={(e) => setPhosphorus(e.target.value)} placeholder="Phosphorus Level" className="border p-3 rounded-md shadow-sm w-full" />
        <input type="text" value={potassium} onChange={(e) => setPotassium(e.target.value)} placeholder="Potassium Level" className="border p-3 rounded-md shadow-sm w-full" />
        <input type="text" value={temperature} onChange={(e) => setTemperature(e.target.value)} placeholder="Temperature" className="border p-3 rounded-md shadow-sm w-full" />
        <input type="text" value={humidity} onChange={(e) => setHumidity(e.target.value)} placeholder="Humidity" className="border p-3 rounded-md shadow-sm w-full" />
        <input type="text" value={rainfall} onChange={(e) => setRainfall(e.target.value)} placeholder="Rainfall" className="border p-3 rounded-md shadow-sm w-full" />
      </div>
      <button 
        onClick={handleRecommend} 
        className="bg-green-600 text-white px-6 py-3 mt-6 rounded-lg hover:bg-green-700 transition duration-200 disabled:bg-gray-400 disabled:cursor-not-allowed"
        disabled={loading}
      >
        {loading ? "Fetching..." : "Get Recommendations"}
      </button>
      {recommendations && <p className="mt-6 text-lg font-semibold text-gray-700">{recommendations}</p>}
    </div>
  );
}