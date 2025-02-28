"use client";
import { useState } from "react";
import axios from "axios";

export default function CropRecommendation() {
  const [soilType, setSoilType] = useState("");
  const [ph, setPh] = useState("");
  const [recommendations, setRecommendations] = useState<string>("");

  const handleRecommend = async () => {
    try {
      const response = await axios.post("http://127.0.0.1:8000/api/crop-recommendation/", {
        soil_type: soilType,
        ph: ph
      });
      setRecommendations(response.data.recommended_crops);
    } catch (error) {
      setRecommendations("Error fetching recommendations.");
    }
  };

  return (
    <div className="p-6 text-center">
      <h1 className="text-2xl font-bold">Crop Recommendation</h1>
      <input type="text" value={soilType} onChange={(e) => setSoilType(e.target.value)} placeholder="Soil Type" className="border p-2 m-2" />
      <input type="text" value={ph} onChange={(e) => setPh(e.target.value)} placeholder="pH Level" className="border p-2 m-2" />
      <button onClick={handleRecommend} className="bg-green-600 text-white px-4 py-2 mt-4">Get Recommendations</button>
      {recommendations && <p className="mt-4 text-lg">Recommended Crops: {recommendations}</p>}
    </div>
  );
}
