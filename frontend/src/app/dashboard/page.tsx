"use client";
import { useEffect, useState } from "react";
import axios from "axios";

export default function Dashboard() {
  const [history, setHistory] = useState<any[]>([]);

  useEffect(() => {
    axios.get("http://127.0.0.1:8000/api/user-history/")
      .then(response => setHistory(response.data))
      .catch(() => setHistory([]));
  }, []);

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold">Dashboard</h1>
      {history.length > 0 ? (
        <ul>
          {history.map((item, index) => (
            <li key={index} className="border p-2 mt-2">
              {item.date} - {item.disease || item.crop_recommendation}
            </li>
          ))}
        </ul>
      ) : (
        <p>No history found.</p>
      )}
    </div>
  );
}
