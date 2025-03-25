"use client";
import { useEffect, useState } from "react";
import axios from "axios";

export default function Dashboard() {
  const [history, setHistory] = useState<any[]>([]);

  useEffect(() => {
    axios
      .get("http://127.0.0.1:8000/api/user-history/")
      .then((response) => setHistory(response.data))
      .catch(() => setHistory([]));
  }, []);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-center bg-gradient-to-br from-green-50 to-green-200 p-6">
      <h1 className="text-4xl font-extrabold text-green-800 drop-shadow-lg">
        Dashboard
      </h1>
      <div className="mt-6 bg-white p-6 rounded-xl shadow-lg w-full max-w-lg">
        {history.length > 0 ? (
          <ul className="space-y-3">
            {history.map((item, index) => (
              <li
                key={index}
                className="border border-gray-300 p-3 rounded-lg shadow-sm bg-green-50"
              >
                <span className="font-semibold text-green-700">{item.date}</span> -{" "}
                <span className="text-gray-700">
                  {item.disease || item.crop_recommendation}
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-gray-600">No history found.</p>
        )}
      </div>
    </div>
  );
}
