"use client";
import Link from "next/link";

export default function Navbar() {
  return (
    <nav className="bg-green-600 text-white p-4 flex justify-between">
      <h1 className="text-xl font-bold">Agrisense</h1>
      <div className="space-x-4">
        <Link href="/">Home</Link>
        <Link href="/disease-detection">Disease Detection</Link>
        <Link href="/crop-recommendation">Crop Recommendation</Link>
      </div>
    </nav>
  );
}
