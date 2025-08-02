import Navbar from "./components/Navbar";
import Link from "next/link";

export default function Home() {
  return (
    <>
      <Navbar />
      <div className="flex flex-col items-center justify-center h-screen text-center bg-gradient-to-br from-green-50 to-green-200 p-6">
        <h1 className="text-5xl font-extrabold text-green-800 drop-shadow-lg">
          Welcome to Agrisense 🌱
        </h1>
        <p className="mt-4 text-lg text-gray-700 max-w-xl">
          AI-powered farming assistant to detect plant diseases and recommend
          crops.
        </p>
        <Link href="/disease-detection">
          <button className="mt-6 px-6 py-3 bg-green-600 text-white font-semibold rounded-xl shadow-lg hover:bg-green-700 transition-all">
            Get Started
          </button>
        </Link>
      </div>
    </>
  );
}
