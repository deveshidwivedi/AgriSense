import Navbar from "./components/Navbar";

export default function Home() {
  return (
    <>
      <Navbar />
      <div className="flex flex-col items-center justify-center h-screen text-center">
        <h1 className="text-4xl font-bold">Welcome to Agrisense 🌱</h1>
        <p className="mt-4 text-lg">
          AI-powered farming assistant to detect plant diseases and recommend crops.
        </p>
      </div>
    </>
  );
}
