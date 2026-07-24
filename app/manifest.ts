import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Shadow Futures",
    short_name: "Shadow Futures",
    description:
      "An interactive explanation of contribution uncertainty in self-reinforcing markets.",
    start_url: "/",
    display: "standalone",
    background_color: "#f3efe5",
    theme_color: "#1d2427",
    icons: [
      { src: "/icon-192.png", sizes: "192x192", type: "image/png" },
      { src: "/icon-512.png", sizes: "512x512", type: "image/png" },
    ],
  };
}
