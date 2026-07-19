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
  };
}
