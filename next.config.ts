import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  poweredByHeader: false,
  reactStrictMode: true,
  async redirects() {
    return [{ source: "/comparison-budget", destination: "/playground", permanent: true }];
  },
  experimental: {
    optimizePackageImports: ["framer-motion", "d3-array", "d3-scale", "d3-shape"],
  },
};

export default nextConfig;
