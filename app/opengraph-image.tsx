import { ImageResponse } from "next/og";

export const alt = "Shadow Futures — one realized market history surrounded by unrealized branches";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OpenGraphImage() {
  const branches = Array.from({ length: 11 }, (_, index) => index - 5);
  return new ImageResponse(
    (
      <div
        style={{
          display: "flex",
          position: "relative",
          width: "100%",
          height: "100%",
          overflow: "hidden",
          background: "#f3efe5",
          color: "#1d2427",
          padding: "76px 84px",
          fontFamily: "Georgia, serif",
        }}
      >
        <svg
          width="1200"
          height="630"
          viewBox="0 0 1200 630"
          style={{ position: "absolute", inset: 0 }}
        >
          {branches.map((branch) => (
            <path
              key={branch}
              d={`M 90 480 C 350 470, 540 ${350 + branch * 12}, 1140 ${310 + branch * 34}`}
              fill="none"
              stroke={branch === 1 ? "#ad5b36" : "#9aa0a1"}
              strokeWidth={branch === 1 ? 7 : 2}
              strokeOpacity={branch === 1 ? 1 : 0.32}
              strokeDasharray={branch === 1 ? undefined : "7 12"}
            />
          ))}
        </svg>
        <div style={{ display: "flex", position: "relative", flexDirection: "column" }}>
          <div
            style={{
              fontFamily: "Arial, sans-serif",
              fontSize: 19,
              letterSpacing: "0.18em",
              textTransform: "uppercase",
              color: "#ad5b36",
            }}
          >
            Martin Erlic · Interactive economics essay
          </div>
          <div style={{ display: "flex", marginTop: 26, fontSize: 104, letterSpacing: "-0.065em" }}>
            Shadow Futures
          </div>
          <div style={{ display: "flex", maxWidth: 760, marginTop: 24, fontSize: 38, lineHeight: 1.08 }}>
            A market can keep paying after it has stopped learning.
          </div>
        </div>
      </div>
    ),
    size,
  );
}
