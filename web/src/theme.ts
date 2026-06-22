import { createTheme, type Theme } from "@mui/material/styles";

export type Mode = "light" | "dark";

// Theme factory — dark echoes the UVR desktop app; light is a clean counterpart.
export function makeTheme(mode: Mode): Theme {
  const dark = mode === "dark";
  return createTheme({
    palette: {
      mode,
      primary: { main: dark ? "#5c6bc0" : "#3949ab" },
      secondary: { main: dark ? "#26c6da" : "#0097a7" },
      background: dark
        ? { default: "#0f1117", paper: "#171a23" }
        : { default: "#f4f6fb", paper: "#ffffff" },
    },
    shape: { borderRadius: 10 },
  });
}
