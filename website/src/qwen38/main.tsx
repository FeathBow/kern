import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "../styles.css";
import "./qwen38.css";
import Qwen38Page from "./Qwen38Page";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Qwen38Page />
  </StrictMode>,
);
