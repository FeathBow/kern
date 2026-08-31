import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "../styles.css";
import "./schema.css";
import SchemaPage from "./SchemaPage";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <SchemaPage />
  </StrictMode>,
);
