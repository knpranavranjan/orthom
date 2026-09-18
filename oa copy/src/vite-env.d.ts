/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the AETHER-OA X-ray inference API, e.g. http://localhost:8000.
   *  Unset -> the app falls back to the offline FixtureBackend. */
  readonly VITE_XRAY_API?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
