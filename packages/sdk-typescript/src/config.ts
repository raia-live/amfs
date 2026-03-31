/**
 * Configuration types and defaults for AMFS TS SDK.
 */

import type { AMFSConfig } from "./models.js";

export function defaultConfig(): AMFSConfig {
  return {
    namespace: "default",
    layers: {
      primary: {
        adapter: "in-memory",
        options: {},
      },
    },
  };
}
