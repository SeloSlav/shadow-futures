/// <reference lib="webworker" />

import { simulateWorlds } from "./simulation";
import type { Scenario } from "./types";

type Request = { id: number; scenario: Scenario; count: number };

self.onmessage = (event: MessageEvent<Request>) => {
  const { id, scenario, count } = event.data;
  const worlds = simulateWorlds(scenario, count);
  self.postMessage({ id, worlds });
};

export {};
