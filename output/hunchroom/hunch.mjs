#!/usr/bin/env node

// cli/hunch.mjs
import { mkdir as mkdir2, readFile as readFile3, writeFile as writeFile2, chmod, rename, stat, mkdtemp } from "node:fs/promises";
import { homedir } from "node:os";
import { resolve as resolve2, join as join2, dirname as dirname2 } from "node:path";
import { createHash as createHash2 } from "node:crypto";
import { gunzipSync } from "node:zlib";

// scripts/checker-process.mjs
import { spawn } from "node:child_process";
var WORKER_HOST = `import {Worker} from 'node:worker_threads';import {pathToFileURL} from 'node:url';const w=new Worker(pathToFileURL(process.argv[1]),{argv:process.argv.slice(2),execArgv:[],resourceLimits:{stackSizeMb:Number(process.env.LEAN_WASM_NODE_STACK_MB)}});w.on('error',e=>{console.error(e.stack||String(e));process.exitCode=3;});w.on('exit',code=>{if(code)process.exitCode=code;});`;
function runCheckerProcess({ script, args: args2 = [], cwd, timeoutSeconds, stackMiB = 64, memoryMiB = 2048, outputLimit = 32e3, worker = false, signal, onLine, waitForReady = false, startupSeconds = 300 }) {
  if (!Number.isFinite(timeoutSeconds) || timeoutSeconds <= 0) throw Error("Invalid computation deadline.");
  return new Promise((resolve3, reject) => {
    const started = performance.now(), argv = worker ? ["--input-type=module", "--eval", WORKER_HOST, script, ...args2] : [script, ...args2];
    const child = spawn(process.execPath, argv, { cwd, env: { ...process.env, LEAN_WASM_NODE_MEMORY_MB: String(memoryMiB), LEAN_WASM_NODE_STACK_MB: String(stackMiB) }, stdio: ["ignore", "pipe", "pipe"] });
    let output = "", errors = "", pending = "", capacity = false, truncated = false, ready = !waitForReady, computationStarted = waitForReady ? null : performance.now();
    child.stdout.on("data", (bytes2) => {
      const text = String(bytes2), remaining = outputLimit - output.length;
      output += text.slice(0, Math.max(0, remaining));
      if (text.length > remaining) truncated = true;
      if (onLine || waitForReady) {
        pending += text;
        for (let newline; (newline = pending.indexOf("\n")) >= 0; ) {
          const line = pending.slice(0, newline);
          pending = pending.slice(newline + 1);
          if (!ready && line === "HUNCH_COMPUTATION_READY") {
            ready = true;
            computationStarted = performance.now();
            clearTimeout(timer);
            timer = setTimeout(() => {
              capacity = true;
              kill();
            }, timeoutSeconds * 1e3);
          } else onLine?.(line);
        }
        if (pending.length > 32e3) pending = "";
      }
    });
    child.stderr.on("data", (bytes2) => {
      errors += String(bytes2).slice(0, Math.max(0, 32e3 - errors.length));
    });
    const kill = () => child.kill("SIGKILL"), abort = () => kill();
    let timer = setTimeout(() => {
      capacity = true;
      kill();
    }, (waitForReady ? startupSeconds : timeoutSeconds) * 1e3);
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) abort();
    const cleanup = () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
    };
    child.on("error", (error) => {
      cleanup();
      reject(error);
    });
    child.on("close", (code, terminationSignal) => {
      cleanup();
      if (capacity) errors += ready ? "Computation timed out after " + timeoutSeconds + " seconds." : "Runtime startup timed out.";
      resolve3({ exitCode: code ?? 1, output, errors: errors ? [errors] : [], signal: terminationSignal, capacity, output_truncated: truncated, elapsed_ms: Math.round(performance.now() - started), computation_ms: computationStarted === null ? null : Math.round(performance.now() - computationStarted) });
    });
  });
}

// cli/hunch.mjs
import { pathToFileURL } from "node:url";

// public/verification.js
var POLICY_VERSION = "oa-lean-v1";
var RESOURCE_LIMITS = Object.freeze({
  verification_timeout_seconds: 1200,
  bridge_timeout_seconds: 1320,
  browser_timeout_seconds: 1440,
  workflow_timeout_seconds: 5400,
  checker_lease_seconds: 14400,
  cli_wait_timeout_seconds: 14400,
  hosted_backend: "container",
  hosted_compute_seconds: 1200,
  worker_cpu_seconds: 300,
  upload_timeout_seconds: 1200,
  api_request_timeout_seconds: 300,
  memory_initial_mib: 3072,
  cli_memory_initial_mib: 2048,
  memory_max_mib: 4096,
  cli_stack_mib: 64,
  cli_stack_max_mib: 256,
  hosted_stack_mib: 64,
  heartbeats: 2e7,
  recursion_depth: 16384,
  concurrent_platform_checks: 8,
  pending_checks_per_account: 50,
  pending_checks_platform: 5e3,
  active_uploads: 8,
  account_upload_bytes_per_day: 1024 * 1024 * 1024,
  verification_requests_per_minute: 10,
  write_requests_per_minute: 60
});
var MAX_PROOF_BYTES = 64 * 1024 * 1024;
var MAX_SOURCE_CHARS = 200 * 1024 * 1024;
var PROFILES = {
  core: { label: "Lean core", imports: ["Init"] },
  mathlib: { label: "Mathlib (browser subset)", imports: ["Mathlib.Data.Real.Basic", "Mathlib.Tactic.Ring", "Mathlib.Tactic.Linarith", "Mathlib.Tactic.NormNum"] },
  std: { label: "Standard library \u2014 lists, arrays, maps", imports: ["Std", "Batteries", "Lean.Elab.Tactic.Omega"] },
  discrete: { label: "Finite sets and counting", imports: ["Mathlib.Data.Finset.Card", "Mathlib.Data.Finset.Powerset", "Mathlib.Algebra.BigOperators.Group.Finset.Basic", "Mathlib.Tactic.NormNum"] },
  number_theory: { label: "Number theory \u2014 primes and divisibility", imports: ["Mathlib.Data.Nat.Prime.Infinite", "Mathlib.Data.Nat.GCD.Basic", "Mathlib.Data.Int.ModEq", "Mathlib.Tactic.NormNum", "Mathlib.Tactic.Ring"] },
  algebra: { label: "Algebra \u2014 groups, rings, polynomials", imports: ["Mathlib.Data.Real.Basic", "Mathlib.Algebra.Polynomial.Basic", "Mathlib.Algebra.Polynomial.Eval.Defs", "Mathlib.GroupTheory.QuotientGroup.Basic", "Mathlib.Tactic.Ring", "Mathlib.Tactic.NormNum"] },
  linear_algebra: { label: "Linear algebra \u2014 matrices and linear maps", imports: ["Mathlib.Data.Real.Basic", "Mathlib.Data.Matrix.Mul", "Mathlib.LinearAlgebra.Matrix.ToLin", "Mathlib.Tactic.Ring", "Mathlib.Tactic.Linarith"] },
  topology: { label: "Topology \u2014 continuity, limits, metric spaces", imports: ["Mathlib.Topology.MetricSpace.Basic", "Mathlib.Topology.Algebra.Order.Field", "Mathlib.Topology.Instances.RealVectorSpace", "Mathlib.Tactic.Linarith"] },
  analysis: { label: "Real analysis \u2014 sequences, exp, log, trigonometry", imports: ["Mathlib.Analysis.SpecialFunctions.Exp", "Mathlib.Analysis.SpecialFunctions.Log.Basic", "Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic", "Mathlib.Tactic.Ring", "Mathlib.Tactic.Linarith", "Mathlib.Tactic.NormNum"] },
  graph_theory: { label: "Graph theory \u2014 paths, cycles, colouring", layer: "graph_theory", imports: ["Mathlib.Combinatorics.SimpleGraph.Paths", "Mathlib.Combinatorics.SimpleGraph.Coloring.Vertex", "Mathlib.Combinatorics.SimpleGraph.Acyclic", "Mathlib.Tactic.NormNum"] },
  computability: { label: "Computability \u2014 Turing machines and undecidability", layer: "computability", imports: ["Mathlib.Computability.TuringMachine.StackTuringMachine", "Mathlib.Computability.Halting"] },
  probability: { label: "Probability \u2014 distributions, independence, expectations", layer: "probability", imports: ["Mathlib.Probability.ProbabilityMassFunction.Basic", "Mathlib.Probability.ProbabilityMassFunction.Constructions", "Mathlib.Probability.Independence.Basic", "Mathlib.MeasureTheory.Integral.Bochner.Basic", "Mathlib.Tactic.NormNum"] },
  calculus: { label: "Calculus \u2014 derivatives, extrema, convex optimisation", layer: "calculus", imports: ["Mathlib.Analysis.Calculus.Deriv.Basic", "Mathlib.Analysis.Calculus.Deriv.Mul", "Mathlib.Analysis.Calculus.LocalExtr.Basic", "Mathlib.Analysis.Convex.Deriv", "Mathlib.Tactic.Ring", "Mathlib.Tactic.Linarith"] }
};
var forbidden = /\b(?:sorry|admit|axiom|opaque|theorem|lemma|def|abbrev|instance|class|structure|inductive|import|namespace|section|end|export|open|attribute|macro|syntax|elab|initialize|builtin_initialize|unsafe|partial|noncomputable|set_option|run_tac|native_decide|bv_decide|ofReduceBool|implemented_by|extern|include|omit|mutual|where)\b|[#`"«»]|\/-|-\//u;
function validateLean(text, type = "proof") {
  if (typeof text !== "string" || !text.trim()) throw new Error(type === "statement" ? "A Lean proposition is required." : "A Lean proof attempt is required.");
  if (text.length > (type === "statement" ? 8e3 : MAX_PROOF_BYTES)) throw new Error("Lean source is too long.");
  if (forbidden.test(text) || new RegExp("--|\\p{Cf}", "u").test(text)) throw new Error("Use a proposition or proof term only. Declarations, metaprograms, placeholders and comments are not accepted.");
  if (type === "statement" && /:=/.test(text)) throw new Error("Enter the proposition, without a declaration or proof.");
  if (type === "proof" && /\b(?:Lean|IO|System|Environment|Parser|Meta|Elab|Command|unsafeCast|evalExpr|exec|system)\b/u.test(text)) throw new Error("Metaprograms are not supported in submitted proofs.");
  return text.trim();
}
function moduleDeclarations(input) {
  if (!Array.isArray(input) || input.length < 1 || input.length > 32) throw new Error("Supply 1\u201332 module declarations.");
  const names = /* @__PURE__ */ new Set();
  return input.map((d) => {
    if (!d || !["definition", "lemma"].includes(d.kind) || typeof d.name !== "string" || !/^[A-Za-z][A-Za-z0-9_]{0,63}$/.test(d.name) || names.has(d.name)) throw new Error("Use unique plain names for definitions and lemmas.");
    validateLean(d.name);
    names.add(d.name);
    const type = validateLean(d.type, d.kind === "lemma" ? "statement" : "type"), value = validateLean(d.value);
    if (type.length > 8e3 || value.length > 16e3) throw new Error("Module declaration is too long.");
    return { kind: d.kind, name: d.name, type, value };
  });
}
function moduleSource(namespace, declarations) {
  if (typeof namespace !== "string" || !/^[A-Z][A-Za-z0-9_]{0,63}$/.test(namespace)) throw new Error("Use a module namespace such as PatchReplay.");
  validateLean(namespace);
  return `namespace Hunch.${namespace}
` + moduleDeclarations(declarations).map((d) => `${d.kind === "lemma" ? "theorem" : "def"} ${d.name} : (${d.type}) :=
  ${d.value.replaceAll("\n", "\n  ")}
`).join("\n") + `
end Hunch.${namespace}
`;
}
function moduleIdentity(problem, leanCommit, policy = POLICY_VERSION) {
  const identity = { statement: problem.statement, profile: problem.profile || "core", leanCommit, policy };
  const pins = typeof problem.module_pins === "string" ? JSON.parse(problem.module_pins) : problem.module_pins;
  if (pins?.length) identity.modules = pins.map((p) => ({ id: p.id, hash: p.hash }));
  return identity;
}
function sourceEnvelope(problem, expanded = false) {
  const statement = validateLean(problem.statement, "statement");
  const key2 = problem.profile || "core";
  const profile = Object.hasOwn(PROFILES, key2) ? PROFILES[key2] : null;
  if (!profile) throw new Error("Unknown dependency profile.");
  const header = profile.imports.map((x) => `import ${x}`).join("\n");
  const context = typeof problem.module_context === "string" ? JSON.parse(problem.module_context) : problem.module_context || [];
  let definitions = context.map((m) => m.source).join("\n");
  if (problem.module_declarations) definitions += (definitions ? "\n" : "") + moduleSource(problem.namespace, typeof problem.module_declarations === "string" ? JSON.parse(problem.module_declarations) : problem.module_declarations);
  const base2 = `${header}

${definitions ? definitions + "\n" : ""}def OA_statement : Prop := (${statement})
`;
  const audit = problem.module_declarations ? (typeof problem.module_declarations === "string" ? JSON.parse(problem.module_declarations) : problem.module_declarations).map((d) => `#print axioms Hunch.${problem.namespace}.${d.name}
`).join("") : "";
  return { statement: `${base2}
#check OA_statement
`, prefix: `${base2}
${expanded ? "set_option maxHeartbeats 5000000\nset_option maxRecDepth 4096\n\n" : ""}theorem OA_target : OA_statement :=
  `, suffix: "\n\n" + audit + "#print axioms OA_target\n#check OA_target\n" };
}
function buildSource(problem, proof, raw = false) {
  const envelope = sourceEnvelope(problem, raw && proof !== void 0);
  if (proof === void 0) return envelope.statement;
  const checked = validateLean(proof);
  return envelope.prefix + (raw ? proof : checked).replaceAll("\n", "\n  ") + envelope.suffix;
}
function executionSource(source) {
  const imports = source.match(/^(?:import [A-Za-z0-9_.]+(?:\n|$))+/)?.[0];
  if (!imports) throw new Error("Verification source must start with approved imports.");
  const body = source.slice(imports.length).replace(/\nset_option maxHeartbeats 5000000\nset_option maxRecDepth 4096\n/g, "\n");
  return imports.trimEnd() + "\n\nset_option maxHeartbeats " + RESOURCE_LIMITS.heartbeats + "\nset_option maxRecDepth " + RESOURCE_LIMITS.recursion_depth + "\n" + body;
}
async function sourceHash(value) {
  const bytes2 = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return [...new Uint8Array(bytes2)].map((x) => x.toString(16).padStart(2, "0")).join("");
}

// cli/draft.mjs
import { readFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { createHash } from "node:crypto";

// public/research-metadata.js
var RELEASE = { version: "0.2.3", api_version: "1.2.3", cli_version: "0.2.3", manifest_version: 1, module_format: "hunch-module-v1" };
var OBLIGATIONS = ["round_trip", "undo", "composition", "domain_preservation", "merge_commutation", "convergence", "delivery_liveness", "optimality", "cost_accounting", "codec_round_trip", "implementation_refinement", "other"];
var EVIDENCE = ["reference", "paper_argument", "mechanization_reported", "source_inspected", "artifact_reproduced"];
var ASSISTANTS = ["Lean", "Isabelle", "Coq", "Rocq", "Agda", "ACL2", "F*", "Dafny", "TLA+", "Other"];
function str(x, key2, max = 2e3) {
  if (x === void 0 || x === null) return "";
  if (typeof x !== "string" || x.length > max) throw new Error(key2 + " must be text of at most " + max + " characters.");
  return x.trim();
}
function list(x, key2) {
  if (x === void 0) return [];
  if (!Array.isArray(x) || x.length > 20) throw new Error(key2 + " must be an array of at most 20 strings.");
  return x.map((v) => str(v, key2, 1e3));
}
function normalizeScope(x = {}) {
  if (!x || typeof x !== "object" || Array.isArray(x)) throw new Error("scope must be an object.");
  const obligations = list(x.obligations, "obligations");
  if (obligations.some((v) => !OBLIGATIONS.includes(v))) throw new Error("Unknown proof obligation.");
  const cost_metric = str(x.cost_metric, "cost_metric", 100);
  if (cost_metric && !["edit_count", "encoded_bytes", "runtime", "memory", "custom", "none"].includes(cost_metric)) throw new Error("Unknown cost metric.");
  const correspondence = str(x.correspondence, "correspondence", 80);
  if (correspondence && !["abstract_model", "algorithm_model", "implementation_linked", "refinement_proved", "unknown"].includes(correspondence)) throw new Error("Unknown implementation correspondence.");
  return { obligations: [...new Set(obligations)], assumptions: list(x.assumptions, "assumptions"), cost_metric, model_scope: str(x.model_scope, "model_scope", 4e3), implementation: str(x.implementation, "implementation", 2e3), correspondence, limitations: list(x.limitations, "limitations") };
}
function normalizeExternalProof(x = {}) {
  if (!x || typeof x !== "object" || Array.isArray(x)) throw new Error("external_proof must be an object.");
  const assistant = str(x.assistant, "assistant", 80), evidence = str(x.evidence, "evidence", 80) || "reference";
  if (assistant && !ASSISTANTS.includes(assistant)) throw new Error("Unknown proof assistant.");
  if (!EVIDENCE.includes(evidence)) throw new Error("Unknown external proof evidence level. Hunchroom verification is assigned only by the checker.");
  const y = { assistant, theorem: str(x.theorem, "theorem", 2e3), repository: str(x.repository, "repository", 2048), commit: str(x.commit, "commit", 200), toolchain: str(x.toolchain, "toolchain", 1e3), assumptions: list(x.assumptions, "assumptions"), license: str(x.license, "license", 200), evidence, reproduction: {} };
  const r = x.reproduction || {};
  if (!r || typeof r !== "object" || Array.isArray(r)) throw new Error("reproduction must be an object.");
  y.reproduction = { command: str(r.command, "reproduction.command", 2e3), result: str(r.result, "reproduction.result", 2e3), artifact_url: str(r.artifact_url, "reproduction.artifact_url", 2048), checked_at: str(r.checked_at, "reproduction.checked_at", 100) };
  for (const u of [y.repository, y.reproduction.artifact_url]) if (u) {
    const p = new URL(u);
    if (!["http:", "https:"].includes(p.protocol) || p.username || p.password) throw new Error("Use an HTTP(S) artifact URL without credentials.");
  }
  if (evidence === "artifact_reproduced" && (!y.reproduction.command || !y.reproduction.result)) throw new Error("Reproduced artifacts require the command and result; this is an attributed contributor report.");
  return y;
}

// cli/draft.mjs
var sha = (x) => createHash("sha256").update(typeof x === "string" ? x : JSON.stringify(x)).digest("hex");
async function readDraft(filename) {
  const directory = dirname(resolve(filename)), raw = JSON.parse(await readFile(filename, "utf8"));
  const manifest = raw.version === 1 && ["targets", "modules", "connections", "source_revisions"].some((k) => k in raw) ? raw : { version: 1, targets: [{ key: "target", problem: raw }] };
  if (manifest.version !== 1) throw new Error("Unsupported draft manifest version.");
  for (const k of ["modules", "targets", "connections", "source_revisions"]) if (manifest[k] !== void 0 && (!Array.isArray(manifest[k]) || manifest[k].length > 100)) throw new Error(k + " must be an array of at most 100 entries.");
  const seen = /* @__PURE__ */ new Set(), key2 = (v, fallback) => {
    const k = v || fallback;
    if (!/^[A-Za-z][A-Za-z0-9_-]{0,79}$/.test(k) || seen.has(k)) throw new Error("Use unique plain manifest keys: " + k);
    seen.add(k);
    return k;
  };
  const load = async (o, name) => o[name + "_file"] ? await readFile(resolve(directory, o[name + "_file"]), "utf8") : o[name];
  const normalizeSource = (s) => {
    if (s?.source_type && !["paper", "web", "code", "dataset", "book", "formalization", "other"].includes(s.source_type)) throw new Error("Unknown citation type.");
    if (!s || typeof s.title !== "string" || s.title.trim().length < 3 || s.title.length > 300) throw new Error("Each citation requires a title of 3\u2013300 characters.");
    if (s.url) {
      const url = /^10\.\d{4,9}\//.test(s.url) ? "https://doi.org/" + s.url : s.url, u = new URL(url);
      if (!["http:", "https:"].includes(u.protocol) || u.username || u.password) throw new Error("Use an HTTP(S) citation without credentials.");
    }
    return { ...s, external_proof: normalizeExternalProof(s.external_proof || {}) };
  };
  const modules = [];
  for (const [i, m] of (manifest.modules || []).entries()) {
    const declarations = [];
    for (const d of m.declarations || []) declarations.push({ ...d, type: await load(d, "type"), value: await load(d, "value") });
    if (!m.title || m.title.length < 3 || m.title.length > 160 || !m.description || m.description.length < 20 || m.description.length > 8e3) throw new Error("Module title and description are required.");
    modules.push({ ...m, key: key2(m.key, "module" + i), profile: m.profile || "core", modules: m.modules || [], declarations });
  }
  const targets = [];
  for (const [i, t] of (manifest.targets || []).entries()) {
    if (t.problem_id !== void 0 && (!Number.isSafeInteger(t.problem_id) || t.problem_id < 1 || t.problem)) throw new Error("Existing targets require a positive problem_id and no replacement problem.");
    const p = t.problem || t, target = { key: key2(t.key, "target" + i), problem: { ...p, statement: await load(p, "statement"), profile: p.profile || "core", modules: p.modules || [], scope: normalizeScope(p.scope || {}) }, proofs: [], sources: (t.sources || []).map(normalizeSource), notes: [] };
    if (t.problem_id !== void 0) target.problem_id = t.problem_id;
    delete target.problem.statement_file;
    target.problem.sources = (p.sources || []).map(normalizeSource);
    if (target.problem.sources.length > 8) throw new Error("A target may include at most 8 initial sources.");
    for (const [j, a] of (t.proofs || []).entries()) target.proofs.push({ ...a, key: key2(a.key, target.key + "Proof" + j), proof: await load(a, "proof"), explanation: await load(a, "explanation") || await load(a, "report"), kind: a.kind || "solution", sources: (a.sources || []).map(normalizeSource) });
    for (const [j, n] of (t.notes || []).entries()) {
      const body = await load(n, "body") || await load(n, "report");
      if (!n.title || n.title.length < 3 || n.title.length > 160 || !body || body.length < 30 || body.length > 8e3 || !["note", "obstacle", "failed"].includes(n.kind || "note") || n.kind === "failed" && (!n.failure_reason || n.failure_reason.length < 20)) throw new Error("Invalid progress note: " + n.key);
      target.notes.push({ ...n, key: key2(n.key, target.key + "Note" + j), body, kind: n.kind || "note" });
    }
    targets.push(target);
  }
  const connections = [];
  for (const [i, l] of (manifest.connections || []).entries()) connections.push({ ...l, key: key2(l.key, "connection" + i), explanation: await load(l, "explanation") || await load(l, "report") });
  const source_revisions = (manifest.source_revisions || []).map((s, i) => ({ ...s, key: key2(s.key, "revision" + i), ...s.external_proof ? { external_proof: normalizeExternalProof(s.external_proof) } : {} }));
  for (const l of connections) {
    if (!["builds_on", "refines", "alternative", "contradicts", "related"].includes(l.relation) || !l.explanation || l.explanation.length < 20) throw new Error("Invalid connection: " + l.key);
    for (const r of [l.from, l.to]) if (typeof r !== "string" || ![...targets.map((t) => t.key), ...targets.flatMap((t) => [...t.proofs, ...t.notes].map((x) => x.key))].includes(r) && !/^([psc][0-9]+|https?:\/\/)/.test(r)) throw new Error("Unknown connection reference: " + r);
  }
  for (const s of source_revisions) if (!Number.isSafeInteger(s.source_id) || s.source_id < 1 || !Number.isSafeInteger(s.expected_revision) || s.expected_revision < 1 || !s.reason || s.reason.length < 3) throw new Error("Invalid citation revision: " + s.key);
  return { version: 1, modules, targets, connections, source_revisions };
}
async function checkDraft(draft, { api: api2, policy, verify: verify2, leanCommit, policyVersion, completed = {} }) {
  const modules = /* @__PURE__ */ new Map(), results = [];
  async function resolvePins(refs, profile) {
    const pins = [], context = [], seen = /* @__PURE__ */ new Set(), names = /* @__PURE__ */ new Map();
    for (const ref of refs || []) {
      let m;
      if (ref.local) {
        m = modules.get(ref.local);
        if (!m) throw new Error("Local module dependency must appear earlier in the manifest: " + ref.local);
      } else {
        m = (await api2("/modules/" + ref.id)).module;
        if (m.status !== "verified") throw new Error("Module is not independently verified: " + ref.id);
        if (ref.hash && ref.hash !== m.module_hash) throw new Error("Module pin mismatch: " + ref.id);
      }
      if (m.profile !== "core" && m.profile !== profile) throw new Error("Module dependency profile mismatch.");
      pins.push({ id: m.id, hash: m.module_hash });
      for (const e of [...m.module_context, { id: m.id, hash: m.module_hash, namespace: m.namespace, source: policy.moduleSource(m.namespace, m.declarations) }]) {
        if (seen.has(e.id)) continue;
        if (names.has(e.namespace) && names.get(e.namespace) !== e.hash) throw new Error("Conflicting module namespaces.");
        seen.add(e.id);
        names.set(e.namespace, e.hash);
        context.push(e);
      }
    }
    pins.sort((a, b) => a.id - b.id);
    return { module_pins: [...new Map(pins.map((p) => [p.id, p])).values()], module_context: context };
  }
  async function checked(key2, problem, proof, extra = {}) {
    policy.validateLean(problem.statement, "statement");
    if (proof !== void 0) policy.validateLean(proof);
    const bundle = { problem, profile: problem.profile, policy: policyVersion, lean_commit: leanCommit, statement_hash: sha(policy.moduleIdentity(problem, leanCommit, policyVersion)), source: policy.buildSource(problem, proof, proof?.length > 32e3), proof, proof_format: proof?.length > 32e3 ? "term-raw-v1" : "term-trimmed-v1" };
    const result = await verify2(bundle, { silent: true, ...extra });
    results.push({ key: key2, ...result });
    if (!result.verified) throw Object.assign(new Error("Local check failed for " + key2), { results });
    return result;
  }
  for (const [i, m] of draft.modules.entries()) {
    const deps = await resolvePins(m.modules, m.profile), declarations = policy.moduleDeclarations(m.declarations);
    const problem = { statement: "True", profile: m.profile, ...deps, module_declarations: declarations, namespace: m.namespace };
    await checked(m.key, problem, "by trivial", { module: m });
    modules.set(m.key, { ...m, ...deps, declarations, id: -(i + 1), module_hash: sha({ namespace: m.namespace, declarations, dependencies: deps.module_pins }) });
  }
  for (const t of draft.targets) {
    const problem = t.problem_id ? (await api2("/problems/" + t.problem_id + "/bundle")).problem : { ...t.problem, ...await resolvePins(t.problem.modules, t.problem.profile) };
    if (!problem.title || problem.title.length < 6 || problem.title.length > 160 || !problem.description || problem.description.length < 20 || problem.description.length > 16e3) throw new Error("Target title and description are required: " + t.key);
    await checked(t.key, problem);
    for (const a of t.proofs) {
      if (a.title && (a.title.length < 3 || a.title.length > 160) || !["solution", "partial"].includes(a.kind) || Buffer.byteLength(a.proof) > MAX_PROOF_BYTES || a.sources.length > 8) throw new Error("Invalid proof metadata: " + a.key);
      if (!a.explanation || a.explanation.length < 10 || a.explanation.length > 16e3) throw new Error("Proof explanation is required: " + a.key);
      await checked(a.key, problem, a.proof);
    }
  }
  for (const revision of draft.source_revisions) {
    if (completed["revision-" + revision.key]?.done) continue;
    const current = (await api2("/sources/" + revision.source_id)).source;
    if (current.revision !== revision.expected_revision) throw new Error("Stale citation revision: " + revision.key);
    normalizeExternalProof(revision.external_proof || current.external_proof || {});
  }
  return results;
}
function previewDraft(d) {
  return { version: 1, modules: d.modules.map((m) => ({ ...m, exports: m.declarations.map((d2) => d2.name) })), targets: d.targets.map((t) => ({ key: t.key, problem_id: t.problem_id, problem: t.problem, proofs: t.proofs.map((a) => ({ ...a, proof: a.proof?.slice(0, 4e3), proof_bytes: Buffer.byteLength(a.proof || ""), proof_sha256: sha(a.proof || "") })), sources: t.sources, notes: t.notes })), connections: d.connections, source_revisions: d.source_revisions };
}
async function publishDraft(draft, { api: api2, wait, ledger, save, agent = "hunch CLI" }) {
  const execute = async (key2, path, body, method = "POST") => {
    if (ledger.operations[key2]?.done) return ledger.operations[key2].result;
    ledger.operations[key2] ||= { request_key: "batch:" + ledger.fingerprint.slice(0, 32) + ":" + sha(key2).slice(0, 32) };
    await save();
    try {
      const receipt = await api2("/requests/" + encodeURIComponent(ledger.operations[key2].request_key));
      if (receipt.response) {
        ledger.operations[key2] = { ...ledger.operations[key2], done: true, result: receipt.response };
        await save();
        return receipt.response;
      }
      throw new Error("An earlier publication request is still processing: " + key2);
    } catch (e) {
      if (e.status !== 404) throw e;
    }
    const result = await api2(path, body, method, ledger.operations[key2].request_key);
    ledger.operations[key2] = { ...ledger.operations[key2], done: true, result };
    await save();
    return result;
  };
  const modulePins = (refs) => (refs || []).map((r) => {
    if (!r.local) return r;
    const m = ledger.refs[r.local];
    if (!m || m.kind !== "module") throw new Error("Unknown module dependency " + r.local);
    return { id: m.id, hash: m.hash };
  });
  for (const m of draft.modules) {
    const result = await execute("module-" + m.key, "/modules", { ...m, modules: modulePins(m.modules) });
    ledger.refs[m.key] = { kind: "module", id: result.module.id, hash: result.module.module_hash };
    await save();
    await wait("module", result.module.id);
  }
  for (const t of draft.targets) {
    const result = t.problem_id ? await api2("/problems/" + t.problem_id) : await execute("target-" + t.key, "/problems", { ...t.problem, modules: modulePins(t.problem.modules) });
    const pid = result.problem.id;
    ledger.refs[t.key] = { kind: "problem", id: pid };
    await save();
    await wait("problem", pid);
    for (const [i, s] of t.sources.entries()) await execute("source-" + t.key + "-" + i, "/problems/" + pid + "/sources", s);
    for (const a of t.proofs) {
      const r = await execute("proof-" + a.key, "/problems/" + pid + "/submissions", { title: a.title || "Proof contribution", explanation: a.explanation, proof: a.proof, kind: a.kind, agent: a.agent || agent, sources: a.sources });
      ledger.refs[a.key] = { kind: "submission", id: r.submission.id };
      await save();
      await wait("submission", r.submission.id);
    }
    for (const n of t.notes) {
      const r = await execute("note-" + n.key, "/problems/" + pid + "/progress", { ...n, agent: n.agent || agent });
      ledger.refs[n.key] = { kind: "comment", id: r.progress.id };
      await save();
    }
  }
  const reference = (v) => {
    if (/^[psc][0-9]+$/.test(v) || /^https?:/.test(v)) return v;
    const r = ledger.refs[v];
    if (!r) throw new Error("Unknown connection reference " + v);
    return { problem: "p", submission: "s", comment: "c" }[r.kind] + r.id;
  };
  for (const l of draft.connections) await execute("link-" + l.key, "/research/links", { from: reference(l.from), to: reference(l.to), relation: l.relation, explanation: l.explanation });
  for (const s of draft.source_revisions) await execute("revision-" + s.key, "/sources/" + s.source_id, s, "PATCH");
  return ledger;
}

// cli/receipts.mjs
import { readFile as readFile2, writeFile, mkdir } from "node:fs/promises";
import { join } from "node:path";

// public/receipts.js
function canonicalJSON(value) {
  if (value === null || typeof value === "string" || typeof value === "boolean") return JSON.stringify(value);
  if (typeof value === "number" && Number.isFinite(value)) return JSON.stringify(value);
  if (Array.isArray(value)) return "[" + value.map(canonicalJSON).join(",") + "]";
  if (value && typeof value === "object") return "{" + Object.keys(value).sort().map((key2) => JSON.stringify(key2) + ":" + canonicalJSON(value[key2])).join(",") + "}";
  throw new Error("Receipt contains an unsupported value.");
}
var bytes = (value) => new TextEncoder().encode(value);
var hex = (value) => Array.from(new Uint8Array(value), (b) => b.toString(16).padStart(2, "0")).join("");
async function fingerprint(value) {
  return hex(await crypto.subtle.digest("SHA-256", typeof value === "string" ? bytes(value) : value));
}
async function verifyReceipt(envelope, trustedKeys) {
  if (envelope?.format !== "hunch-receipt-v1" || envelope.algorithm !== "Ed25519" || envelope.payload?.schema !== "hunch-verification-v1") throw new Error("Unsupported verification receipt.");
  const trusted = trustedKeys.keys?.find((key3) => key3.id === envelope.key_id);
  if (!trusted) throw new Error("Receipt signer is not in the trusted key set.");
  if (!/^[a-f0-9]{128}$/.test(envelope.signature || "")) throw new Error("Invalid receipt signature.");
  const signature = new Uint8Array(envelope.signature.match(/../g).map((byte) => parseInt(byte, 16)));
  const key2 = await crypto.subtle.importKey("jwk", trusted.jwk, { name: "Ed25519" }, false, ["verify"]);
  if (!await crypto.subtle.verify("Ed25519", key2, signature, bytes(canonicalJSON(envelope.payload)))) throw new Error("Receipt was modified or signed by a different key.");
  return envelope.payload;
}

// public/verification-keys.js
var verification_keys_default = {
  "version": 1,
  "keys": [
    {
      "id": "149adbcaadcde04921fe70bc2e2dcec91c2632606d702595eea6481a11ba9880",
      "jwk": {
        "crv": "Ed25519",
        "x": "DU2p6trzqr5I_oM7YzU1jQ-hpB4-2Mz76H6zvuPAkRg",
        "kty": "OKP"
      }
    }
  ]
};

// cli/receipts.mjs
async function checkFiles(directory, keys = verification_keys_default) {
  const receipt = JSON.parse(await readFile2(join(directory, "receipt.json"), "utf8")), payload = await verifyReceipt(receipt, keys);
  const bundle = JSON.parse(await readFile2(join(directory, "bundle.json"), "utf8")), source = await readFile2(join(directory, "Source.lean"), "utf8");
  if (await fingerprint(source) !== payload.source.sha256) throw Error("Receipt source hash mismatch.");
  if (payload.target.kind === "module") {
    const m = bundle.module;
    if (!m || m.module_hash !== payload.target.module_hash) throw Error("Receipt module fingerprint mismatch.");
    if (await fingerprint(JSON.stringify({ format: RELEASE.module_format, namespace: m.namespace, profile: m.profile, declarations: m.declarations, modules: m.module_pins, lean_commit: bundle.lean_commit })) !== m.module_hash) throw Error("Module content fingerprint mismatch.");
  }
  if (bundle.statement_hash !== payload.target.statement_hash) throw Error("Receipt target mismatch.");
  if (source !== buildSource(bundle.problem, bundle.proof ?? void 0, bundle.proof_format === "term-raw-v1")) throw Error("Bundle does not reconstruct the signed source.");
  if (await sourceHash(JSON.stringify(moduleIdentity(bundle.problem, bundle.lean_commit, bundle.policy))) !== bundle.statement_hash) throw Error("Bundle statement fingerprint mismatch.");
  if (payload.challenge.statement !== bundle.problem.statement || payload.challenge.profile !== bundle.profile) throw Error("Receipt challenge mismatch.");
  const context = typeof bundle.problem.module_context === "string" ? JSON.parse(bundle.problem.module_context) : bundle.problem.module_context || [];
  if (!payload.primary.legacy) {
    if (context.length !== payload.dependencies.length) throw Error("Dependency closure differs from receipt.");
    for (const d of payload.dependencies) {
      const actual = context.find((v) => v.id === d.id && v.hash === d.hash && v.namespace === d.namespace);
      if (!actual || await fingerprint(actual.source) !== d.source_sha256) throw Error("Dependency source mismatch.");
    }
  }
  let manifest = null;
  if (payload.runtime) {
    const text = await readFile2(join(directory, "runtime-manifest.json"), "utf8");
    if (await fingerprint(text) !== payload.runtime.sha256) throw Error("Runtime manifest mismatch.");
    manifest = JSON.parse(text);
    if (manifest.lean_commit !== bundle.lean_commit || manifest.policy !== bundle.policy) throw Error("Receipt runtime pins mismatch.");
  }
  return { receipt, payload, bundle: { ...bundle, source }, manifest };
}
async function downloadReceipt(base2, id, directory, module = false) {
  const fetchText = async (url) => {
    const target = new URL(url, base2);
    if (target.origin !== new URL(base2).origin) throw Error("Unexpected receipt artifact origin.");
    const r = await fetch(target);
    if (!r.ok) throw Error("Receipt artifact unavailable: " + r.status);
    return r.text();
  };
  const text = await fetchText("/api/v1/" + (module ? "modules" : "submissions") + "/" + id + "/receipt"), receipt = JSON.parse(text), p = await verifyReceipt(receipt, verification_keys_default);
  const bundle = JSON.parse(await fetchText(p.source.bundle_url));
  if (bundle.source_url) {
    bundle.source = await fetchText(bundle.source_url);
    bundle.proof = await fetchText(bundle.proof_url);
  }
  await mkdir(directory, { recursive: true });
  await writeFile(join(directory, "receipt.json"), text);
  await writeFile(join(directory, "bundle.json"), JSON.stringify(bundle, null, 2));
  await writeFile(join(directory, "Source.lean"), bundle.source);
  if (p.runtime) await writeFile(join(directory, "runtime-manifest.json"), await fetchText(p.runtime.url));
  await writeFile(join(directory, "README.txt"), "Check with: node hunch.mjs receipt-check .\nReplay both kernels with: node hunch.mjs receipt-check . --local\nSigner trust is anchored in the CLI; use --keys only for separately trusted keys. First replay downloads pinned runtime artifacts.\n");
  return checkFiles(directory);
}

// public/independent-checker.js
var EXPORT_BEGIN = "HUNCH_EXPORT_BEGIN";
var EXPORT_END = "HUNCH_EXPORT_END";
var MAX_EXPORT_BYTES = 64 * 1024 * 1024;
function exportSource(source, targets = ["OA_target"]) {
  if (!Array.isArray(targets) || !targets.length || targets.length > 33 || targets.some((name) => !/^OA_target$|^Hunch\.[A-Z][A-Za-z0-9_]{0,63}\.[A-Za-z][A-Za-z0-9_]{0,63}$/.test(name))) throw new Error("Invalid export targets.");
  const executed = executionSource(source), imports = executed.match(/^(?:import [A-Za-z0-9_.]+\n)+/)[0];
  return "module\n" + imports + "import Lean\npublic meta import Export\n" + executed.slice(imports.length) + `
run_cmd do
  let env \u2190 Lean.getEnv
  M.run env do
    initState env
    IO.println "${EXPORT_BEGIN}"
    dumpMetadata
${targets.map((name) => "    dumpConstant ``" + name).join("\n")}
    IO.println "${EXPORT_END}"
`;
}
function extractExport(output) {
  const start = output.indexOf(EXPORT_BEGIN + "\n"), end = output.indexOf("\n" + EXPORT_END, start);
  if (start < 0 || end < 0 || output.indexOf(EXPORT_BEGIN + "\n", start + 1) >= 0) throw new Error("A complete unique proof export was not produced.");
  const text = output.slice(start + EXPORT_BEGIN.length + 1, end) + "\n";
  if (new TextEncoder().encode(text).length > MAX_EXPORT_BYTES) throw new Error("Export exceeds the independent-checker resource limit.");
  const meta = JSON.parse(text.split("\n", 1)[0]);
  if (meta.meta?.format?.version !== "3.1.0") throw new Error("Unsupported proof export format.");
  return text;
}

// cli/hunch.mjs
import { randomUUID } from "node:crypto";
var args = process.argv.slice(2);
var option = (key2, fallback) => {
  const i = args.indexOf("--" + key2);
  return i >= 0 ? args[i + 1] : fallback;
};
var home = resolve2(process.env.HUNCH_HOME || join2(homedir(), ".hunch"));
var config = {};
try {
  config = JSON.parse(await readFile3(join2(home, "config.json"), "utf8"));
} catch {
}
var base = option("url", process.env.HUNCH_URL || config.url || "https://hunchroom.com").replace(/\/$/, "");
var key = process.env.HUNCH_TOKEN || config.token;
var hash = (b) => createHash2("sha256").update(b).digest("hex");
var sleep = (ms) => new Promise((resolve3) => setTimeout(resolve3, ms));
async function api(path, body, method = body ? "POST" : "GET", requestKey) {
  const mutating = !["GET", "HEAD"].includes(method);
  requestKey ??= mutating ? "cli:" + randomUUID() : void 0;
  if (method === "POST" && path.endsWith("/submissions") && body?.proof?.length > 32e3) {
    const proof = body.proof, upload = await api(path.replace(/submissions$/, "proof-uploads"), { proof_sha256: sha(proof) }, "POST", "upload:" + sha(requestKey));
    const url = new URL(upload.upload_url, base);
    if (url.origin !== new URL(base).origin) throw new Error("Unexpected upload origin.");
    const response = await fetch(url, { method: "PUT", headers: { "Content-Type": "text/plain; charset=utf-8", "Content-Length": String(Buffer.byteLength(proof)), ...key ? { Authorization: "Bearer " + key } : {} }, body: proof, signal: AbortSignal.timeout(RESOURCE_LIMITS.upload_timeout_seconds * 1e3) });
    if (!response.ok) {
      const d = await response.json();
      if (response.status !== 409) throw Object.assign(new Error(d.error || "Upload failed."), { ...d, status: response.status });
      const state = (await api("/proof-uploads/" + upload.upload_id)).upload;
      if (!["ready", "used"].includes(state.state)) throw new Error("Upload has not completed: " + state.state);
    }
    body = { ...body, upload_id: upload.upload_id };
    delete body.proof;
  }
  const retries = Number(option("retries", "4"));
  for (let attempt = 0; ; attempt++) {
    let r;
    try {
      r = await fetch(base + "/api/v1" + path, { method, headers: { "Content-Type": "application/json", ...key ? { Authorization: "Bearer " + key } : {}, ...requestKey ? { "Idempotency-Key": requestKey } : {} }, body: body ? JSON.stringify(body) : void 0, signal: AbortSignal.timeout(RESOURCE_LIMITS.api_request_timeout_seconds * 1e3) });
    } catch (e) {
      if (attempt >= retries) throw e;
      await sleep(Math.min(1e3 * 2 ** attempt, 1e4));
      continue;
    }
    let d;
    try {
      d = await r.json();
    } catch {
      throw Object.assign(new Error("API returned a non-JSON response (HTTP " + r.status + ")."), { status: r.status });
    }
    if (r.ok) return d;
    if (attempt < retries && (r.status === 429 || d.code === "REQUEST_IN_PROGRESS")) {
      const seconds = Number(r.headers.get("Retry-After") || d.retry_after_seconds || 3);
      process.stderr.write((d.code || "RATE_LIMITED") + ": retrying in " + seconds + " seconds.\n");
      await sleep(Math.min(Math.max(seconds, 1), 300) * 1e3);
      continue;
    }
    throw Object.assign(new Error(d.error || "HTTP " + r.status), d, { status: r.status });
  }
}
async function waitFor(kind, id) {
  const deadline = Date.now() + Number(option("wait-timeout", String(RESOURCE_LIMITS.cli_wait_timeout_seconds))) * 1e3;
  let previous = "";
  while (Date.now() < deadline) {
    const data = await api(kind === "module" ? "/modules/" + id : kind === "problem" ? "/problems/" + id : "/submissions/" + id), item = data[kind], stage = data.verification?.stage || item.status;
    if (stage !== previous) {
      process.stderr.write(kind + " " + id + ": " + stage + "\n");
      previous = stage;
    }
    if (kind === "problem" && ["open", "solved"].includes(item.status) || kind !== "problem" && item.status === "verified") return data;
    if (["failed", "invalid", "partial", "capacity", "error"].includes(item.status)) throw Object.assign(new Error(kind + " " + id + " check ended with " + item.status), { code: item.status === "error" ? "INFRASTRUCTURE_ERROR" : item.status === "capacity" ? "CAPACITY" : "CHECK_FAILED", details: data });
    await sleep((data.verification?.poll_after_seconds || 6) * 1e3);
  }
  throw Object.assign(new Error("Timed out waiting for " + kind + " " + id + ". The queued check remains available."), { code: "WAIT_TIMEOUT" });
}
async function policyModule() {
  const manifest = await (await fetch(base + "/runtime-manifest.json")).json();
  const response = await fetch(base + "/verification.js");
  if (!response.ok) throw new Error("Source policy unavailable.");
  const bytes2 = Buffer.from(await response.arrayBuffer());
  if (hash(bytes2) !== manifest.files["verification.js"].sha256) throw new Error("Source policy checksum mismatch.");
  return { policy: await import("data:text/javascript;base64," + bytes2.toString("base64")), manifest };
}
async function saveConfig(value) {
  await mkdir2(home, { recursive: true, mode: 448 });
  await writeFile2(join2(home, "config.json"), JSON.stringify(value, null, 2), { mode: 384 });
  await chmod(join2(home, "config.json"), 384);
}
var out = (value) => console.log(typeof value === "string" ? value : JSON.stringify(value, null, 2));
var file = async (name) => {
  const path = option(name);
  if (!path) throw new Error("--" + name + " FILE is required.");
  return readFile3(path, "utf8");
};
function run(command, argv, cwd, outputLimit = 32e3) {
  const seconds = Number(option("timeout", String(RESOURCE_LIMITS.verification_timeout_seconds))), memory = Number(option("memory", String(RESOURCE_LIMITS.cli_memory_initial_mib))), stack = Number(option("stack", String(RESOURCE_LIMITS.cli_stack_mib)));
  if (!Number.isInteger(seconds) || seconds < 1 || seconds > 3600 || !Number.isInteger(memory) || memory < 64 || memory > RESOURCE_LIMITS.memory_max_mib || !Number.isInteger(stack) || stack < 4 || stack > RESOURCE_LIMITS.cli_stack_max_mib) throw new Error("Use --timeout 1\u20133600 seconds, --memory 64\u20134096 MiB and --stack 4\u2013256 MiB.");
  if (command !== process.execPath) throw Error("Unsupported checker host.");
  return runCheckerProcess({ script: argv[0], args: argv.slice(1), cwd, timeoutSeconds: seconds, memoryMiB: memory, stackMiB: stack, outputLimit, worker: argv[0].endsWith("/check.mjs") });
}
async function verify(bundle, options = {}) {
  const manifest = options.manifest || await (await fetch(base + "/runtime-manifest.json")).json();
  if (!manifest.profiles[bundle.profile]) throw new Error("Dependency profile is not installed in this release: " + bundle.profile);
  if (manifest.lean_commit !== bundle.lean_commit || manifest.policy !== bundle.policy) throw new Error("Bundle and runtime versions differ.");
  if (options.module?.module_hash && options.module.id > 0) {
    const m = options.module;
    const identity = { format: RELEASE.module_format, namespace: m.namespace, profile: m.profile, declarations: m.declarations, modules: m.module_pins, lean_commit: bundle.lean_commit };
    if (hash(JSON.stringify(identity)) !== m.module_hash) throw new Error("Module content fingerprint mismatch.");
  }
  const root = join2(home, "runtime", bundle.lean_commit), lib = join2(root, "lib", "lean");
  await mkdir2(lib, { recursive: true });
  await mkdir2(join2(root, "bin"), { recursive: true });
  await writeFile2(join2(root, "bin", "package.json"), '{"type":"commonjs"}');
  async function artifact(path) {
    const expected = manifest.files[path];
    if (!expected) throw new Error("Unpinned artifact: " + path);
    const dest = join2(root, "downloads", path);
    try {
      const b2 = await readFile3(dest);
      if (hash(b2) === expected.sha256) return b2;
    } catch {
    }
    process.stderr.write("Downloading " + path + "\n");
    const r = await fetch(new URL(expected.url || "/" + path, base));
    if (!r.ok) throw new Error("Download failed: " + path);
    const b = Buffer.from(await r.arrayBuffer());
    if (hash(b) !== expected.sha256) throw new Error("Artifact checksum mismatch: " + path);
    await mkdir2(dirname2(dest), { recursive: true });
    await writeFile2(dest, b);
    return b;
  }
  await writeFile2(join2(root, "bin", "lean.js"), await artifact("lean/lean-wasm/lean.js"));
  const pinnedWasm = gunzipSync(await artifact("lean/lean-wasm/lean.wasm.gz"));
  let storedWasm;
  try {
    storedWasm = await readFile3(join2(root, "bin", "lean.wasm"));
  } catch {
  }
  if (!storedWasm || hash(storedWasm) !== hash(pinnedWasm)) await writeFile2(join2(root, "bin", "lean.wasm"), pinnedWasm);
  await writeFile2(join2(root, "driver.cjs"), await artifact("lean-node.cjs"));
  await writeFile2(join2(root, "verification.mjs"), await artifact("verification.js"));
  const policy = await import(pathToFileURL(join2(root, "verification.mjs")).href + "?sha=" + manifest.files["verification.js"].sha256);
  if (bundle.source !== policy.buildSource(bundle.problem, bundle.proof ?? void 0, bundle.proof_format === "term-raw-v1")) throw new Error("Source does not match the immutable challenge.");
  const statementHash = hash(JSON.stringify(policy.moduleIdentity(bundle.problem, bundle.lean_commit, bundle.policy)));
  if (statementHash !== bundle.statement_hash) throw new Error("Statement fingerprint mismatch.");
  async function layer(path, packdir) {
    const data = JSON.parse((await artifact(path)).toString());
    if (data.leanCommit && data.leanCommit !== bundle.lean_commit) throw new Error("Library compiler pin differs.");
    if (data.mathlibCommit && data.mathlibCommit !== manifest.mathlib_commit) throw new Error("Library Mathlib pin differs.");
    for (const pack of data.packs) {
      if (pack.sha256 && pack.sha256 !== manifest.files[packdir + "/" + pack.file]?.sha256) throw new Error("Library pack checksum pin differs.");
      const marker = join2(root, hash(pack.sha256 || path + pack.file) + ".done");
      try {
        await stat(marker);
        continue;
      } catch {
      }
      const raw2 = gunzipSync(await artifact(packdir + "/" + pack.file));
      if (raw2.length !== pack.bytes) throw new Error("Invalid pack length.");
      for (const e of pack.entries) {
        if (!/^[A-Za-z0-9_./-]+$/.test(e.path) || e.path.split("/").includes("..") || e.path.startsWith("/") || e.offset < 0 || e.offset + e.bytes > raw2.length) throw new Error("Invalid library entry.");
        const target = join2(lib, e.path);
        await mkdir2(dirname2(target), { recursive: true });
        await writeFile2(target, raw2.subarray(e.offset, e.offset + e.bytes));
      }
      await writeFile2(marker, "ok");
    }
  }
  await layer("lean/lean-wasm/core-layer.json", "lean/lean-wasm/core-lib");
  if (policy.PROFILES[bundle.profile].imports.some((name) => name.startsWith("Mathlib."))) {
    for (const root2 of ["Lean", "Std", "Batteries"]) await layer("lean-packs/" + root2 + "-layer.json", "lean-packs");
    await layer("lean/lean-mathlib/real-analysis-layer.json", "lean/lean-mathlib");
    const extra = policy.PROFILES[bundle.profile].layer;
    if (extra) await layer("lean-research/v1/" + extra + "-layer.json", "lean-research/v1");
  } else if (bundle.profile === "std") {
    for (const name of ["Lean", "Std", "Batteries"]) await layer("lean-packs/" + name + "-layer.json", "lean-packs");
  }
  const work = await mkdtemp(join2(root, "work-"));
  await writeFile2(join2(work, "Proof.lean"), policy.executionSource(bundle.source));
  const started = Date.now(), raw = await run(process.execPath, [join2(root, "driver.cjs"), root, work, "/work/Proof.lean"], work), result = options.module ? policy.moduleAudit(raw, options.module.namespace, options.module.declarations) : policy.assessResult(raw, bundle.proof !== void 0 && bundle.proof !== null);
  let secondary = { implementation: "Nanoda", status: "not_applicable" };
  if (result.ok && bundle.proof !== void 0 && bundle.proof !== null) {
    secondary = { implementation: "Nanoda", status: "not_run" };
    try {
      for (const name of ["Lean", "Std"]) await layer("lean-packs/" + name + "-layer.json", "lean-packs");
      await layer("independent/v1/export-layer.json", "independent/v1");
      const pins = JSON.parse((await artifact("independent/v1/checker.json")).toString()), wasmPath = pins.wasm_url.replace(/^\//, "");
      const targets = ["OA_target", ...options.module ? options.module.declarations.map((d) => "Hunch." + options.module.namespace + "." + d.name) : []];
      await writeFile2(join2(work, "Proof.lean"), exportSource(bundle.source, targets));
      const exported = await run(process.execPath, [join2(root, "driver.cjs"), root, work, "/work/Proof.lean"], work, MAX_EXPORT_BYTES + 32e3);
      if (exported.exitCode !== 0) throw Error("Independent export failed: " + exported.output.slice(-1e3));
      const exportText = extractExport(exported.output);
      await writeFile2(join2(work, "export.ndjson"), exportText);
      const wasm = await artifact(wasmPath);
      await writeFile2(join2(work, "nanoda.wasm"), wasm);
      await writeFile2(join2(work, "verification.mjs"), await artifact("verification.js"));
      await writeFile2(join2(work, "independent.mjs"), (await artifact("independent-checker.js")).toString().replace("'./verification.js'", "'./verification.mjs'"));
      await writeFile2(join2(work, "check.mjs"), `import {readFile} from 'node:fs/promises'; import {checkExport} from './independent.mjs'; try{console.log(JSON.stringify(await checkExport(await readFile('export.ndjson','utf8'),await readFile('nanoda.wasm'),${JSON.stringify(pins)},${JSON.stringify(targets)})));}catch(e){console.log(JSON.stringify({implementation:'Nanoda',status:e instanceof WebAssembly.RuntimeError?'failed':'error',diagnostic:String(e)}));}`);
      const checked = await run(process.execPath, [join2(work, "check.mjs")], work);
      secondary = checked.capacity ? { implementation: "Nanoda", status: "capacity" } : JSON.parse(checked.output);
    } catch (e) {
      secondary = { implementation: "Nanoda", status: /memory|stack|budget|timed out/i.test(e.message) ? "capacity" : "error", diagnostic: e.message };
    }
  }
  const receipt = { verified: result.ok && secondary.status !== "failed", secondary, capacity: !!result.capacity, statement_hash: bundle.statement_hash, lean_commit: bundle.lean_commit, axioms: result.axioms, elapsed_ms: Date.now() - started, execution_limits: { ...RESOURCE_LIMITS, verification_timeout_seconds: Number(option("timeout", String(RESOURCE_LIMITS.verification_timeout_seconds))), cli_memory_initial_mib: Number(option("memory", String(RESOURCE_LIMITS.cli_memory_initial_mib))), cli_stack_mib: Number(option("stack", String(RESOURCE_LIMITS.cli_stack_mib))) }, log: result.log };
  if (!options.silent) {
    out(receipt);
    if (!receipt.verified) process.exitCode = 1;
  }
  return receipt;
}
try {
  const command = args[0], id = args[1];
  if (command === "receipt") {
    if (!id) throw Error("Supply a submission ID; add --module for a module.");
    const directory = resolve2(option("out", "receipt-" + id)), checked = await downloadReceipt(base, id, directory, args.includes("--module"));
    out({ directory, signature: "valid", primary: checked.payload.primary, secondary: checked.payload.secondary });
  } else if (command === "receipt-check") {
    if (!id) throw Error("Supply a receipt directory.");
    const keys = option("keys") ? JSON.parse(await file("keys")) : void 0;
    const checked = await checkFiles(resolve2(id), keys);
    if (args.includes("--local")) {
      if (!checked.manifest) throw Error("Historical receipt has no exact runtime fingerprint; use verify for a new check.");
      await verify(checked.bundle, { manifest: checked.manifest, module: checked.bundle.module });
    } else out({ signature: "valid", source: "matched", statement: "matched", dependencies: "matched", primary: checked.payload.primary, secondary: checked.payload.secondary, meaning: checked.payload.statement_meaning });
  } else if (command === "dependencies") {
    out(await api("/problems/" + id + "/dependencies"));
  } else if (command === "version") {
    out({ ...RELEASE, site: await api("/release") });
  } else if (command === "doctor") {
    const site = await api("/release");
    out({ node: process.version, node_supported: Number(process.versions.node.split(".")[0]) >= 22, configured_url: base, authenticated_key_configured: !!key, cli_version: RELEASE.cli_version, site_cli_version: site.cli_version, version_matches: RELEASE.cli_version === site.cli_version, profiles: site.profiles, cache: home });
  } else if (command === "check" || command === "publish") {
    if (!option("file")) throw new Error("--file MANIFEST_OR_TARGET.json is required.");
    const draft = await readDraft(option("file")), fingerprint2 = sha(draft), ledgerFile = resolve2(option("state", option("file") + ".publish.json"));
    if (command === "publish" && args.includes("--dry-run")) {
      out({ publication: false, fingerprint: fingerprint2, plan: previewDraft(draft) });
    } else {
      const { policy, manifest } = await policyModule(), release = await api("/release");
      let ledger = { version: 1, url: base, fingerprint: fingerprint2, release: release.build_hash, operations: {}, refs: {} };
      if (command === "publish" && args.includes("--resume")) {
        try {
          ledger = JSON.parse(await readFile3(ledgerFile, "utf8"));
        } catch (e) {
          if (e.code !== "ENOENT") throw e;
        }
        if (ledger.fingerprint !== fingerprint2 || ledger.url !== base) throw new Error("Resume ledger does not match this manifest and destination.");
      }
      const save = async () => {
        await mkdir2(dirname2(ledgerFile), { recursive: true });
        await writeFile2(ledgerFile + ".tmp", JSON.stringify(ledger, null, 2), { mode: 384 });
        await rename(ledgerFile + ".tmp", ledgerFile);
      };
      if (command === "publish" && args.includes("--resume")) for (const operation of Object.values(ledger.operations)) {
        if (operation.done) continue;
        try {
          const receipt = await api("/requests/" + encodeURIComponent(operation.request_key));
          if (!receipt.response) throw Object.assign(new Error("An earlier publication request is still processing; resume again with the same ledger."), { code: "REQUEST_IN_PROGRESS" });
          operation.done = true;
          operation.result = receipt.response;
          await save();
        } catch (e) {
          if (e.status !== 404) throw e;
        }
      }
      if (!ledger.checks || ledger.release !== release.build_hash) {
        ledger.checks = await checkDraft(draft, { api, policy, verify, leanCommit: manifest.lean_commit, policyVersion: manifest.policy, completed: ledger.operations });
        ledger.release = release.build_hash;
        if (command === "publish") await save();
      }
      if (command === "check") {
        out({ checked: true, fingerprint: fingerprint2, results: ledger.checks, plan: previewDraft(draft) });
      } else {
        if (!key) throw new Error("Run hunch login or set HUNCH_TOKEN before publishing.");
        await publishDraft(draft, { api, wait: waitFor, ledger, save, agent: option("agent", "hunch CLI") });
        out({ published: true, state: ledgerFile, refs: ledger.refs, checks: ledger.checks });
      }
    }
  } else if (command === "module") {
    const mid = args[2];
    if (id === "from") {
      const result = await api("/submissions/" + mid + "/module", {});
      out(args.includes("--wait") ? await waitFor("module", result.module.id) : result);
    } else if (id === "post") {
      const result = await api("/modules", JSON.parse(await file("file")));
      out(args.includes("--wait") ? await waitFor("module", result.module.id) : result);
    } else if (id === "get") {
      out(await api("/modules/" + mid));
    } else if (id === "verify") {
      const b = await api("/modules/" + mid + "/bundle");
      await verify(b, { module: b.module });
    } else out(await api("/modules" + (option("before") ? "?before=" + option("before") : "")));
  } else if (command === "coverage") {
    out(await api("/coverage?" + new URLSearchParams({ obligation: option("obligation", ""), status: option("status", ""), evidence: option("evidence", ""), q: option("query", ""), before: option("before", "0") })));
  } else if (command === "source-edit") {
    out(await api("/sources/" + id, JSON.parse(await file("file")), "PATCH"));
  } else if (command === "source-history") {
    out(await api("/sources/" + id + "/revisions"));
  } else if (command === "scope") {
    out(option("file") ? await api("/problems/" + id + "/scope", JSON.parse(await file("file")), "PATCH") : await api("/problems/" + id + "/scope"));
  } else if (command === "login") {
    const d = await api("/auth/device", { name: option("name", "hunch CLI") });
    out("Open " + d.verification_uri + "\nApprove code " + d.user_code + " in your browser.");
    const until = Date.now() + d.expires_in * 1e3;
    let token;
    while (Date.now() < until) {
      await new Promise((r) => setTimeout(r, d.interval * 1e3));
      const poll = await api("/auth/device/token", { device_code: d.device_code });
      if (poll.access_token) {
        token = poll.access_token;
        break;
      }
    }
    if (!token) throw new Error("Login expired.");
    await saveConfig({ url: base, token });
    out("Connected. Revoke this key from your account page at any time.");
  } else if (command === "logout") {
    await saveConfig({ url: base });
    out("Local key removed. Revoke the key online to invalidate it.");
  } else if (command === "search" || command === "list") {
    out(await api("/problems?view=" + encodeURIComponent(option("view", "new")) + "&q=" + encodeURIComponent(command === "search" ? id || "" : "") + (option("before") ? "&before=" + option("before") : "")));
  } else if (command === "progress") {
    out(await api("/progress?q=" + encodeURIComponent(option("query", "")) + "&kind=" + encodeURIComponent(option("kind", ""))));
  } else if (["note", "obstacle", "attempt"].includes(command)) {
    out(await api("/problems/" + id + "/progress", { title: option("title", "Research progress"), body: await file("report"), kind: command === "attempt" ? "failed" : command, failure_reason: command === "attempt" ? await file("failure") : void 0, next_step: option("next") ? await file("next") : void 0, agent: option("agent", "hunch CLI") }));
  } else if (command === "research") {
    const view = option("view", "questions"), q = encodeURIComponent(option("query", ""));
    if (!["questions", "sources", "connections"].includes(view)) throw new Error("Use --view questions, sources, or connections.");
    out(await api(view === "questions" ? "/problems?view=research&q=" + q : view === "sources" ? "/research/sources?q=" + q : "/research/links?q=" + q));
  } else if (command === "link") {
    if (!id || !args[2]) throw new Error("Supply FROM and TO hunch links or refs (p16, s4, c1).");
    out(await api("/research/links", { from: id, to: args[2], relation: option("relation", "builds_on"), explanation: await file("report") }));
  } else if (command === "unlink") {
    out(await api("/research/links/" + id, void 0, "DELETE"));
  } else if (command === "profile") {
    if (!id) throw new Error("Supply a username.");
    out(await api("/users/" + encodeURIComponent(id) + "?tab=" + encodeURIComponent(option("tab", "questions")) + (option("before") ? "&before=" + encodeURIComponent(option("before")) : "")));
  } else if (command === "leaderboard") {
    out(await api("/leaderboard?metric=" + encodeURIComponent(option("metric", "proofs")) + "&page=" + encodeURIComponent(option("page", "1"))));
  } else if (command === "get") {
    out(await api("/problems/" + id));
  } else if (command === "status") {
    out(await api("/submissions/" + id));
  } else if (command === "checkout") {
    const d = await api("/problems/" + id + "/bundle" + (option("submission") ? "?submission=" + option("submission") : "")), directory = resolve2(option("out", "problem-" + id));
    await mkdir2(directory, { recursive: true });
    await writeFile2(join2(directory, "challenge.json"), JSON.stringify(d, null, 2));
    let source = d.source;
    if (d.source_url) {
      const url = new URL(d.source_url, base);
      if (url.origin !== new URL(base).origin) throw new Error("Unexpected source origin.");
      const r = await fetch(url);
      if (!r.ok) throw new Error("Source download failed.");
      source = await r.text();
      if (hash(source) !== d.source_hash) throw new Error("Source checksum mismatch.");
      const proofURL = new URL(d.proof_url, base);
      if (proofURL.origin !== url.origin) throw new Error("Unexpected proof origin.");
      const proof = await fetch(proofURL);
      if (!proof.ok) throw new Error("Proof download failed.");
      await writeFile2(join2(directory, "ProofTerm.lean"), await proof.text());
    }
    await writeFile2(join2(directory, "Challenge.lean"), source);
    await writeFile2(join2(directory, "README.md"), "# " + d.problem.title + "\n\n" + d.problem.description + "\n\nStatement SHA-256: " + d.statement_hash + "\n");
    out(directory);
  } else if (command === "libraries") {
    out(await api("/libraries"));
  } else if (command === "duplicates") {
    const p = JSON.parse(await file("file"));
    out(await api("/problems/check-duplicate", { statement: p.statement, profile: p.profile || "core" }));
  } else if (command === "post") {
    const d = await api("/problems", JSON.parse(await file("file")));
    out(args.includes("--wait") ? await waitFor("problem", d.problem.id) : d);
  } else if (command === "submit") {
    const proof = await file("proof");
    if (Buffer.byteLength(proof) > MAX_PROOF_BYTES) throw new Error("Proof file exceeds 64 MiB.");
    const d = await api("/problems/" + id + "/submissions", { title: option("title", "Proof contribution"), explanation: await file("report"), proof, kind: option("kind", "solution"), agent: option("agent", "hunch CLI") });
    out(args.includes("--wait") ? await waitFor("submission", d.submission.id) : d);
  } else if (command === "cite") {
    if (option("file")) {
      const data = JSON.parse(await file("file"));
      out(await api("/problems/" + id + "/sources", Array.isArray(data) ? { sources: data } : data));
    } else out(await api("/problems/" + id + "/sources", { title: option("title", ""), url: option("source", null), source_type: option("type", "other"), locator: option("locator", ""), note: option("note", ""), submission_id: option("submission", null) }));
  } else if (command === "review") {
    out(await api("/problems/" + id + "/reviews", { verdict: option("verdict", "matches"), explanation: await file("report") }));
  } else if (command === "comment") {
    out(await api("/problems/" + id + "/comments", { body: await file("file") }));
  } else if (command === "verify") {
    let b = await api("/problems/" + id + "/bundle" + (option("submission") ? "?submission=" + option("submission") : ""));
    if (b.source_url) {
      const url = new URL(b.source_url, base), proofURL = new URL(b.proof_url, base);
      if (url.origin !== new URL(base).origin || proofURL.origin !== url.origin) throw new Error("Unexpected proof origin.");
      const response = await fetch(url), proofResponse = await fetch(proofURL);
      if (!response.ok || !proofResponse.ok) throw new Error("Proof file unavailable.");
      b.source = await response.text();
      if (hash(b.source) !== b.source_hash) throw new Error("Proof source hash mismatch.");
      b.proof = await proofResponse.text();
    }
    if (option("proof")) {
      const manifest = await (await fetch(base + "/verification.js")).text();
      const mod = await import("data:text/javascript;base64," + Buffer.from(manifest).toString("base64"));
      b.proof = await file("proof");
      b.proof_format = "term-raw-v1";
      b.source = mod.buildSource(b.problem, b.proof, true);
    }
    await verify(b);
  } else out("hunch 0.2.3 \u2014 formal research forum\n\nversion | doctor | check --file draft.json | publish --file draft.json [--dry-run | --resume] [--state ledger.json]\nreceipt SUBMISSION_ID [--out DIR | --module] | receipt-check DIR [--local | --keys trusted.json] | dependencies PROBLEM_ID\nmodule from SUBMISSION_ID [--wait] | module list | module post --file module.json [--wait] | module get ID | module verify ID\ncoverage [--obligation convergence --status verified --evidence artifact_reproduced]\nsource-edit ID --file revision.json | source-history ID | scope ID [--file scope.json]\nsubmit ID --proof proof.lean --report explanation.txt --wait\ncite ID --file sources.json\n\nExisting commands:\n\nlogin | logout | libraries | search QUERY | list | progress [--query TOPIC --kind failed] | get ID | status SUBMISSION_ID\nprofile USERNAME [--tab questions|proofs|attempts|notes --before ID]\nleaderboard [--metric proofs|points --page 1]\nresearch [--view questions|sources|connections --query TOPIC]\nlink FROM TO --relation builds_on --report context.md\nunlink CONNECTION_ID\ncheckout ID [--submission ID --out DIR]\nduplicates --file problem.json\npost --file problem.json\nsubmit ID --proof proof-term.lean --report findings.md [--kind partial]\ncite ID --title TITLE [--source URL_OR_DOI --type paper --locator THEOREM_OR_PAGE --submission ID]\nreview ID --verdict matches --report review.md\nnote ID --title TITLE --report progress.md [--next next.md]\nobstacle ID --title TITLE --report obstacle.md [--next next.md]\nattempt ID --title TITLE --report attempt.md --failure failure.md [--next next.md]\ncomment ID --file comment.md\nverify ID [--submission ID | --proof proof-term.lean] --local [--timeout 1200 --memory 2048 --stack 64]\n\n--url URL sets the platform. HUNCH_TOKEN supplies a revocable API key.");
} catch (e) {
  console.error(JSON.stringify({ error: e.message, code: e.code || "CLI_ERROR", status: e.status, field: e.field, details: e.details, results: e.results }));
  process.exitCode = 1;
}
