import { spawn } from "node:child_process";
import { createReadStream } from "node:fs";
import { mkdtemp, readFile, rm, stat } from "node:fs/promises";
import { createServer } from "node:http";
import { createServer as createTcpServer } from "node:net";
import os from "node:os";
import path from "node:path";
import process from "node:process";

const root = path.resolve(import.meta.dirname, "..");
const profile = await mkdtemp(path.join(os.tmpdir(), "wgpu-matcher-parity-"));
const mime = {
  ".html": "text/html", ".js": "text/javascript", ".wgsl": "text/plain",
  ".json": "application/json", ".u8": "application/octet-stream",
};
const server = createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
    const file = path.resolve(root, `.${pathname}`);
    if (!file.startsWith(`${root}${path.sep}`) || !(await stat(file)).isFile()) throw new Error("not found");
    response.writeHead(200, { "content-type": mime[path.extname(file)] || "application/octet-stream" });
    createReadStream(file).pipe(response);
  } catch {
    response.writeHead(404).end("Not found");
  }
});
await new Promise((resolve, reject) => {
  server.once("error", reject);
  server.listen(0, "127.0.0.1", resolve);
});
const httpPort = server.address().port;
const portProbe = createTcpServer();
await new Promise((resolve, reject) => {
  portProbe.once("error", reject);
  portProbe.listen(0, "127.0.0.1", resolve);
});
const debugPort = portProbe.address().port;
await new Promise(resolve => portProbe.close(resolve));
let browser;

async function poll(read, predicate, timeoutMs, label) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const value = await read();
      if (predicate(value)) return value;
    } catch {}
    await new Promise(resolve => setTimeout(resolve, 250));
  }
  throw new Error(`Timed out waiting for ${label}`);
}

function pythonReport() {
  return new Promise((resolve, reject) => {
    const python = process.env.PYTHON || "python3";
    const child = spawn(python, [path.join(root, "tests/matcher_parity_python.py")], {
      cwd: root,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8").on("data", chunk => { stdout += chunk; });
    child.stderr.setEncoding("utf8").on("data", chunk => { stderr += chunk; });
    child.once("error", reject);
    child.once("exit", code => {
      if (code !== 0) return reject(new Error(`Python parity process failed (${code}): ${stderr}`));
      try { resolve(JSON.parse(stdout)); } catch (error) {
        reject(new Error(`Could not parse Python parity output: ${error}\n${stderr}\n${stdout}`));
      }
    });
  });
}

async function stopBrowser() {
  if (!browser || browser.exitCode !== null || browser.signalCode !== null) return;
  await new Promise(resolve => {
    const timeout = setTimeout(() => {
      browser.kill("SIGKILL");
      setTimeout(resolve, 250);
    }, 5000);
    browser.once("close", () => {
      clearTimeout(timeout);
      resolve();
    });
    browser.kill("SIGTERM");
  });
}

function compare(browserReport, python, contract) {
  const summary = {
    pythonAdapter: python.adapter,
    browserAdapter: browserReport.adapter,
    browserOptionValidation: browserReport.optionValidation,
    cases: {},
  };
  if (JSON.stringify(browserReport.transforms) !== JSON.stringify(python.transforms)) {
    throw new Error("Browser and Python affine fixture contracts differ");
  }
  if (Object.values(browserReport.optionValidation ?? {}).some(valid => valid !== true)) {
    throw new Error("Browser rejected-option checks did not pass");
  }
  for (const name of ["deepmatching", "horg", "cpm"]) {
    const spec = contract.algorithms[name];
    const browserOptions = browserReport.contractDefaults[name];
    const pythonOptions = python.apiDefaults[name];
    for (const [parameter, expected] of Object.entries(spec.defaults)) {
      const pythonKey = spec.option_keys[parameter].python;
      const javascriptKey = spec.option_keys[parameter].javascript;
      if (pythonOptions[pythonKey] !== expected || browserOptions[javascriptKey] !== expected) {
        throw new Error(`${name}.${parameter}: contract default mismatch ` +
          `(contract=${expected}, Python=${pythonOptions[pythonKey]}, browser=${browserOptions[javascriptKey]})`);
      }
    }
  }
  for (const name of ["deepmatching", "horg"]) {
    for (const caseName of ["translation", "affine", "manual_translation"]) {
      const jsMatches = browserReport.methods[name][caseName];
      const pyMatches = python.methods[name][caseName];
      const expectedCount = browserReport.seedCounts[caseName];
      if (jsMatches.length !== expectedCount || pyMatches.length !== expectedCount) {
        throw new Error(`${name}/${caseName}: expected ${expectedCount} matches, got JS=${jsMatches.length}, Python=${pyMatches.length}`);
      }
      let accepted = 0;
      let correct = 0;
      let maxDisplacementDifference = 0;
      let maxScoreDifference = 0;
      let maxDiagnosticDifference = 0;
      for (let index = 0; index < jsMatches.length; index++) {
        const js = jsMatches[index];
        const py = pyMatches[index];
        if (js.usable !== py.usable) {
          throw new Error(`${name}/${caseName}[${index}]: validity differs (JS=${js.usable}, Python=${py.usable})`);
        }
        if (Math.abs(js.x - py.x) > 1e-4 || Math.abs(js.y - py.y) > 1e-4) {
          throw new Error(`${name}/${caseName}[${index}]: source coordinates differ`);
        }
        for (const [field, keys] of Object.entries(contract.result_fields)) {
          if ((field === "dx_px" || field === "dy_px") && !js.usable) continue;
          const jsValue = js[keys.javascript];
          const pyValue = py[keys.python];
          if (jsValue === null || pyValue === null || jsValue === undefined || pyValue === undefined) {
            if (jsValue !== pyValue) {
              throw new Error(`${name}/${caseName}[${index}].${field}: nullability differs`);
            }
            continue;
          }
          if (typeof jsValue === "boolean" || typeof jsValue === "string") {
            if (jsValue !== pyValue) {
              throw new Error(`${name}/${caseName}[${index}].${field} differs`);
            }
            continue;
          }
          const difference = Math.abs(jsValue - pyValue);
          const tolerance = ["score", "second_score", "margin", "inlier_fraction"].includes(field) ? 0.02
            : ["source_x_px", "source_y_px", "seed_x_px", "seed_y_px", "root_x_px", "root_y_px", "target_x_px", "target_y_px"].includes(field) ? 1e-4
              : ["dx_px", "dy_px", "residual_px", "local_consistency_px"].includes(field) ? 0.05
                : 0;
          maxDiagnosticDifference = Math.max(maxDiagnosticDifference, difference);
          if (field === "score") maxScoreDifference = Math.max(maxScoreDifference, difference);
          if (difference > tolerance) {
            throw new Error(`${name}/${caseName}[${index}].${field} differs by ${difference.toFixed(5)}`);
          }
        }
        if (js.usable) {
          const displacementDifference = Math.hypot(js.dx - py.dx, js.dy - py.dy);
          maxDisplacementDifference = Math.max(maxDisplacementDifference, displacementDifference);
          if (displacementDifference > 0.05) {
            throw new Error(`${name}/${caseName}[${index}]: usable displacement differs by ${displacementDifference.toFixed(5)} px`);
          }
          const truth = browserReport.transforms[caseName];
          const expected = caseName === "translation" || caseName === "manual_translation"
            ? truth
            : [
              (truth.matrix[0][0] - 1) * (js.x - truth.center[0])
                + truth.matrix[0][1] * (js.y - truth.center[1]) + truth.translation[0],
              truth.matrix[1][0] * (js.x - truth.center[0])
                + (truth.matrix[1][1] - 1) * (js.y - truth.center[1]) + truth.translation[1],
            ];
          if (Math.hypot(js.dx - expected[0], js.dy - expected[1]) <= 2) correct++;
          accepted++;
        }
      }
      const minimumCorrect = caseName === "manual_translation" ? 2 : 6;
      if (accepted < minimumCorrect || correct < minimumCorrect) {
        throw new Error(`${name}/${caseName}: accepted=${accepted}/${expectedCount} and <=2px=${correct}/${expectedCount}; expected at least ${minimumCorrect}`);
      }
      summary.cases[`${name}/${caseName}`] = {
        count: jsMatches.length, accepted, within2px: correct,
        maxDisplacementDifference, maxScoreDifference, maxDiagnosticDifference,
      };
    }
  }
  if (process.env.WGPU_MATCHING_PARITY_DEVICE_TYPE === "cpu") {
    summary.cases["cpm/manual_translation"] = {
      status: "covered by the following standalone browser CPM smoke test",
    };
    return summary;
  }
  const cpmJS = browserReport.methods.cpm.manual_translation;
  const cpmPython = python.methods.cpm.manual_translation;
  if (cpmJS.length !== 3 || cpmPython.length !== 3) {
    throw new Error(`cpm/manual_translation: expected 3 seeds, got JS=${cpmJS.length}, Python=${cpmPython.length}`);
  }
  for (const key of ["workGridPointCount", "gridWidth", "gridHeight"]) {
    if (browserReport.methods.cpm.metadata[key] !== python.methods.cpm.metadata[key]) {
      throw new Error(
        `cpm metadata ${key} differs between browser (${browserReport.methods.cpm.metadata[key]}) `
        + `and Python (${python.methods.cpm.metadata[key]})`,
      );
    }
  }
  const cpmWorkGridCounts = {
    browser: {
      reciprocal: browserReport.methods.cpm.metadata.workGridReciprocalCount,
      usable: browserReport.methods.cpm.metadata.workGridUsableCount,
    },
    python: {
      reciprocal: python.methods.cpm.metadata.workGridReciprocalCount,
      usable: python.methods.cpm.metadata.workGridUsableCount,
    },
  };
  if (process.env.WGPU_MATCHING_PARITY_DEVICE_TYPE !== "cpu") {
    for (const key of ["workGridReciprocalCount", "workGridUsableCount"]) {
      if (browserReport.methods.cpm.metadata[key] !== python.methods.cpm.metadata[key]) {
        throw new Error(
          `cpm metadata ${key} differs between browser (${browserReport.methods.cpm.metadata[key]}) `
          + `and Python (${python.methods.cpm.metadata[key]})`,
        );
      }
    }
  }
  let cpmAccepted = 0;
  let cpmCorrect = 0;
  let cpmMaxDisplacementDifference = 0;
  let cpmMaxDiagnosticDifference = 0;
  const softwareDiagnosticTolerance =
    process.env.WGPU_MATCHING_PARITY_DEVICE_TYPE === "cpu" ? 0.1 : 0.05;
  for (let index = 0; index < cpmJS.length; index++) {
    const js = cpmJS[index];
    const py = cpmPython[index];
    if (js.usable !== py.usable || js.reciprocal !== py.reciprocal) {
      throw new Error(`cpm/manual_translation[${index}]: usability or reciprocity differs`);
    }
    for (const key of ["x", "y", "measuredX", "measuredY", "measuredDx", "measuredDy"]) {
      const difference = Math.abs(js[key] - py[key]);
      cpmMaxDiagnosticDifference = Math.max(cpmMaxDiagnosticDifference, difference);
      if (difference > 1e-4) throw new Error(`cpm/manual_translation[${index}].${key} differs by ${difference}`);
    }
    for (const key of ["gridDistancePx", "reciprocalErrorPx", "score", "residual", "inlierFraction"]) {
      const a = js[key];
      const b = py[key];
      if (a == null || b == null) {
        if (a !== b) throw new Error(`cpm/manual_translation[${index}].${key} nullability differs`);
        continue;
      }
      const difference = Math.abs(a - b);
      cpmMaxDiagnosticDifference = Math.max(cpmMaxDiagnosticDifference, difference);
      const tolerance = key === "residual" ? softwareDiagnosticTolerance : 0.05;
      if (difference > tolerance) {
        throw new Error(
          `cpm/manual_translation[${index}].${key} differs by ${difference} (limit ${tolerance})`,
        );
      }
    }
    const difference = Math.hypot(js.dx - py.dx, js.dy - py.dy);
    cpmMaxDisplacementDifference = Math.max(cpmMaxDisplacementDifference, difference);
    if (difference > 0.05) throw new Error(`cpm/manual_translation[${index}] displacement differs by ${difference} px`);
    if (js.usable) {
      cpmAccepted++;
      if (Math.hypot(js.dx - 8, js.dy + 4) <= 2) cpmCorrect++;
    }
  }
  if (cpmAccepted !== 3 || cpmCorrect !== 3) {
    throw new Error(`cpm/manual_translation: accepted=${cpmAccepted}/3 and <=2px=${cpmCorrect}/3`);
  }
  const cpmOracleJS = browserReport.methods.cpm.oracle?.perSeed ?? [];
  const cpmOraclePython = python.methods.cpm.oracle?.perSeed ?? [];
  if (cpmOracleJS.length !== 3 || cpmOraclePython.length !== 3) {
    throw new Error("CPM analytic-cost diagnostics are missing from one frontend");
  }
  let cpmOracleMaxCostDifference = 0;
  for (let seedIndex = 0; seedIndex < cpmOracleJS.length; seedIndex++) {
    const jsLevels = cpmOracleJS[seedIndex].levels;
    const pyLevels = cpmOraclePython[seedIndex].levels;
    if (jsLevels.length !== pyLevels.length) throw new Error("CPM oracle pyramid depth differs");
    for (let levelIndex = 0; levelIndex < jsLevels.length; levelIndex++) {
      const js = jsLevels[levelIndex];
      const py = pyLevels[levelIndex];
      for (const key of ["foundValid", "oracleValid"]) {
        if (js[key] !== py[key]) throw new Error(`CPM oracle ${key} differs at seed ${seedIndex}, level ${levelIndex}`);
      }
      for (const key of ["foundCost", "oracleCost", "oracleMinusFoundCost"]) {
        if (js[key] == null || py[key] == null) {
          if (js[key] !== py[key]) throw new Error(`CPM oracle ${key} nullability differs`);
          continue;
        }
        const difference = Math.abs(js[key] - py[key]);
        cpmOracleMaxCostDifference = Math.max(cpmOracleMaxCostDifference, difference);
        if (difference > 0.02) throw new Error(`CPM oracle ${key} differs by ${difference}`);
      }
      for (const key of ["foundDxPx", "foundDyPx", "oracleDxPx", "oracleDyPx"]) {
        if (js[key] !== py[key]) throw new Error(`CPM oracle displacement ${key} differs`);
      }
    }
  }
  summary.cases["cpm/manual_translation"] = {
    count: cpmJS.length,
    accepted: cpmAccepted,
    within2px: cpmCorrect,
    maxDisplacementDifference: cpmMaxDisplacementDifference,
    maxDiagnosticDifference: cpmMaxDiagnosticDifference,
    maxOracleCostDifference: cpmOracleMaxCostDifference,
  };
  return summary;
}

try {
  const contract = JSON.parse(await readFile(
    path.join(root, "python/src/wgpu_matching/matcher-contracts.json"), "utf8"));
  const python = await pythonReport();
  const chrome = process.env.CHROMIUM_BIN || process.env.CHROME_PATH || "google-chrome";
  browser = spawn(chrome, [
    "--headless=new", "--no-sandbox", "--disable-dev-shm-usage", "--use-gl=angle",
    "--use-angle=vulkan", "--enable-features=Vulkan", "--enable-unsafe-webgpu",
    "--ignore-gpu-blocklist", "--disable-vulkan-surface",
    `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`, "about:blank",
  ], { stdio: "ignore" });

  await poll(async () => {
    const response = await fetch(`http://127.0.0.1:${debugPort}/json/version`);
    return response.ok ? response.json() : null;
  }, value => Boolean(value?.webSocketDebuggerUrl), 30000, "Chromium DevTools endpoint");
  const target = await poll(async () => {
    const parityPage = process.env.WGPU_MATCHING_PARITY_DEVICE_TYPE === "cpu"
      ? "matcher-parity-browser.html?software=1"
      : "matcher-parity-browser.html";
    const response = await fetch(`http://127.0.0.1:${debugPort}/json/new?http://127.0.0.1:${httpPort}/tests/${parityPage}`, { method: "PUT" });
    if (!response.ok) throw new Error(`DevTools target creation failed: ${response.status}`);
    return response.json();
  }, value => value?.webSocketDebuggerUrl, 15000, "browser parity page");

  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    ws.addEventListener("open", resolve, { once: true });
    ws.addEventListener("error", reject, { once: true });
  });
  let nextId = 0;
  const pending = new Map();
  ws.addEventListener("message", event => {
    const message = JSON.parse(event.data);
    if (!message.id || !pending.has(message.id)) return;
    const { resolve, reject } = pending.get(message.id);
    pending.delete(message.id);
    if (message.error) reject(new Error(message.error.message));
    else resolve(message.result);
  });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++nextId;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = expression => send("Runtime.evaluate", { expression, returnByValue: true });
  const titleResult = await poll(async () => (await evaluate("document.title")).result?.value,
    title => title?.startsWith("MATCH_PARITY:") || title?.startsWith("MATCH_PARITY_ERROR:"),
    180000, "browser parity result");
  if (titleResult.startsWith("MATCH_PARITY_ERROR:")) {
    throw new Error(decodeURIComponent(escape(atob(titleResult.slice("MATCH_PARITY_ERROR:".length)))));
  }
  const browserReport = JSON.parse(decodeURIComponent(escape(atob(titleResult.slice("MATCH_PARITY:".length)))));
  const navigation = await send("Page.navigate", { url: `http://127.0.0.1:${httpPort}/index.html` });
  if (navigation.errorText) throw new Error(`Could not load the browser application: ${navigation.errorText}`);
  const uiStateText = await poll(async () => {
    const value = (await evaluate(`JSON.stringify({
      status: document.querySelector('#status')?.textContent || '',
      controls: Object.fromEntries(['patch-size','root-patch-size','grid-stride','downscale-levels',
        'search-radius','horg-reduction','horg-max-features','cpm-grid-stride','cpm-matching-support',
        'cpm-levels','cpm-initial-hypotheses','cpm-propagation-iterations','cpm-random-trials',
        'cpm-local-radius','cpm-seed','cpm-reciprocal-threshold'].map(id =>
        [id, document.getElementById(id)?.value || '']))
    })`)).result?.value;
    return value || null;
  }, value => value && JSON.parse(value).status.includes("WebGPU prêt"), 30000, "application defaults and WebGPU init");
  ws.close();
  const uiState = JSON.parse(uiStateText);
  const expectedUi = {
    "patch-size": String(contract.algorithms.deepmatching.defaults.patch_size_px),
    "root-patch-size": String(contract.algorithms.deepmatching.defaults.root_patch_size_px),
    "grid-stride": String(contract.algorithms.deepmatching.defaults.stride_px),
    "downscale-levels": String(contract.algorithms.deepmatching.defaults.downscale_levels),
    "search-radius": String(contract.algorithms.deepmatching.defaults.search_radius_px),
    "horg-reduction": String(contract.algorithms.horg.defaults.reduction),
    "horg-max-features": String(contract.algorithms.horg.defaults.max_features),
    "cpm-grid-stride": String(contract.algorithms.cpm.defaults.work_grid_stride_px),
    "cpm-matching-support": String(contract.algorithms.cpm.defaults.matching_support_px),
    "cpm-levels": String(contract.algorithms.cpm.defaults.levels),
    "cpm-initial-hypotheses": String(contract.algorithms.cpm.defaults.initial_hypotheses),
    "cpm-propagation-iterations": String(contract.algorithms.cpm.defaults.propagation_iterations),
    "cpm-random-trials": String(contract.algorithms.cpm.defaults.random_trials),
    "cpm-local-radius": String(contract.algorithms.cpm.defaults.local_search_radius_px),
    "cpm-seed": String(contract.algorithms.cpm.defaults.seed),
    "cpm-reciprocal-threshold": String(contract.algorithms.cpm.defaults.reciprocal_threshold_px),
  };
  for (const [control, expected] of Object.entries(expectedUi)) {
    if (uiState.controls[control] !== expected) {
      throw new Error(`Browser UI default ${control}=${uiState.controls[control]}, expected ${expected}`);
    }
  }
  const summary = compare(browserReport, python, contract);
  summary.browserUiDefaults = uiState.controls;
  summary.cpmWorkGridCounts = cpmWorkGridCounts;
  console.log(JSON.stringify(summary, null, 2));
} finally {
  await stopBrowser();
  server.close();
  await rm(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
}
