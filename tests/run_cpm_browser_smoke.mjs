import { spawn } from "node:child_process";
import { mkdtemp, rm, stat } from "node:fs/promises";
import { createReadStream } from "node:fs";
import { createServer } from "node:http";
import { createServer as createTcpServer } from "node:net";
import os from "node:os";
import path from "node:path";
import process from "node:process";

const root = path.resolve(import.meta.dirname, "..");
const profile = await mkdtemp(path.join(os.tmpdir(), "cpm-webgpu-smoke-"));
const mime = { ".html": "text/html", ".js": "text/javascript", ".wgsl": "text/plain", ".json": "application/json" };
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

function waitForExit(child, label) {
  return new Promise((_, reject) => {
    child.once("exit", (code, signal) => reject(new Error(`${label} exited (${code ?? signal})`)));
    child.once("error", reject);
  });
}

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

try {
  const chrome = process.env.CHROMIUM_BIN || process.env.CHROME_PATH || "google-chrome";
  browser = spawn(chrome, [
    "--headless=new",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--use-gl=angle",
    "--use-angle=vulkan",
    "--enable-features=Vulkan",
    "--enable-unsafe-webgpu",
    "--ignore-gpu-blocklist",
    "--disable-vulkan-surface",
    `--remote-debugging-port=${debugPort}`,
    `--user-data-dir=${profile}`,
    "about:blank",
  ], { stdio: "ignore" });

  await poll(async () => {
    const response = await fetch(`http://127.0.0.1:${debugPort}/json/version`);
    return response.ok ? response.json() : null;
  }, value => Boolean(value?.webSocketDebuggerUrl), 30000, "Chromium DevTools endpoint").catch(async error => {
    await Promise.race([waitForExit(browser, "Chromium"), new Promise(resolve => setTimeout(resolve, 100))]);
    throw error;
  });
  const target = await poll(async () => {
    const response = await fetch(`http://127.0.0.1:${debugPort}/json/new?http://127.0.0.1:${httpPort}/tests/cpm-browser-smoke.html`, { method: "PUT" });
    if (!response.ok) throw new Error(`DevTools target creation failed: ${response.status}`);
    return response.json();
  }, value => value?.webSocketDebuggerUrl, 15000, "browser page");

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
  const evaluate = expression => new Promise((resolve, reject) => {
    const id = ++nextId;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ id, method: "Runtime.evaluate", params: { expression, returnByValue: true } }));
  });
  await poll(async () => (await evaluate("document.title")).result?.value,
    title => title?.startsWith("CPM_RESULT:") || title?.startsWith("CPM_ERROR:"), 120000, "CPM WebGPU result");
  const title = (await evaluate("document.title")).result.value;
  ws.close();
  if (title.startsWith("CPM_ERROR:")) throw new Error(decodeURIComponent(escape(atob(title.slice("CPM_ERROR:".length)))));
  const report = JSON.parse(decodeURIComponent(escape(atob(title.slice("CPM_RESULT:".length)))));
  if (report.count !== 3 || report.accepted !== 3 || report.maxErrorPx > 0.5) {
    throw new Error(`CPM smoke criteria failed: ${JSON.stringify(report)}`);
  }
  console.log(JSON.stringify(report));
} finally {
  browser?.kill("SIGTERM");
  server.close();
  await rm(profile, { recursive: true, force: true });
}
