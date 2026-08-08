#!/usr/bin/env node
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

function platformKey() {
  const key = `${process.platform}-${process.arch}`;
  return ["darwin-arm64", "darwin-x64", "linux-x64", "linux-arm64", "win32-x64"].includes(key) ? key : key;
}
function resolvePackage() {
  const pkg = `autoconduck-${platformKey()}`;
  try { return path.dirname(require.resolve(`${pkg}/package.json`, { paths: [__dirname, process.cwd()] })); } catch {}
  for (const candidate of [path.join(__dirname, "..", "..", pkg), path.join(__dirname, "..", "node_modules", pkg), path.join(process.cwd(), "node_modules", pkg)]) {
    if (fs.existsSync(path.join(candidate, "package.json"))) return candidate;
  }
  return null;
}
function findPython() {
  const candidates = process.env.AUTOCONDUCK_PYTHON ? [process.env.AUTOCONDUCK_PYTHON] : (process.platform === "win32" ? ["python3", "python", "py"] : ["python3", "python"]);
  return candidates[0];
}
const python = findPython();
if (!python) { console.error("[autoconduck] Python 3.11+ is required. Set AUTOCONDUCK_PYTHON."); process.exit(1); }
const env = { ...process.env };
const pkg = resolvePackage();
if (pkg) env.AUTOCONDUCK_WHEEL_DIR = path.join(pkg, "python");
const child = spawn(python, [path.join(__dirname, "bootstrap.py"), ...process.argv.slice(2)], { stdio: "inherit", env });
child.on("error", (error) => { console.error(`[autoconduck] unable to start Python: ${error.message}`); process.exit(1); });
child.on("exit", (code, signal) => { if (signal) process.kill(process.pid, signal); else process.exit(code ?? 1); });
