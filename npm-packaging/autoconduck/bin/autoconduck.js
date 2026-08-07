#!/usr/bin/env node
const { spawn, spawnSync } = require("child_process");
const path = require("path");
const fs = require("fs");

function platformKey() {
  const plat = process.platform; // darwin, linux, win32
  const arch = process.arch; // x64, arm64
  if (plat === "darwin" && arch === "arm64") return "darwin-arm64";
  if (plat === "darwin" && arch === "x64") return "darwin-x64";
  if (plat === "linux" && arch === "x64") return "linux-x64";
  if (plat === "linux" && arch === "arm64") return "linux-arm64";
  if (plat === "win32" && arch === "x64") return "win32-x64";
  return `${plat}-${arch}`;
}

function resolveBinary() {
  const key = platformKey();
  const pkg = `autoconduck-${key}`;
  // try to resolve platform package
  try {
    const resolved = require.resolve(`${pkg}/bin/autoconduck`, { paths: [__dirname, process.cwd(), path.join(__dirname, "..")] });
    return resolved;
  } catch {}
  // try direct path relative to node_modules
  const candidates = [
    path.join(__dirname, "..", "..", pkg, "bin", process.platform === "win32" ? "autoconduck.exe" : "autoconduck"),
    path.join(__dirname, "..", "node_modules", pkg, "bin", process.platform === "win32" ? "autoconduck.exe" : "autoconduck"),
    path.join(process.cwd(), "node_modules", pkg, "bin", process.platform === "win32" ? "autoconduck.exe" : "autoconduck"),
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  // fallback: try python -m autoconduck (dev mode)
  return null;
}

const bin = resolveBinary();
const args = process.argv.slice(2);

if (!bin) {
  // dev fallback: try python
  const pyCandidates = ["python", "python3", "py"];
  let ran = false;
  for (const py of pyCandidates) {
    const r = spawnSync(py, ["-m", "autoconduck.main", ...args], { stdio: "inherit" });
    if (r.error && r.error.code === "ENOENT") continue;
    ran = true;
    process.exit(r.status ?? 0);
    break;
  }
  if (!ran) {
    console.error(`[autoconduck] binary not found for platform ${platformKey()}.`);
    console.error(`  Install the matching optionalDependency: autoconduck-${platformKey()}`);
    console.error(`  Or run via python: python -m autoconduck ${args.join(" ")}`);
    process.exit(1);
  }
} else {
  const child = spawn(bin, args, { stdio: "inherit" });
  child.on("exit", (code, signal) => {
    if (signal) process.kill(process.pid, signal);
    else process.exit(code ?? 0);
  });
}
