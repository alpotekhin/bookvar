import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import process from "node:process";

const root = resolve(import.meta.dirname, "../..");

const matrices = [
  {
    file: "05 Источники/Source maps/Harvard ML Systems — complete transfer matrix.md",
    sourceMarkers: ["Harvard ML Systems", "harvard-edge/cs249r_book"],
  },
  {
    file: "05 Источники/Source maps/Efficient DL Systems — complete transfer matrix.md",
    sourceMarkers: ["Efficient DL Systems", "mryab/efficient-dl-systems"],
  },
];

const failures = [];
let checkedRows = 0;
let checkedDestinations = 0;

for (const matrix of matrices) {
  const matrixPath = resolve(root, matrix.file);
  const lines = readFileSync(matrixPath, "utf8").split("\n");
  const prefix = matrix.file.includes("Harvard") ? "| H-" : "| E-";
  const rows = lines.filter((line) => line.startsWith(prefix));
  checkedRows += rows.length;

  const destinations = new Set();
  for (const row of rows) {
    const cells = row.split("|").map((cell) => cell.trim());
    const decision = cells[5];
    const destination = cells[6]?.replace(/^`|`$/g, "");

    if (!["integrate", "cross-link"].includes(decision)) continue;
    if (!destination || destination.startsWith("05 Источники/")) continue;
    destinations.add(destination);
  }

  for (const destination of destinations) {
    checkedDestinations += 1;
    const destinationPath = resolve(root, destination);
    if (!existsSync(destinationPath)) {
      failures.push(`${matrix.file}: destination does not exist: ${destination}`);
      continue;
    }

    const content = readFileSync(destinationPath, "utf8");
    if (!matrix.sourceMarkers.some((marker) => content.includes(marker))) {
      failures.push(
        `${matrix.file}: destination has no direct source marker: ${destination}`,
      );
    }
  }
}

if (failures.length > 0) {
  console.error(
    `Source coverage check failed: ${failures.length} problem(s) across ` +
      `${checkedRows} matrix rows and ${checkedDestinations} destinations.`,
  );
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(
  `Source coverage check passed: ${checkedRows} matrix rows, ` +
    `${checkedDestinations} existing destinations with direct source links.`,
);
