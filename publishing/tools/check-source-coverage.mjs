import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
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
let checkedHarvardLabs = 0;
let checkedHarvardSlides = 0;
let checkedHseArtifacts = 0;

function filesBelow(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = resolve(directory, entry.name);
    return entry.isDirectory() ? filesBelow(path) : [path];
  });
}

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

const scikitCourse = {
  manifest:
    "05 Источники/Courses/Scikit-learn MOOC/import-manifest.json",
  figures: "Assets/Sources/Scikit-learn MOOC/figures",
  expectedPages: 215,
  expectedFigures: 212,
  routePrefix: "sources/courses/scikit-learn-mooc/",
  canonicalDirectory: "00 Учебник/01 Классическое машинное обучение",
  canonicalFiles: [
    "00 Карта модуля.md",
    "01 Задача обучения и predictive pipeline.md",
    "02 Линейная регрессия и классификация.md",
    "03 Деревья решений.md",
    "04 Bagging, random forest и gradient boosting.md",
    "05 Оценивание, кросс-валидация и выбор порога.md",
    "06 Кластеризация и её ограничения.md",
  ],
  sourceMarkers: [
    "Scikit-learn MOOC",
    "INRIA/scikit-learn-mooc",
    "05 Источники/Courses/Scikit-learn MOOC",
  ],
};

const manifestPath = resolve(root, scikitCourse.manifest);
if (!existsSync(manifestPath)) {
  failures.push(`${scikitCourse.manifest}: manifest does not exist`);
} else {
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  if (manifest.length !== scikitCourse.expectedPages) {
    failures.push(
      `${scikitCourse.manifest}: expected ${scikitCourse.expectedPages} pages, ` +
        `found ${manifest.length}`,
    );
  }

  const routes = new Set();
  for (const entry of manifest) {
    const outputPath = resolve(root, entry.output);
    if (!existsSync(outputPath)) {
      failures.push(`${scikitCourse.manifest}: output does not exist: ${entry.output}`);
      continue;
    }
    if (!entry.route?.startsWith(scikitCourse.routePrefix)) {
      failures.push(`${scikitCourse.manifest}: invalid route: ${entry.route}`);
    }
    if (routes.has(entry.route)) {
      failures.push(`${scikitCourse.manifest}: duplicate route: ${entry.route}`);
    }
    routes.add(entry.route);

    if (!/^[0-9a-f]{64}$/.test(entry.sha256 ?? "")) {
      failures.push(`${scikitCourse.manifest}: invalid source hash: ${entry.output}`);
    }
    const output = readFileSync(outputPath, "utf8");
    if (!output.includes(entry.source_path) || !output.includes("source_commit:")) {
      failures.push(`${scikitCourse.manifest}: incomplete provenance: ${entry.output}`);
    }
  }

  const navigation = readFileSync(resolve(root, "publishing/navigation.yml"), "utf8");
  for (const route of routes) {
    if (!navigation.includes(`route: ${route}`)) {
      failures.push(`${scikitCourse.manifest}: route missing from navigation: ${route}`);
    }
  }
}

const figuresPath = resolve(root, scikitCourse.figures);
if (!existsSync(figuresPath)) {
  failures.push(`${scikitCourse.figures}: figure directory does not exist`);
} else {
  const figureFiles = filesBelow(figuresPath).filter((path) => statSync(path).isFile());
  if (figureFiles.length !== scikitCourse.expectedFigures) {
    failures.push(
      `${scikitCourse.figures}: expected ${scikitCourse.expectedFigures} figures, ` +
        `found ${figureFiles.length}`,
    );
  }
}

const canonicalPath = resolve(root, scikitCourse.canonicalDirectory);
if (!existsSync(canonicalPath)) {
  failures.push(`${scikitCourse.canonicalDirectory}: canonical module does not exist`);
} else {
  const canonicalPages = scikitCourse.canonicalFiles.map((file) =>
    resolve(canonicalPath, file),
  );
  for (const page of canonicalPages) {
    if (!existsSync(page)) {
      failures.push(`${scikitCourse.canonicalDirectory}: page does not exist: ${page}`);
      continue;
    }
    const content = readFileSync(page, "utf8");
    if (!scikitCourse.sourceMarkers.some((marker) => content.includes(marker))) {
      failures.push(
        `${scikitCourse.canonicalDirectory}: page has no direct course marker: ${page}`,
      );
    }
  }
}

const harvardPracticeManifest =
  "05 Источники/Courses/Harvard ML Systems/labs-slides-manifest.json";
const harvardPracticePath = resolve(root, harvardPracticeManifest);
if (!existsSync(harvardPracticePath)) {
  failures.push(`${harvardPracticeManifest}: manifest does not exist`);
} else {
  const records = JSON.parse(readFileSync(harvardPracticePath, "utf8"));
  const navigation = readFileSync(resolve(root, "publishing/navigation.yml"), "utf8");
  checkedHarvardLabs = records.filter(({ kind }) => kind === "lab").length;
  checkedHarvardSlides = records.filter(({ kind }) => kind === "slides").length;
  if (checkedHarvardLabs !== 34) {
    failures.push(
      `${harvardPracticeManifest}: expected 34 labs, found ${checkedHarvardLabs}`,
    );
  }
  if (checkedHarvardSlides !== 35) {
    failures.push(
      `${harvardPracticeManifest}: expected 35 slide decks, found ${checkedHarvardSlides}`,
    );
  }
  for (const record of records) {
    const outputPath = resolve(root, record.output);
    if (!existsSync(outputPath)) {
      failures.push(`${harvardPracticeManifest}: output missing: ${record.output}`);
      continue;
    }
    const content = readFileSync(outputPath, "utf8");
    if (
      !content.includes(record.source) ||
      !content.includes("45ecc8d82fcae70c149cdce550d3b3d3411df913") ||
      !content.includes("Complete original")
    ) {
      failures.push(
        `${harvardPracticeManifest}: incomplete provenance: ${record.output}`,
      );
    }
    if (!navigation.includes(`route: ${record.route}`)) {
      failures.push(
        `${harvardPracticeManifest}: route missing from navigation: ${record.route}`,
      );
    }
  }
}

const hseManifest = "05 Источники/Courses/HSE ML course/import-manifest.json";
const hseManifestPath = resolve(root, hseManifest);
if (!existsSync(hseManifestPath)) {
  failures.push(`${hseManifest}: manifest does not exist`);
} else {
  const records = JSON.parse(readFileSync(hseManifestPath, "utf8"));
  const navigation = readFileSync(resolve(root, "publishing/navigation.yml"), "utf8");
  checkedHseArtifacts = records.length;
  if (checkedHseArtifacts !== 67) {
    failures.push(`${hseManifest}: expected 67 artifacts, found ${checkedHseArtifacts}`);
  }
  for (const record of records) {
    const outputPath = resolve(root, record.output);
    if (!existsSync(outputPath)) {
      failures.push(`${hseManifest}: output missing: ${record.output}`);
      continue;
    }
    const content = readFileSync(outputPath, "utf8");
    if (
      !content.includes(record.source) ||
      !content.includes("4b21051531fb72dc9eef58632332ad971c92d006")
    ) {
      failures.push(`${hseManifest}: incomplete provenance: ${record.output}`);
    }
    for (const asset of record.assets ?? []) {
      if (!existsSync(resolve(root, asset))) {
        failures.push(`${hseManifest}: asset missing: ${asset}`);
      }
    }
    if (!navigation.includes(`route: ${record.route}`)) {
      failures.push(`${hseManifest}: route missing from navigation: ${record.route}`);
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
    `${checkedDestinations} existing destinations with direct source links, ` +
    `${scikitCourse.expectedPages} Scikit-learn MOOC pages, ` +
    `${scikitCourse.expectedFigures} figures, and ` +
    `${scikitCourse.canonicalFiles.length} canonical course-linked pages; ` +
    `${checkedHarvardLabs} Harvard labs, ${checkedHarvardSlides} slide decks, ` +
    `and ${checkedHseArtifacts} HSE course artifacts.`,
);
