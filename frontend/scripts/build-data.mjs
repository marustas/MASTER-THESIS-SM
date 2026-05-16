#!/usr/bin/env node
/**
 * Convert the latest export from src/export_results.py into a JSON file
 * bundled with the React app.
 *
 *   CSV  : ../experiments/results/exports/programme_job_mapping.csv
 *   JSON : src/data/rankings.json
 *
 * Run:  npm run build:data
 */

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import Papa from "papaparse";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, "..", "..");
const CSV_PATH = resolve(REPO_ROOT, "experiments", "results", "exports", "programme_job_mapping.csv");
const OUT_PATH = resolve(__dirname, "..", "src", "data", "rankings.json");

const csv = readFileSync(CSV_PATH, "utf8");
const parsed = Papa.parse(csv, {
  header: true,
  dynamicTyping: true,
  skipEmptyLines: true,
});

if (parsed.errors.length > 0) {
  console.error("CSV parse errors:", parsed.errors);
  process.exit(1);
}

const cleanField = (value) => {
  if (value === null || value === undefined) return "";
  const s = String(value).trim();
  return s.toLowerCase() === "nan" ? "" : s;
};

// Group rows by (programme_name, institution)
const grouped = new Map();
for (const row of parsed.data) {
  const programmeName = cleanField(row.programme_name);
  const institution = cleanField(row.institution);
  const key = `${programmeName}|${institution}`;
  if (!grouped.has(key)) {
    grouped.set(key, {
      programme_name: programmeName,
      institution,
      jobs: [],
    });
  }
  grouped.get(key).jobs.push({
    rank: Number(row.rank),
    job_title: cleanField(row.job_title),
    employer_sector: cleanField(row.employer_sector),
    location: cleanField(row.location),
    job_url: cleanField(row.job_url),
    hybrid_score: Number(row.hybrid_score),
    semantic_score: Number(row.semantic_score),
    top_skill_gaps: cleanField(row.top_skill_gaps)
      .split(";")
      .map((s) => s.trim())
      .filter(Boolean),
  });
}

const programmes = [...grouped.values()]
  .map((p) => ({ ...p, jobs: p.jobs.sort((a, b) => a.rank - b.rank) }))
  .sort((a, b) => a.programme_name.localeCompare(b.programme_name));

mkdirSync(dirname(OUT_PATH), { recursive: true });
writeFileSync(OUT_PATH, JSON.stringify(programmes, null, 2));

const totalJobs = programmes.reduce((acc, p) => acc + p.jobs.length, 0);
console.log(`Wrote ${programmes.length} programmes (${totalJobs} jobs) → ${OUT_PATH}`);
