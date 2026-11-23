// Grid layout explorer on the grid how-to page. The formulas copy array_layout and
// idle_share in src/proteus/grid/slurm.py and the throttle of Grid.slurm_config: change them together.
const IDLE_WARN_SHARE = 0.1;

const SLIDERS = [
  ["cases", "Cases in the grid", 1, 200, 1, 50],
  ["cores", "cores_per_job", 1, 16, 1, 4],
  ["maxJobs", "max_jobs", 1, 64, 1, 16],
  ["maxArray", "MaxArraySize", 1, 40, 1, 40],
  ["days", "max_days", 0.5, 3, 0.5, 1],
  ["mem", "max_mem [GB]", 0.5, 8, 0.25, 1.75],
];

// Snellius partitions: cores and GiB per node, SBU per charged core-hour (sources on the
// grid how-to page). A single-node job is charged in eighths of a node, by cores or memory.
const PARTITIONS = {
  rome: [128, 224, 1.0],
  genoa: [192, 336, 1.0],
  fat_rome: [128, 960, 1.5],
  fat_genoa: [192, 1440, 1.5],
};

function layout(n, k, maxArray) {
  const jobs = Math.min(Math.ceil(n / k), maxArray);
  const waves = Math.ceil(Math.ceil(n / jobs) / k);
  return { jobs, waves };
}

// Rows summed over all jobs: core time in units of one case on cores_per_job cores.
function runs(n, k, jobs) {
  const q = Math.floor(n / jobs);
  const r = n % jobs;
  return r * Math.ceil((q + 1) / k) + (jobs - r) * Math.ceil(q / k);
}

function idleShare(n, k, jobs) {
  return 1 - n / (runs(n, k, jobs) * k);
}

function charge(k, mem, partition) {
  const [cores, nodeMem, weight] = PARTITIONS[partition];
  const eighths = Math.ceil(Math.max((8 * k) / cores, (8 * k * mem) / nodeMem));
  const charged = (eighths * cores) / 8;
  return { charged, perHour: charged * weight, fits: eighths <= 8, cores, nodeMem };
}

function sbuStats(v, jobs) {
  const c = charge(v.cores, v.mem, v.partition);
  const rows = runs(v.cases, v.cores, jobs);
  const grid = rows * v.days * 24 * c.perHour;
  const stats = [
    ["Charged cores per job", c.charged],
    ["SBU per job-hour", c.perHour],
    ["SBU per case-hour", (grid / (v.cases * v.days * 24)).toFixed(2)],
    ["SBU for the grid", Math.round(grid)],
    ["Charged cores without a case", `${((1 - v.cases / (rows * c.charged)) * 100).toFixed(1)} %`],
  ];
  const out = c.fits ? [] : [
    `A job of ${v.cores} cores and ${v.cores * v.mem} GiB doesn't fit one ${v.partition} node (${c.cores} cores, ${c.nodeMem} GiB).`,
  ];
  return { stats, out };
}

function statList(stats) {
  return stats.map(([name, value]) => `<div><dt>${name}</dt><dd>${value}</dd></div>`).join("");
}

function warnings(v, jobs, waves, throttle, idle) {
  const out = [];
  if (v.maxJobs < v.cores) {
    out.push(`max_jobs is smaller than cores_per_job: 1 job still runs, so up to ${v.cores} cases run at once.`);
  } else if (v.maxJobs % v.cores && Math.floor(v.maxJobs / v.cores) < jobs) {
    out.push(`max_jobs isn't a multiple of cores_per_job: it's rounded down to ${throttle} jobs, so ${throttle * v.cores} cases at once.`);
  }
  if (idle > IDLE_WARN_SHARE) {
    out.push(`${(idle * 100).toFixed(1)} % of the allocated cores are idle; the grid manager warns above 10 %.`);
  }
  if (waves > 1) {
    out.push(`MaxArraySize limits the array to ${jobs} jobs, so each core runs up to ${waves} cases in turn.`);
  }
  return out;
}

function jobBox(j, v, jobs, running) {
  const cases = [];
  for (let c = j; c < v.cases; c += jobs) cases.push(c);
  const cells = Math.ceil(cases.length / v.cores) * v.cores;
  let html = "";
  for (let i = 0; i < cells; i++) {
    html += i < cases.length
      ? `<span class="gx-cell">${cases[i]}</span>`
      : '<span class="gx-cell gx-idle" title="idle core"></span>';
  }
  return `<div class="gx-job ${running ? "gx-running" : "gx-queued"}">
    <div class="gx-caption">Job ${j}${running ? "" : " (queued)"}</div>
    <div class="gx-cells" style="grid-template-columns: repeat(${v.cores}, 1fr)">${html}</div>
  </div>`;
}

function render(root) {
  const v = { partition: root.querySelector("select").value };
  for (const input of root.querySelectorAll("input")) {
    v[input.name] = Number(input.value);
    input.nextElementSibling.textContent = input.value;
  }
  const { jobs, waves } = layout(v.cases, v.cores, v.maxArray);
  const throttle = Math.min(Math.max(1, Math.floor(v.maxJobs / v.cores)), jobs);
  const idle = idleShare(v.cases, v.cores, jobs);
  const stats = [
    ["Slurm jobs", jobs],
    ["Waves", waves],
    ["Jobs at once", throttle],
    ["Cases at once", Math.min(throttle * v.cores, v.cases)],
    ["Walltime per job [days]", v.days * waves],
    ["Memory per job [GB]", v.cores * v.mem],
    ["Idle core share", `${(idle * 100).toFixed(1)} %`],
  ];
  const sbu = sbuStats(v, jobs);
  root.querySelector(":scope > .gx-stats").innerHTML = statList(stats);
  root.querySelector(".gx-sbu .gx-stats").innerHTML = statList(sbu.stats);
  root.querySelector(":scope > .gx-warnings").innerHTML = warnings(v, jobs, waves, throttle, idle)
    .map((w) => `<li>${w}</li>`).join("");
  root.querySelector(".gx-sbu .gx-warnings").innerHTML = sbu.out.map((w) => `<li>${w}</li>`).join("");
  const shown = Math.min(jobs, 24);
  let boxes = "";
  for (let j = 0; j < shown; j++) boxes += jobBox(j, v, jobs, j < throttle);
  if (jobs > shown) boxes += `<p class="gx-more">+ ${jobs - shown} more jobs</p>`;
  root.querySelector(".gx-jobs").innerHTML = boxes;
}

function wire() {
  for (const root of document.querySelectorAll(".grid-explorer:not([data-wired])")) {
    root.dataset.wired = "1";
    root.innerHTML = `<div class="gx-controls">${SLIDERS.map(([name, label, min, max, step, value]) =>
      `<label><span>${label}</span><input type="range" name="${name}" min="${min}" max="${max}" step="${step}" value="${value}"><output></output></label>`
    ).join("")}</div>
    <dl class="gx-stats"></dl><ul class="gx-warnings"></ul>
    <details class="gx-sbu"><summary>Snellius SBU cost</summary>
      <label>Partition <select name="partition">${Object.keys(PARTITIONS).map((p) => `<option>${p}</option>`).join("")}</select></label>
      <dl class="gx-stats"></dl><ul class="gx-warnings"></ul>
    </details>
    <div class="gx-jobs"></div>`;
    root.addEventListener("input", () => render(root));
    render(root);
  }
}

document.addEventListener("DOMContentLoaded", wire);
if (window.document$?.subscribe) window.document$.subscribe(wire);
