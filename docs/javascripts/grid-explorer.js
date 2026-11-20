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

function layout(n, k, maxArray) {
  const jobs = Math.min(Math.ceil(n / k), maxArray);
  const waves = Math.ceil(Math.ceil(n / jobs) / k);
  return { jobs, waves };
}

function idleShare(n, k, jobs) {
  const q = Math.floor(n / jobs);
  const r = n % jobs;
  const runs = r * Math.ceil((q + 1) / k) + (jobs - r) * Math.ceil(q / k);
  return 1 - n / (runs * k);
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
  const v = {};
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
  root.querySelector(".gx-stats").innerHTML = stats
    .map(([name, value]) => `<div><dt>${name}</dt><dd>${value}</dd></div>`).join("");
  root.querySelector(".gx-warnings").innerHTML = warnings(v, jobs, waves, throttle, idle)
    .map((w) => `<li>${w}</li>`).join("");
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
    <dl class="gx-stats"></dl><ul class="gx-warnings"></ul><div class="gx-jobs"></div>`;
    root.addEventListener("input", () => render(root));
    render(root);
  }
}

document.addEventListener("DOMContentLoaded", wire);
if (window.document$?.subscribe) window.document$.subscribe(wire);
