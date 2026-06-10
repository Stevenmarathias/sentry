// Sentry per-class concept map (Pass 17).
//
// Renders a D3 force-directed graph of:
//   - one centre node = the class itself, themed with the class accent
//   - one node per concept, sized subtly by importance_score and
//     coloured by category
//   - faint "spokes" from centre to each concept so isolated concepts
//     don't float free
//   - relationship edges (Pass 16) drawn brighter, distinct from spokes
//
// Click a concept node = navigate to the existing Pass 7 concept detail
// page (full reuse of the in-depth explanation flow).
//
// Empty state: when no relationships have been generated yet, the
// canvas hides and the "Generate map" button POSTs to the existing
// /class/<n>/relationships/generate route, then reloads the page.

// ---- Category palette (synced with the legend in the template) ------------
//
// Spec mapping: person purple, term blue, framework teal, technique amber,
// event coral, claim pink; everything else neutral slate.
const CATEGORY_COLORS = {
  person:    "#c084fc",
  term:      "#60a5fa",
  framework: "#2dd4bf",
  technique: "#fbbf24",
  event:     "#fb7185",
  claim:     "#f472b6",
  formula:   "#a78bfa",   // not in the spec's six but appears in concepts.json
  other:     "#94a3b8",
};
function colorFor(category) {
  return CATEGORY_COLORS[category] || CATEGORY_COLORS.other;
}

// ---- Data ingest ----------------------------------------------------------

const dataNode = document.getElementById("map-data");
const className = document.body.dataset.className || "";
const raw = JSON.parse(dataNode.textContent);
const concepts = raw.concepts || [];
const storedEdges = raw.edges || [];

const canvas = document.getElementById("map-canvas");
const svg = d3.select("#map-svg");
const emptyState = document.getElementById("map-empty");
const generateBtn = document.getElementById("map-generate-btn");
const emptyStatus = document.getElementById("map-empty-status");
const legend = document.getElementById("map-legend");


// ---- Empty state ----------------------------------------------------------

function showEmptyState() {
  if (canvas) canvas.hidden = true;
  if (emptyState) emptyState.hidden = false;
}

async function generateMap() {
  if (!generateBtn) return;
  generateBtn.disabled = true;
  generateBtn.textContent = "Generating…";
  if (emptyStatus) {
    emptyStatus.hidden = false;
    emptyStatus.textContent =
      "Asking Claude to derive the relationships — usually under a minute.";
    emptyStatus.classList.remove("error");
  }
  try {
    const res = await fetch(
      `/class/${encodeURIComponent(className)}/relationships/generate`,
      { method: "POST", headers: { "Content-Type": "application/json" } },
    );
    const data = await res.json();
    if (data.ok) {
      window.location.reload();
      return;
    }
    if (emptyStatus) {
      emptyStatus.textContent = data.error || "Could not generate the map.";
      emptyStatus.classList.add("error");
    }
  } catch (err) {
    if (emptyStatus) {
      emptyStatus.textContent = "Could not reach server.";
      emptyStatus.classList.add("error");
    }
  } finally {
    generateBtn.disabled = false;
    generateBtn.textContent = "Generate map";
  }
}

if (generateBtn) generateBtn.addEventListener("click", generateMap);


// ---- No-edges short-circuit ------------------------------------------------

if (!concepts.length || !storedEdges.length) {
  // Nothing to draw yet. The empty-state explains what's happening and
  // gives the user the Generate button — same URL the backend cron / dev
  // tools use, just from the UI.
  showEmptyState();
} else {
  renderGraph();
}


// ---- Graph rendering -------------------------------------------------------

function renderGraph() {
  // Snapshot the canvas size now; d3 needs a numeric width/height for the
  // centring force. The svg will resize via CSS but the simulation uses
  // these snapshots throughout the lifetime of the page.
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(640, rect.width);
  const height = Math.max(560, rect.height || 640);
  svg.attr("viewBox", `0 0 ${width} ${height}`)
     .attr("width", "100%")
     .attr("height", height);

  // Build the node + link arrays D3 will mutate in-place (x/y/vx/vy on
  // each node, source/target ref-swapping on each link).
  const centreId = "__class__";
  const nodes = [
    { id: centreId, isCentre: true, name: raw.class_name || className,
      importance: 0 },
    ...concepts.map((c) => ({
      id:         c.name,
      name:       c.name,
      category:   c.category || "other",
      importance: Number(c.importance) || 0,
      isCentre:   false,
    })),
  ];
  // Faint centre-spokes so isolated concepts don't float free.
  const spokeLinks = concepts.map((c) => ({
    source: centreId, target: c.name, kind: "spoke",
  }));
  // Brighter Pass-16 edges. Each edge is between two concepts that are
  // present in the node list (Pass 16 already filtered phantom endpoints).
  const relLinks = storedEdges.map((e) => ({
    source: e.from, target: e.to, reason: e.reason || "", kind: "rel",
  }));
  const links = spokeLinks.concat(relLinks);

  // Build a category legend from the categories actually present so we
  // don't show swatches for buckets the class hasn't used.
  buildLegend(nodes);

  // Layers (back to front): edges → nodes → labels. Drawing labels last
  // keeps them on top of the strokes even when nodes overlap briefly.
  const linkLayer = svg.append("g").attr("class", "link-layer");
  const nodeLayer = svg.append("g").attr("class", "node-layer");
  const labelLayer = svg.append("g").attr("class", "label-layer");

  const link = linkLayer.selectAll("line")
    .data(links)
    .enter()
    .append("line")
    .attr("class", (d) => `map-edge map-edge-${d.kind}`);

  // Concept-to-concept edges show their reason as a native tooltip on hover.
  link.filter((d) => d.kind === "rel")
    .append("title")
    .text((d) => d.reason);

  const accent = getComputedStyle(document.body)
    .getPropertyValue("--class-accent").trim() || "#2f7bff";

  const node = nodeLayer.selectAll("g")
    .data(nodes)
    .enter()
    .append("g")
    .attr("class", (d) => d.isCentre ? "map-node map-centre" : "map-node")
    .style("cursor", (d) => d.isCentre ? "default" : "pointer")
    .call(d3.drag()
      .on("start", dragStarted)
      .on("drag", dragged)
      .on("end", dragEnded));

  // Concept nodes navigate to the Pass-7 detail page on click. Centre
  // node is the class — clicking it does nothing (the class home is one
  // back-link away in the header).
  node.filter((d) => !d.isCentre)
    .on("click", (event, d) => {
      window.location =
        `/class/${encodeURIComponent(className)}/concept/${encodeURIComponent(d.id)}`;
    });

  // Node circle. Importance maps to radius via a clamped linear scale so
  // a single mention concept doesn't dwarf a 15-mention one.
  function nodeRadius(d) {
    if (d.isCentre) return 24;
    const imp = Math.max(0, Math.min(d.importance, 12));
    return 6 + imp * 0.9;       // 6 → 16.8 across imp 0 → 12
  }
  const circle = node.append("circle")
    .attr("r", nodeRadius)
    .attr("fill", (d) => d.isCentre ? accent : colorFor(d.category))
    .attr("stroke", (d) => d.isCentre ? accent : "rgba(255,255,255,0.18)")
    .attr("stroke-width", (d) => d.isCentre ? 2 : 1);

  // Subtle glow ring on hover so the tappable area is obvious without
  // adding shadow chrome to every node.
  node.filter((d) => !d.isCentre)
    .on("mouseenter", function () {
      d3.select(this).select("circle")
        .transition().duration(120)
        .attr("stroke", "rgba(255,255,255,0.55)")
        .attr("stroke-width", 2);
    })
    .on("mouseleave", function () {
      d3.select(this).select("circle")
        .transition().duration(160)
        .attr("stroke", "rgba(255,255,255,0.18)")
        .attr("stroke-width", 1);
    });

  // Label: concept name sits just below the node. The label layer is the
  // last child so labels paint over any overlapping strokes.
  const label = labelLayer.selectAll("text")
    .data(nodes)
    .enter()
    .append("text")
    .attr("class", (d) => d.isCentre ? "map-label map-label-centre" : "map-label")
    .attr("text-anchor", "middle")
    .text((d) => d.name);

  // ---- Force simulation ----
  //
  // Tuning notes:
  //   - charge: strong negative repulsion so dense clusters separate
  //   - link distance: short on relationship edges (they're the spiderweb;
  //     should pull connected concepts together) and longer + much weaker
  //     on spokes (centring force without dragging connected clusters
  //     toward the centre)
  //   - x/y centring force keeps the whole thing on the canvas
  const simulation = d3.forceSimulation(nodes)
    .force("charge",
      d3.forceManyBody().strength((d) => d.isCentre ? -400 : -240))
    .force("link",
      d3.forceLink(links).id((d) => d.id)
        .distance((d) => d.kind === "rel" ? 90 : 220)
        .strength((d) => d.kind === "rel" ? 0.6 : 0.05))
    .force("centre", d3.forceCenter(width / 2, height / 2))
    .force("collide",
      d3.forceCollide().radius((d) => nodeRadius(d) + 14))
    .alpha(1)
    .on("tick", () => {
      link
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);
      node.attr("transform", (d) => `translate(${d.x},${d.y})`);
      label
        .attr("x", (d) => d.x)
        .attr("y", (d) => d.y + nodeRadius(d) + 12);
    });

  // Idle the simulation after it settles. D3 handles this automatically
  // via alpha decay; we just don't restart on every interaction.

  function dragStarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x; d.fy = d.y;
  }
  function dragged(event, d) {
    d.fx = event.x; d.fy = event.y;
  }
  function dragEnded(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    // Release the pin so the node settles back into the flow when the
    // user lets go. Comment out these two lines to make drags sticky.
    d.fx = null; d.fy = null;
  }
}


// ---- Legend ---------------------------------------------------------------

function buildLegend(nodes) {
  if (!legend) return;
  const present = new Set();
  for (const n of nodes) {
    if (!n.isCentre && n.category) present.add(n.category);
  }
  // Render categories in a stable order so the legend doesn't jitter
  // class-to-class.
  const order = [
    "person", "term", "framework", "technique",
    "event", "claim", "formula", "other",
  ];
  legend.innerHTML = "";
  for (const cat of order) {
    if (!present.has(cat)) continue;
    const li = document.createElement("li");
    li.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.background = colorFor(cat);
    const label = document.createElement("span");
    label.className = "legend-label";
    label.textContent = cat;
    li.appendChild(swatch);
    li.appendChild(label);
    legend.appendChild(li);
  }
}
