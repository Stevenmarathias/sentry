// Sentry per-class concept map (Pass 17, layout fixes Pass 18).
//
// Renders a D3 force-directed graph of:
//   - one centre node = the class itself, themed with the class accent
//   - one node per concept, sized subtly by importance_score and
//     coloured by category
//   - faint "spokes" from centre to each concept so isolated concepts
//     don't float free
//   - relationship edges (Pass 16) drawn brighter, distinct from spokes
//
// Pass 18 layout additions:
//   - Every visual element lives inside a single .map-zoom-root <g> that
//     d3.zoom mutates via transform=. Scroll/pinch zooms; dragging the
//     background pans; dragging a node keeps its Pass-17 behaviour
//     because the zoom is filtered to exclude pointer events that
//     started on .map-node.
//   - Simulation is settled synchronously (one batch of ticks before
//     the browser paints), then an auto-fit transform is computed
//     from the final node bbox EXPANDED to cover label boxes, so the
//     first thing the user sees is the entire graph centred and
//     visible with comfortable padding.
//   - forceCollide radius now includes each label's measured half-width,
//     so horizontally adjacent labels no longer cram into each other.
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
const resetBtn = document.getElementById("map-reset-btn");


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
  // Snapshot the canvas size now; the force simulation and the fit
  // calculation both need numeric dimensions. The svg resizes via CSS
  // but we use these snapshots for forces / zoom transforms.
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(640, rect.width);
  const height = Math.max(560, rect.height || 640);
  svg.attr("viewBox", `0 0 ${width} ${height}`)
     .attr("width", "100%")
     .attr("height", height)
     .style("cursor", "grab");

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

  // Pass 18: a single zoom-root <g> holds every visual element. The
  // d3.zoom() handler below mutates THIS group's transform — nothing
  // else moves. Layer order: edges → nodes → labels (labels stay on
  // top of edges even when nodes overlap briefly).
  const zoomRoot = svg.append("g").attr("class", "map-zoom-root");
  const linkLayer = zoomRoot.append("g").attr("class", "link-layer");
  const nodeLayer = zoomRoot.append("g").attr("class", "node-layer");
  const labelLayer = zoomRoot.append("g").attr("class", "label-layer");

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
  node.append("circle")
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

  // ---- Label-aware collision sizing (Pass 18) ----
  //
  // forceCollide is purely radial, but our labels sit BELOW each node,
  // so two nodes that are vertically near share a collision band and
  // their labels also share a horizontal band. Measuring each label's
  // rendered width and using HALF of it as a horizontal padding budget
  // is a cheap proxy for "don't let labels touch". The labels are
  // already rendered above, so getComputedTextLength is exact.
  const labelHalfWidths = new Map();
  let maxLabelHalfWidth = 0;
  labelLayer.selectAll("text").each(function (d) {
    let w = 80;
    try { w = this.getComputedTextLength(); } catch (e) { /* node not in DOM */ }
    const half = w / 2;
    labelHalfWidths.set(d.id, half);
    if (half > maxLabelHalfWidth) maxLabelHalfWidth = half;
  });

  function collideRadius(d) {
    if (d.isCentre) return 36;
    const r = nodeRadius(d);
    const lhw = labelHalfWidths.get(d.id) || 40;
    // Take the larger of: node-radius + a small gap, OR label-half
    // width + a small gap. Either constraint pushes nodes far enough
    // apart that neither circles nor labels can crash.
    return Math.max(r + 10, lhw + 6);
  }

  // ---- Force simulation (Pass 18 — tuned for spread) ----
  //
  // Notes:
  //   - charge: stronger repulsion than Pass 17 so ~30-node graphs
  //     fill the canvas instead of bunching
  //   - link distance: rel-edges (the spiderweb) stay short to keep
  //     connected concepts clustered; centre-spokes are long+weak so
  //     they anchor outliers without dragging cluster centroids
  //   - centre force: low-strength gravity toward the canvas centre
  //     keeps the cloud from drifting off to one corner over many
  //     ticks (the auto-fit at the end then re-centres for the user)
  //   - collide: label-aware (above)
  const cx = width / 2, cy = height / 2;
  const simulation = d3.forceSimulation(nodes)
    .force("charge",
      d3.forceManyBody().strength((d) => d.isCentre ? -600 : -340))
    .force("link",
      d3.forceLink(links).id((d) => d.id)
        .distance((d) => d.kind === "rel" ? 95 : 210)
        .strength((d) => d.kind === "rel" ? 0.65 : 0.04))
    .force("centre", d3.forceCenter(cx, cy).strength(0.07))
    .force("collide",
      d3.forceCollide().radius(collideRadius).iterations(2))
    .alpha(1);

  // ---- Settle synchronously, then auto-fit ----
  //
  // Run a fixed batch of ticks in one JS frame so the browser doesn't
  // paint until the simulation has cooled. Then compute the fit
  // transform from the final positions (and the measured label box)
  // and apply it through d3.zoom so the user sees the entire graph
  // centred and at the right scale on first paint.
  const SETTLE_TICKS = 360;
  for (let i = 0; i < SETTLE_TICKS; i++) simulation.tick();
  simulation.alpha(0);
  redraw();

  // ---- Zoom + pan ----
  //
  // Standard d3.zoom on the svg, applied to the zoomRoot transform.
  // The .filter excludes pointer events whose target lives inside a
  // .map-node so dragging a node doesn't ALSO pan; node-drag wins.
  // Double-click is filtered out so accidental dbl-clicks don't snap
  // the user to an unexpected zoom level.
  const zoom = d3.zoom()
    .scaleExtent([0.3, 4])
    .filter((event) => {
      if (event.type === "dblclick") return false;
      if (event.type === "wheel") return true;
      // Pointer events: only let zoom-pan handle them if they started
      // on the background, not on a node or label.
      const t = event.target;
      if (t && t.closest && t.closest(".map-node")) return false;
      return true;
    })
    .on("zoom", (event) => {
      zoomRoot.attr("transform", event.transform);
    });
  svg.call(zoom)
     // Background drag cursor cue.
     .on("mousedown.cursor", function () { d3.select(this).style("cursor", "grabbing"); })
     .on("mouseup.cursor",   function () { d3.select(this).style("cursor", "grab"); });

  // Apply the initial auto-fit. computeFitTransform returns a
  // d3.zoomIdentity-based transform; passing it through zoom.transform
  // keeps the fitted view and any subsequent user gestures in the same
  // coordinate system.
  const initialFit = computeFitTransform();
  svg.call(zoom.transform, initialFit);

  // Reset-view button (Pass 18 optional nicety) — animates back to the
  // initial fit. Re-computing each time accounts for nodes the user may
  // have dragged in the meantime.
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      const t = computeFitTransform();
      svg.transition().duration(360).call(zoom.transform, t);
    });
  }

  // After the initial settle, hand off to live ticks so dragging a node
  // smoothly reflows the rest. The auto-fit transform isn't re-applied
  // on tick (that would yank the view around); the user can hit Reset
  // View to re-centre.
  simulation.on("tick", redraw);

  // ---- helpers ----

  function redraw() {
    link
      .attr("x1", (d) => d.source.x)
      .attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x)
      .attr("y2", (d) => d.target.y);
    node.attr("transform", (d) => `translate(${d.x},${d.y})`);
    label
      .attr("x", (d) => d.x)
      .attr("y", (d) => d.y + nodeRadius(d) + 12);
  }

  function computeFitTransform() {
    // Bounding box of node positions, expanded per-node to cover the
    // node radius AND the label that sits below it. The max label
    // half-width (measured above) widens the horizontal extents so
    // outermost-node labels never get clipped at the canvas edge.
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    for (const n of nodes) {
      const r = nodeRadius(n);
      const lhw = labelHalfWidths.get(n.id) || maxLabelHalfWidth;
      // Horizontal: half-label extends past the node centre. Vertical:
      // node radius up, then node radius + label baseline + label height
      // (~12px) down.
      minX = Math.min(minX, n.x - Math.max(r, lhw) - 4);
      maxX = Math.max(maxX, n.x + Math.max(r, lhw) + 4);
      minY = Math.min(minY, n.y - r - 4);
      maxY = Math.max(maxY, n.y + r + 14 + 12);
    }
    const bbW = Math.max(1, maxX - minX);
    const bbH = Math.max(1, maxY - minY);
    // Padding inside the canvas so the graph never kisses the edge.
    const padding = 28;
    const rawScale = Math.min(
      (width  - padding * 2) / bbW,
      (height - padding * 2) / bbH,
    );
    // Clamp to the zoom scaleExtent so d3 doesn't reject the transform.
    const scale = Math.max(0.3, Math.min(4, rawScale));
    const tx = width  / 2 - (minX + bbW / 2) * scale;
    const ty = height / 2 - (minY + bbH / 2) * scale;
    return d3.zoomIdentity.translate(tx, ty).scale(scale);
  }

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
