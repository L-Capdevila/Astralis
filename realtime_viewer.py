"""
core/realtime_viewer.py — Visualisation temps réel de la simulation N-corps.

Modes disponibles :
  "matplotlib" : fenêtre matplotlib mise à jour en direct (thread dédié)
  "web"        : serveur Flask + Three.js dans le navigateur (http://localhost:PORT)
  "both"       : les deux simultanément

Interface attendue par Simulation_Orbite.py :
  viewer = RealtimeViewer(mode="matplotlib", port=5050)
  viewer.start(n_bodies=3, body_names=["Soleil", "Terre", "Mars"])
  viewer.send(state, last_H, last_L, dt)   # appelé toutes les REALTIME_EVERY frames
  viewer.stop()

Dépendances :
  matplotlib          (mode matplotlib)
  flask + flask-cors  (mode web)   — pip install flask flask-cors
"""

from __future__ import annotations

import threading
import queue
import time
import math
from typing import Optional

import numpy as np

# ── Palette couleurs ──────────────────────────────────────────────────────────
PALETTE = [
    "#FFD700",  # étoile — or
    "#4FC3F7",  # bleu clair
    "#EF5350",  # rouge
    "#66BB6A",  # vert
    "#AB47BC",  # violet
    "#FF7043",  # orange
    "#26C6DA",  # cyan
    "#D4E157",  # jaune-vert
]


# ══════════════════════════════════════════════════════════════════════════════
#  VIEWER MATPLOTLIB (fenêtre locale)
# ══════════════════════════════════════════════════════════════════════════════

class _MatplotlibViewer:
    """Fenêtre matplotlib mise à jour en temps réel dans un thread dédié."""

    def __init__(self):
        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.n_bodies = 0
        self.body_names: list[str] = []

    def start(self, n_bodies: int, body_names: list[str]):
        self.n_bodies = n_bodies
        self.body_names = body_names
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def send(self, state, last_H: float, last_L: float, dt: float):
        """Envoie un snapshot non-bloquant (drop si la queue est pleine)."""
        snap = {
            "pos":    state.pos.copy(),
            "vel":    state.vel.copy(),
            "t":      float(state.t),
            "H":      float(last_H) if last_H is not None else math.nan,
            "L":      float(last_L) if last_L is not None else math.nan,
            "dt":     float(dt),
            "E0":     float(state.E0),
        }
        try:
            self._q.put_nowait(snap)
        except queue.Full:
            pass  # on sacrifie le frame, pas la simulation

    def stop(self):
        self._running = False

    # ── Thread matplotlib ─────────────────────────────────────────────────────
    def _run(self):
        import matplotlib
        matplotlib.use("TkAgg")          # backend GUI non-bloquant
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from collections import deque

        MAX_TRAIL = 300                  # points de traîne par corps

        fig = plt.figure(figsize=(8, 8), facecolor="#0d1117")
        fig.canvas.manager.set_window_title("Simulation N-corps — Vue temps réel")

        ax_xy = fig.add_subplot(111, facecolor="#0d1117")
        ax_xy.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax_xy.spines.values():
            spine.set_color("#30363d")
        ax_xy.set_title("Orbites — Plan XY (UA)", color="#c9d1d9", fontsize=9, pad=6)

        # Historiques
        trails   = [{"x": deque(maxlen=MAX_TRAIL),
                      "y": deque(maxlen=MAX_TRAIL),
                      "z": deque(maxlen=MAX_TRAIL)}
                    for _ in range(self.n_bodies)]
        t_hist   = deque(maxlen=2000)
        dh_hist  = deque(maxlen=2000)
        v_hist   = {i: deque(maxlen=2000) for i in range(self.n_bodies)}

        snap: Optional[dict] = None
        UA = 1.496e11

        def _update(_frame):
            nonlocal snap
            # Vider la queue, garder le plus récent
            while True:
                try:
                    snap = self._q.get_nowait()
                except queue.Empty:
                    break

            if snap is None:
                return

            pos  = snap["pos"]
            vel  = snap["vel"]
            t_s  = snap["t"]
            H    = snap["H"]
            E0   = snap["E0"]
            t_yr = t_s / (365.25 * 86400)

            # Mettre à jour traînes
            for i in range(self.n_bodies):
                trails[i]["x"].append(pos[i, 0] / UA)
                trails[i]["y"].append(pos[i, 1] / UA)
                trails[i]["z"].append(pos[i, 2] / UA)

            # Dérive énergie
            if E0 != 0 and not math.isnan(H):
                dh_hist.append((H - E0) / abs(E0) * 100)
                t_hist.append(t_yr)

            # Vitesses
            for i in range(self.n_bodies):
                v = math.sqrt(vel[i,0]**2 + vel[i,1]**2 + vel[i,2]**2) / 1e3
                v_hist[i].append(v)

            # ── Tracer XY ────────────────────────────────────────────────────
            ax_xy.cla()
            ax_xy.set_facecolor("#0d1117")
            ax_xy.tick_params(colors="#8b949e", labelsize=8)
            for spine in ax_xy.spines.values():
                spine.set_color("#30363d")
            ax_xy.set_xlabel("X (UA)", color="#8b949e", fontsize=8)
            ax_xy.set_ylabel("Y (UA)", color="#8b949e", fontsize=8)

            for i in range(self.n_bodies):
                col  = PALETTE[i % len(PALETTE)]
                name = self.body_names[i] if i < len(self.body_names) else f"Corps {i}"
                tx   = list(trails[i]["x"])
                ty   = list(trails[i]["y"])
                if len(tx) > 1:
                    ax_xy.plot(tx, ty, color=col, lw=0.8, alpha=0.5)
                ax_xy.scatter([pos[i, 0]/UA], [pos[i, 1]/UA],
                              color=col, s=60 if i == 0 else 35,
                              zorder=5, edgecolors="#ffffff", linewidths=0.4,
                              label=name)

            ax_xy.set_aspect("equal", adjustable="datalim")
            ax_xy.legend(fontsize=8, loc="upper right",
                         labelcolor="#c9d1d9", framealpha=0.2,
                         facecolor="#161b22", edgecolor="#30363d")

            dh_str = ""
            if E0 != 0 and not math.isnan(H):
                dh = (H - E0) / abs(E0) * 100
                dh_str = f"  |  ΔH/H₀ = {dh:.2e} %"
            ax_xy.set_title(
                f"Orbites XY  —  t = {t_yr:.3f} ans{dh_str}",
                color="#c9d1d9", fontsize=9, pad=6
            )

        anim = FuncAnimation(fig, _update, interval=200, cache_frame_data=False)
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  VIEWER WEB (Flask + Three.js)
# ══════════════════════════════════════════════════════════════════════════════

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Simulation N-corps — Temps réel</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0d1117; color: #c9d1d9; font-family: 'Courier New', monospace;
         display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
  #header { padding: 8px 16px; background: #161b22; border-bottom: 1px solid #30363d;
            display: flex; align-items: center; gap: 16px; font-size: 12px; flex-shrink: 0; }
  #header h1 { font-size: 13px; color: #FFD700; letter-spacing: 1px; }
  .stat { color: #8b949e; }
  .stat span { color: #4FC3F7; font-weight: bold; }
  #canvas-container { flex: 1; position: relative; }
  canvas { display: block; width: 100%; height: 100%; }
  #legend { position: absolute; top: 12px; right: 12px; background: rgba(22,27,34,0.85);
            border: 1px solid #30363d; border-radius: 6px; padding: 8px 12px; font-size: 11px; }
  .leg-item { display: flex; align-items: center; gap: 6px; margin: 3px 0; }
  .leg-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  #status { position: absolute; bottom: 8px; left: 12px; font-size: 10px; color: #8b949e; }
</style>
</head>
<body>
<div id="header">
  <h1>⚡ SIMULATION N-CORPS</h1>
  <div class="stat">t = <span id="st-t">0.000</span> ans</div>
  <div class="stat">ΔH/H₀ = <span id="st-dh">—</span></div>
  <div class="stat">dt = <span id="st-dt">—</span> s</div>
  <div class="stat">|L| = <span id="st-l">—</span></div>
</div>
<div id="canvas-container">
  <canvas id="c"></canvas>
  <div id="legend"></div>
  <div id="status">En attente de données…</div>
</div>

<script>
const PALETTE = ["#FFD700","#4FC3F7","#EF5350","#66BB6A","#AB47BC","#FF7043","#26C6DA","#D4E157"];
const MAX_TRAIL = 400;

let bodies = [];
let trails = [];
let bodyNames = [];
let nBodies = 0;
let animFrame = null;
let lastSnap = null;

// ── Canvas 2D (projection XY) ─────────────────────────────────────────────
const canvas = document.getElementById("c");
const ctx    = canvas.getContext("2d");

function resize() {
  canvas.width  = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
}
window.addEventListener("resize", resize);
resize();

// Zoom / pan
let scale  = 80;   // px / UA
let offsetX = 0;
let offsetY = 0;
let dragging = false;
let dragStart = {x:0, y:0};

canvas.addEventListener("wheel", e => {
  e.preventDefault();
  scale *= e.deltaY < 0 ? 1.12 : 0.89;
  scale = Math.max(2, Math.min(scale, 5000));
}, {passive: false});

canvas.addEventListener("mousedown", e => { dragging = true; dragStart = {x:e.clientX, y:e.clientY}; });
canvas.addEventListener("mouseup",   () => dragging = false);
canvas.addEventListener("mousemove", e => {
  if (!dragging) return;
  offsetX += e.clientX - dragStart.x;
  offsetY += e.clientY - dragStart.y;
  dragStart = {x: e.clientX, y: e.clientY};
});

function toScreen(x, y) {
  return {
    sx: canvas.width/2  + x * scale + offsetX,
    sy: canvas.height/2 - y * scale + offsetY,
  };
}

function drawFrame() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Fond étoilé (statique, généré une fois)
  if (!drawFrame._stars) {
    drawFrame._stars = Array.from({length: 180}, () => ({
      x: Math.random(), y: Math.random(),
      r: Math.random() * 1.2 + 0.3,
      a: Math.random() * 0.7 + 0.2,
    }));
  }
  drawFrame._stars.forEach(s => {
    ctx.beginPath();
    ctx.arc(s.x * canvas.width, s.y * canvas.height, s.r, 0, Math.PI*2);
    ctx.fillStyle = `rgba(255,255,255,${s.a})`;
    ctx.fill();
  });

  // Grille
  ctx.strokeStyle = "rgba(48,54,61,0.5)";
  ctx.lineWidth = 0.5;
  const gridStep = scale;
  const ox = (canvas.width/2 + offsetX) % gridStep;
  const oy = (canvas.height/2 + offsetY) % gridStep;
  for (let x = ox; x < canvas.width;  x += gridStep) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,canvas.height); ctx.stroke(); }
  for (let y = oy; y < canvas.height; y += gridStep) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(canvas.width,y);  ctx.stroke(); }

  // Axes
  const {sx: ax0, sy: ay0} = toScreen(0,0);
  ctx.strokeStyle = "rgba(139,148,158,0.3)";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(ax0, 0); ctx.lineTo(ax0, canvas.height); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, ay0); ctx.lineTo(canvas.width, ay0); ctx.stroke();

  if (nBodies === 0) return;

  // Traînes + corps
  for (let i = 0; i < nBodies; i++) {
    const col  = PALETTE[i % PALETTE.length];
    const trail = trails[i];
    if (!trail || trail.length < 2) continue;

    // Traîne avec alpha dégradé
    for (let k = 1; k < trail.length; k++) {
      const alpha = (k / trail.length) * 0.6;
      const {sx: x1, sy: y1} = toScreen(trail[k-1][0], trail[k-1][1]);
      const {sx: x2, sy: y2} = toScreen(trail[k][0],   trail[k][1]);
      ctx.beginPath();
      ctx.strokeStyle = col + Math.round(alpha*255).toString(16).padStart(2,"0");
      ctx.lineWidth = 1;
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }

    // Corps
    const {sx, sy} = toScreen(bodies[i][0], bodies[i][1]);
    const radius = i === 0 ? 8 : 5;
    const grd = ctx.createRadialGradient(sx, sy, 0, sx, sy, radius*2);
    grd.addColorStop(0, col);
    grd.addColorStop(1, col + "00");
    ctx.beginPath();
    ctx.arc(sx, sy, radius*2, 0, Math.PI*2);
    ctx.fillStyle = grd;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(sx, sy, radius, 0, Math.PI*2);
    ctx.fillStyle = col;
    ctx.shadowColor = col;
    ctx.shadowBlur  = 12;
    ctx.fill();
    ctx.shadowBlur  = 0;

    // Label
    ctx.fillStyle   = col;
    ctx.font        = "10px 'Courier New'";
    ctx.fillText(bodyNames[i] || `Corps ${i}`, sx + radius + 4, sy - 4);
  }

  animFrame = requestAnimationFrame(drawFrame);
}

// ── Polling serveur ───────────────────────────────────────────────────────
async function poll() {
  try {
    const r = await fetch("/state");
    if (!r.ok) return;
    const d = await r.json();

    if (nBodies === 0 && d.n_bodies > 0) {
      nBodies    = d.n_bodies;
      bodyNames  = d.body_names;
      bodies     = Array.from({length: nBodies}, () => [0,0,0]);
      trails     = Array.from({length: nBodies}, () => []);

      // Légende (DOM sécurisé — pas d'innerHTML avec noms utilisateur)
      const leg = document.getElementById("legend");
      leg.replaceChildren();
      bodyNames.forEach((n, i) => {
        const item = document.createElement("div");
        item.className = "leg-item";
        const dot = document.createElement("div");
        dot.className = "leg-dot";
        dot.style.background = PALETTE[i % PALETTE.length];
        const label = document.createElement("span");
        label.textContent = n;
        item.appendChild(dot);
        item.appendChild(label);
        leg.appendChild(item);
      });
    }

    if (d.pos && nBodies > 0) {
      for (let i = 0; i < nBodies; i++) {
        bodies[i] = d.pos[i];
        trails[i].push([d.pos[i][0], d.pos[i][1]]);
        if (trails[i].length > MAX_TRAIL) trails[i].shift();
      }
    }

    // Stats
    const t_yr = d.t / (365.25 * 86400);
    document.getElementById("st-t").textContent  = t_yr.toFixed(3);
    document.getElementById("st-dt").textContent = d.dt ? d.dt.toFixed(1) : "—";
    if (d.H !== null && d.E0 && d.E0 !== 0) {
      const dh = (d.H - d.E0) / Math.abs(d.E0) * 100;
      const el = document.getElementById("st-dh");
      el.textContent = dh.toExponential(2) + " %";
      el.style.color = Math.abs(dh) < 1e-4 ? "#66BB6A" : "#EF5350";
    }
    if (d.L !== null) {
      document.getElementById("st-l").textContent = d.L.toExponential(3);
    }
    document.getElementById("status").textContent =
      `Dernière mise à jour : ${new Date().toLocaleTimeString()}  |  frame ${d.frame || 0}`;

  } catch(e) {}
}

// Démarrer rendu + polling
drawFrame();
setInterval(poll, 250);
</script>
</body>
</html>
"""


class _WebViewer:
    """Serveur Flask minimal qui pousse l'état via polling JSON."""

    def __init__(self, port: int = 5050):
        self.port = port
        self._state: dict = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def start(self, n_bodies: int, body_names: list[str]):
        with self._lock:
            self._state = {
                "n_bodies": n_bodies,
                "body_names": body_names,
                "pos": [[0.0, 0.0, 0.0]] * n_bodies,
                "t": 0.0, "H": None, "L": None,
                "E0": None, "dt": None, "frame": 0,
            }
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        time.sleep(0.5)   # laisser Flask démarrer

        # Ouvrir le navigateur automatiquement
        try:
            import webbrowser
            webbrowser.open(f"http://localhost:{self.port}")
        except Exception:
            pass

    def send(self, state, last_H: float, last_L: float, dt: float):
        UA = 1.496e11
        with self._lock:
            self._state["pos"]   = [[float(state.pos[i, k] / UA) for k in range(3)]
                                     for i in range(state.n)]
            self._state["t"]     = float(state.t)
            self._state["H"]     = float(last_H) if last_H is not None else None
            self._state["L"]     = float(last_L) if last_L is not None else None
            self._state["E0"]    = float(state.E0)
            self._state["dt"]    = float(dt)
            self._state["frame"] = self._state.get("frame", 0) + 1

    def stop(self):
        pass   # le thread est daemon, il s'arrête avec le processus

    def _serve(self):
        try:
            from flask import Flask, jsonify, Response
            from flask_cors import CORS
        except ImportError:
            print("  ⚠ flask/flask-cors manquant — pip install flask flask-cors")
            return

        app = Flask(__name__)
        CORS(app)

        @app.route("/")
        def index():
            return Response(_HTML_TEMPLATE, mimetype="text/html")

        @app.route("/state")
        def get_state():
            with self._lock:
                return jsonify(dict(self._state))

        import logging
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)   # silence les logs de requêtes

        print(f"  🌐 Viewer web → http://localhost:{self.port}")
        app.run(host="127.0.0.1", port=self.port, threaded=True, use_reloader=False)


# ══════════════════════════════════════════════════════════════════════════════
#  FACADE PUBLIQUE
# ══════════════════════════════════════════════════════════════════════════════

class RealtimeViewer:
    """
    Point d'entrée unique pour la visualisation temps réel.

    Modes : "matplotlib", "web", "both"
    """

    def __init__(self, mode: str = "matplotlib", port: int = 5050):
        self.mode = mode
        self._mpl: Optional[_MatplotlibViewer] = None
        self._web: Optional[_WebViewer] = None

        if mode in ("matplotlib", "both"):
            self._mpl = _MatplotlibViewer()
        if mode in ("web", "both"):
            self._web = _WebViewer(port=port)

    def start(self, n_bodies: int, body_names: list[str]):
        if self._mpl:
            self._mpl.start(n_bodies, body_names)
        if self._web:
            self._web.start(n_bodies, body_names)

    def send(self, state, last_H: float, last_L: float, dt: float):
        if self._mpl:
            self._mpl.send(state, last_H, last_L, dt)
        if self._web:
            self._web.send(state, last_H, last_L, dt)

    def stop(self):
        if self._mpl:
            self._mpl.stop()
        if self._web:
            self._web.stop()
