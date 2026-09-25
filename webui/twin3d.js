/* DEPTH — 3D digital twin (three.js r160, vendored; zero-build ES module).
   Two physics-grounded scenes built from the server's measured geometry (src/agentic/twin.py):
     • frame twin  — one sonar frame in across-track GROUND range: Stage-1 remap as the seabed, the
                     tracked sonar altitude per ping, finds at their ping/ground range with heights from
                     their measured shadows; picking a find draws the acoustic ray triangle.
     • survey twin — the swath ribbons laid along the boat track, hazards, routes, re-survey passes, and a
                     replay where the boat sweeps its sonar fans and finds appear as it passes them.
   Camera modes (like DepthWizard): orbit · fly (WASD/QE) · top. Render: textured · wireframe;
   backscatter relief (NOT bathymetry); vertical exaggeration. window.DepthTwin.create(el) → viewer. */
import * as THREE from "three";
import { OrbitControls } from "./vendor/three/addons/OrbitControls.js";

const VCOL = { confirmed: 0x2ea043, review: 0xd9a441, low_risk: 0x6b7a88, rejected: 0x6b7a88 };
const CYAN = 0x1fb6d5, VIOLET = 0xb3a8ff;

class TwinViewer {
  constructor(el, opts = {}) {
    this.el = el; this.opts = opts;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    el.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x070b10);
    this.scene.fog = new THREE.FogExp2(0x070b10, 0.006);
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.05, 5000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true; this.controls.dampingFactor = 0.08; this.controls.screenSpacePanning = true;
    this.scene.add(new THREE.HemisphereLight(0xbfdcff, 0x0b1016, 0.9));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4); sun.position.set(-40, 80, -30); this.scene.add(sun);
    this.root = new THREE.Group(); this.scene.add(this.root);
    this.pickables = []; this.labels = []; this.keys = {};
    this.cameraMode = "orbit"; this.wire = false; this.exag = 1; this.relief = 0; this.play = false;
    this.clock = new THREE.Clock();
    this.ray = new THREE.Raycaster(); this.mouse = new THREE.Vector2();
    this.hud = document.createElement("div"); this.hud.className = "tw-labels"; el.appendChild(this.hud);
    this._wire();
    new ResizeObserver(() => this.resize()).observe(el); this.resize();
    const loop = () => { this._raf = requestAnimationFrame(loop); this._tick(); };
    loop();
  }

  _wire() {
    const dom = this.renderer.domElement;
    let down = null;
    dom.addEventListener("pointerdown", e => { down = [e.clientX, e.clientY]; });
    dom.addEventListener("pointerup", e => {
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;   // a drag, not a click
      const r = dom.getBoundingClientRect();
      this.mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      this.ray.setFromCamera(this.mouse, this.camera);
      const hit = this.ray.intersectObjects(this.pickables, false)[0];
      if (hit && hit.object.userData.pick) { this.select(hit.object.userData.pick.id, true); }
    });
    window.addEventListener("keydown", e => { if (this.cameraMode === "fly" && this._visible()) this.keys[e.code] = true; });
    window.addEventListener("keyup", e => { this.keys[e.code] = false; });
  }
  _visible() { return this.el.offsetParent !== null; }
  resize() {
    const w = this.el.clientWidth || 1, h = this.el.clientHeight || 1;
    this.renderer.setSize(w, h, false); this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }
  clear() {
    this.root.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material) [].concat(o.material).forEach(m => { if (m.map) m.map.dispose(); m.dispose(); }); });
    this.scene.remove(this.root); this.root = new THREE.Group(); this.scene.add(this.root);
    this.pickables = []; this.labels = []; this.hud.innerHTML = ""; this.sel = null; this.boat = null; this.play = false;
  }

  /* ------------------------------------------------------------------ frame twin (units: px × k) */
  showFrame(tw) {
    this.clear(); this.mode = "frame"; this.tw = tw;
    const k = 0.1, W = tw.n_pings * k, G = tw.n_ground * k;
    this.k = k; this.scene.fog.density = 0.55 / Math.max(W, G);
    const tex = new THREE.TextureLoader().load(tw.texture); tex.colorSpace = THREE.SRGBColorSpace;
    this.pix = null;
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas"); c.width = img.width; c.height = img.height;
      const cx = c.getContext("2d"); cx.drawImage(img, 0, 0);
      this.pix = { w: img.width, h: img.height, d: cx.getImageData(0, 0, img.width, img.height).data };
      if (this.relief) { this._buildAltitude(); this._buildObjects(); }
    };
    img.src = tw.texture;
    const geo = new THREE.PlaneGeometry(W, G, 256, 256); geo.rotateX(-Math.PI / 2); geo.translate(W / 2, 0, G / 2);
    this.seabedMat = new THREE.MeshStandardMaterial({ map: tex, displacementMap: tex, displacementScale: 0, roughness: 0.95, metalness: 0 });
    this.seabed = new THREE.Mesh(geo, this.seabedMat); this.root.add(this.seabed);
    // the sonar path at the TRACKED altitude (per ping) + altitude sticks down to the seabed
    this.altLine = new THREE.Group(); this.root.add(this.altLine); this._buildAltitude();
    // finds
    this.objGroup = new THREE.Group(); this.root.add(this.objGroup); this._buildObjects();
    const grid = new THREE.GridHelper(Math.max(W, G) * 1.6, 32, 0x1b2a38, 0x111a23); grid.position.set(W / 2, -0.05, G / 2); this.root.add(grid);
    this.home = { pos: new THREE.Vector3(W * 0.5, Math.max(W, G) * 0.78, -G * 0.28), target: new THREE.Vector3(W / 2, 0, G * 0.5) };
    this.goHome();
    this.setRelief(this.relief);
  }
  /* relief height (scene units) of the displaced seabed at (ping px, ground px) — 0 when relief is off */
  _base(ping, ground) {
    if (!this.relief || !this.pix || this.mode !== "frame") return 0;
    const u = Math.min(this.pix.w - 1, Math.max(0, Math.round(ping / this.tw.n_pings * (this.pix.w - 1))));
    const v = Math.min(this.pix.h - 1, Math.max(0, Math.round(ground / this.tw.n_ground * (this.pix.h - 1))));
    return this.pix.d[(v * this.pix.w + u) * 4] / 255 * this.relief;
  }
  _buildAltitude() {
    const tw = this.tw, k = this.k, e = this.exag;
    this.altLine.clear();
    const pts = tw.altitude.pings.map((p, i) => new THREE.Vector3(p * k, tw.altitude.px[i] * k * e, 0));
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: CYAN }));
    this.altLine.add(line);
    const sticks = [];
    pts.forEach((p, i) => { if (i % 8 === 0) sticks.push(p.clone(), new THREE.Vector3(p.x, 0, 0)); });
    const st = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(sticks),
      new THREE.LineDashedMaterial({ color: CYAN, dashSize: 0.3, gapSize: 0.3, transparent: true, opacity: 0.35 }));
    st.computeLineDistances(); this.altLine.add(st);
  }
  _buildObjects() {
    const tw = this.tw, k = this.k, e = this.exag;
    this.objGroup.clear(); this.pickables = this.pickables.filter(o => o.parent && o.parent !== this.objGroup);
    tw.objects.forEach(o => {
      const w = Math.max(0.4, (o.along[1] - o.along[0]) * k), d = Math.max(0.4, (o.across[1] - o.across[0]) * k);
      const h = o.height_px != null ? Math.max(0.15, o.height_px * k * e) : 0.12;
      const col = VCOL[o.verdict] || 0x8ba6c2;
      const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d),
        new THREE.MeshStandardMaterial({ color: col, emissive: col, emissiveIntensity: 0.35, transparent: true, opacity: o.height_px != null ? 0.92 : 0.55, roughness: 0.5 }));
      const base = this._base(o.ping, (o.across[0] + o.across[1]) / 2);
      m.position.set(o.ping * k, base + h / 2, (o.across[0] + o.across[1]) / 2 * k);
      m.userData.pick = o; this.objGroup.add(m); this.pickables.push(m);
      const edge = new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry), new THREE.LineBasicMaterial({ color: col }));
      edge.position.copy(m.position); this.objGroup.add(edge);
      if (o.shadow) {                          // the measured acoustic shadow on the seabed, beyond the find
        const sd = Math.max(0.2, (o.shadow.g1 - o.shadow.g0) * k);
        const sq = new THREE.Mesh(new THREE.PlaneGeometry(Math.max(0.3, w * 0.6), sd),
          new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.55, depthWrite: false }));
        sq.rotation.x = -Math.PI / 2; sq.position.set(o.ping * k, this._base(o.ping, (o.shadow.g0 + o.shadow.g1) / 2) + 0.04, (o.shadow.g0 + o.shadow.g1) / 2 * k);
        this.objGroup.add(sq);
      }
    });
    this._restoreSel();
  }
  _restoreSel() {                         // re-highlight + redraw the ray WITHOUT moving the camera
    if (this.selId == null || !this.tw) return;
    const o = this.tw.objects.find(x => x.id === this.selId); if (!o) return;
    this.labels.forEach(l => l.el.remove()); this.labels = [];
    this.pickables.forEach(m => { m.material.emissiveIntensity = m.userData.pick.id === this.selId ? 1.0 : 0.3; });
    this._drawRay(o);
  }
  /* the acoustic triangle the height came from: sonar → top of the find → end of its shadow */
  _drawRay(o) {
    if (this.rayGroup) { this.root.remove(this.rayGroup); }
    this.rayGroup = new THREE.Group(); this.root.add(this.rayGroup);
    const k = this.k, e = this.exag, x = o.ping * k;
    const S = new THREE.Vector3(x, o.altitude_px * k * e, 0);
    const far = o.across[1] * k;
    if (o.height_px != null && o.shadow) {
      const T = new THREE.Vector3(x, this._base(o.ping, o.across[1]) + o.height_px * k * e, far),
            E = new THREE.Vector3(x, this._base(o.ping, o.shadow.g1) + 0.05, o.shadow.g1 * k);
      const mat = new THREE.LineDashedMaterial({ color: 0xfff1a8, dashSize: 0.5, gapSize: 0.25 });
      const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints([S, T, E]), mat); l.computeLineDistances(); this.rayGroup.add(l);
      const shadowLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x, this._base(o.ping, o.shadow.g0) + 0.06, o.shadow.g0 * k), E]),
        new THREE.LineBasicMaterial({ color: 0xff5a7a })); this.rayGroup.add(shadowLine);
      this._label(T, `h ≈ ${o.height_px.toFixed(1)} px = ${(o.height_rel * 100).toFixed(0)}% of H`, "tw-lab tw-lab-h");
      this._label(S, `sonar · H ${o.altitude_px.toFixed(1)} px (tracked)`, "tw-lab");
      this._label(E, `shadow ${((o.shadow.g1 - o.shadow.g0)).toFixed(0)} px · ${o.shadow.quality}`, "tw-lab tw-lab-s");
    } else {
      const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints([S, new THREE.Vector3(x, 0.1, o.ground * k)]),
        new THREE.LineDashedMaterial({ color: 0x8ba6c2, dashSize: 0.4, gapSize: 0.3 })); l.computeLineDistances(); this.rayGroup.add(l);
      this._label(new THREE.Vector3(x, 0.4, o.ground * k), "no measurable shadow → height not estimated", "tw-lab");
    }
  }
  _label(pos, text, cls) {
    const d = document.createElement("div"); d.className = cls; d.textContent = text; this.hud.appendChild(d);
    this.labels.push({ pos: pos.clone(), el: d });
  }
  select(id, fromClick) {
    if (this.mode !== "frame" || !this.tw) return;
    const o = this.tw.objects.find(x => x.id === id); if (!o) return;
    this.selId = id; this.labels.forEach(l => l.el.remove()); this.labels = [];
    this.pickables.forEach(m => { m.material.emissiveIntensity = m.userData.pick.id === id ? 1.0 : 0.3; });
    this._drawRay(o);
    const k = this.k, tgt = new THREE.Vector3(o.ping * k, 0, o.ground * k);
    this._flyTo(tgt.clone().add(new THREE.Vector3(-14, 11, -12)), tgt);
    if (fromClick && this.opts.onPick) this.opts.onPick(id);
  }

  /* ------------------------------------------------------------------ survey twin (units: metres) */
  showSurvey(tw) {
    this.clear(); this.mode = "survey"; this.tw = tw;
    const xz = p => new THREE.Vector3(p[0], 0, -p[1]);                 // east → +x, north → −z
    this.xz = xz;
    const all = [];
    tw.ribbons.forEach(r => {
      const c = r.corners, P = [c.p0g0, c.p1g0, c.p0g1, c.p1g1].map(xz); all.push(...P);
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(P.flatMap(v => [v.x, 0, v.z]), 3));
      g.setAttribute("uv", new THREE.Float32BufferAttribute([0, 1, 1, 1, 0, 0, 1, 0], 2));
      g.setIndex([0, 2, 1, 1, 2, 3]); g.computeVertexNormals();
      const t = new THREE.TextureLoader().load(r.texture); t.colorSpace = THREE.SRGBColorSpace;
      const m = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ map: t, side: THREE.DoubleSide, roughness: 0.95 }));
      m.userData.ribbon = r; this.root.add(m);
    });
    tw.objects.forEach(o => all.push(xz(o.xy)));
    const box = new THREE.Box3().setFromPoints(all.length ? all : [new THREE.Vector3()]);
    const size = box.getSize(new THREE.Vector3()), mid = box.getCenter(new THREE.Vector3());
    const R = Math.max(size.x, size.z, 30);
    this.scene.fog.density = 0.45 / R;                                  // fog scaled to the survey extent
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(R * 3, R * 3), new THREE.MeshStandardMaterial({ color: 0x0c131a, roughness: 1 }));
    floor.rotation.x = -Math.PI / 2; floor.position.set(mid.x, -0.03, mid.z); this.root.add(floor);
    const grid = new THREE.GridHelper(R * 3, 60, 0x1b2a38, 0x111a23); grid.position.set(mid.x, -0.02, mid.z); this.root.add(grid);
    this.surveyDyn = new THREE.Group(); this.root.add(this.surveyDyn);
    this._buildSurveyDyn();
    // frame the view on where the data IS (median + central-80% spread), not on the bounding box — a survey
    // can have gaps (e.g. chunks far apart on the track)
    const pts = tw.objects.length ? tw.objects.map(o => xz(o.xy)) : all;
    const med = a => { const b = [...a].sort((x, y) => x - y); return b[Math.floor(b.length / 2)] || 0; };
    const c = new THREE.Vector3(med(pts.map(p => p.x)), 0, med(pts.map(p => p.z)));
    const near = pts.map(p => p.distanceTo(c)).filter(d => d <= 150);          // the main cluster
    const r = Math.max(30, Math.min(130, (Math.max(...near, 20)) * 0.75));
    this.home = { pos: new THREE.Vector3(c.x - r * 0.35, r * 0.62, c.z + r * 0.7), target: c };
    this.goHome();
  }
  _buildSurveyDyn() {
    const tw = this.tw, e = this.exag, xz = this.xz, g = this.surveyDyn;
    g.clear(); this.pickables = [];
    const alt = tw.altitude_m * e;
    // the boat track on the surface (altitude above the seabed)
    const trk = tw.track.map(p => xz(p).setY(alt));
    if (trk.length > 1) g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(trk), new THREE.LineBasicMaterial({ color: CYAN })));
    // hazards: error ring on the seabed + pin with the shadow-derived height
    this.objMeshes = [];
    tw.objects.forEach(o => {
      const p = xz(o.xy), col = VCOL[o.verdict] || 0x8ba6c2;
      const h = o.height_m != null ? Math.max(0.12, o.height_m * e) : 0.1;
      const pin = new THREE.Mesh(new THREE.CylinderGeometry(0.7, 0.7, h, 18),     // the object: shadow-derived height
        new THREE.MeshStandardMaterial({ color: col, emissive: col, emissiveIntensity: 0.5, transparent: true, opacity: o.height_m != null ? 0.95 : 0.6 }));
      pin.position.set(p.x, h / 2, p.z); pin.userData.pick = o; g.add(pin); this.pickables.push(pin);
      const by = Math.max(2.5, alt * 0.75);                               // a beacon so finds are visible at survey scale
      const stem = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(p.x, h, p.z), new THREE.Vector3(p.x, by, p.z)]),
        new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.55 }));
      const head = new THREE.Mesh(new THREE.SphereGeometry(o.verdict === "confirmed" ? 1.1 : 0.8, 16, 12),
        new THREE.MeshStandardMaterial({ color: col, emissive: col, emissiveIntensity: 0.8 }));
      head.position.set(p.x, by, p.z); head.userData.pick = o; g.add(stem, head); this.pickables.push(head);
      const ring = new THREE.Mesh(new THREE.RingGeometry(Math.max(0.3, (o.err_m || 3) - 0.08), o.err_m || 3, 40),
        new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.35, side: THREE.DoubleSide, depthWrite: false }));
      ring.rotation.x = -Math.PI / 2; ring.position.set(p.x, 0.04, p.z); g.add(ring);
      this.objMeshes.push({ o, meshes: [pin, ring, stem, head], along: this._along(p) });
    });
    const line = (pts, color, y, dashed) => {
      if (pts.length < 2) return;
      const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts.map(p => xz(p).setY(y))),
        dashed ? new THREE.LineDashedMaterial({ color, dashSize: 1.2, gapSize: 0.8 }) : new THREE.LineBasicMaterial({ color }));
      if (dashed) l.computeLineDistances(); g.add(l);
    };
    line(tw.routes.recovery, CYAN, 0.25, false);
    line(tw.routes.inspection, 0xd9a441, 0.2, true);
    (tw.resurvey || []).forEach(r => {                                     // planned passes, at boat height
      line([r.a, r.b], VIOLET, alt, true);
      const a = xz(r.a).setY(alt), b = xz(r.b).setY(alt), dir = b.clone().sub(a).normalize();
      const cone = new THREE.Mesh(new THREE.ConeGeometry(0.6, 1.6, 12), new THREE.MeshBasicMaterial({ color: VIOLET }));
      cone.position.copy(b); cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir); g.add(cone);
    });
    // the boat + two sonar fans (port / starboard) for the replay
    this.boat = new THREE.Group();
    const hull = new THREE.Mesh(new THREE.ConeGeometry(0.9, 3.2, 10), new THREE.MeshStandardMaterial({ color: 0xe7eef4, emissive: 0x1fb6d5, emissiveIntensity: 0.25 }));
    hull.rotation.x = Math.PI / 2; this.boat.add(hull);
    const swath = 640 * (tw.m_per_px || 0.05);
    const fan = side => {
      const gg = new THREE.BufferGeometry();
      gg.setAttribute("position", new THREE.Float32BufferAttribute([0, 0, 0, side * swath, -alt, -1.2, side * swath, -alt, 1.2], 3));
      return new THREE.Mesh(gg, new THREE.MeshBasicMaterial({ color: CYAN, transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false }));
    };
    this.boat.add(fan(1), fan(-1)); this.boat.position.copy(trk[0] || new THREE.Vector3(0, alt, 0));
    this.boat.visible = false; g.add(this.boat); this.trackPts = trk; this.boatT = 0;
  }
  _along(p) {
    const t = this.tw.track; if (!t || t.length < 2) return 0;
    const a = this.xz(t[0]), b = this.xz(t[t.length - 1]), d = b.clone().sub(a); const L = d.length() || 1;
    return p.clone().sub(a).dot(d.normalize()) / L;
  }
  replay(on) {
    if (this.mode !== "survey" || !this.boat || this.trackPts.length < 2) return false;
    this.play = on ?? !this.play; this.boat.visible = true;
    if (this.play && this.boatT >= 1) this.boatT = 0;
    if (this.play) this.objMeshes.forEach(m => m.meshes.forEach(x => x.visible = m.along <= this.boatT));
    return this.play;
  }

  /* ------------------------------------------------------------------ controls */
  goHome() { if (this.home) this._flyTo(this.home.pos, this.home.target, 0.9); }
  _flyTo(pos, target, dur = 0.8) { this.fly = { p0: this.camera.position.clone(), t0: this.controls.target.clone(), p1: pos, t1: target, t: 0, dur }; }
  setCameraMode(m) {
    this.cameraMode = m;
    const c = this.controls;
    c.enableRotate = m !== "top";
    if (m === "top" && this.home) {
      const t = this.controls.target.clone(), d = this.camera.position.distanceTo(t);
      this._flyTo(new THREE.Vector3(t.x, Math.max(20, d), t.z + 0.001), t);
    }
  }
  setWire(on) { this.wire = on; this.root.traverse(o => { if (o.isMesh && o.material && o.material.map) o.material.wireframe = on; }); }
  setRelief(v) {
    this.relief = v;
    if (this.seabedMat) { this.seabedMat.displacementScale = v; this.seabedMat.needsUpdate = true; }
    if (this.mode === "frame" && this.tw) this._buildObjects();
  }
  setExag(v) {
    this.exag = v;
    if (this.mode === "frame") { this._buildAltitude(); this._buildObjects(); }
    else if (this.mode === "survey") { const t = this.boatT, p = this.play; this._buildSurveyDyn(); this.boatT = t; this.play = p; if (p || t > 0) this.boat.visible = true; }
  }
  snapshot() { this.renderer.render(this.scene, this.camera); return this.renderer.domElement.toDataURL("image/png"); }

  _tick() {
    if (!this._visible()) return;
    const dt = Math.min(0.05, this.clock.getDelta());
    if (this.fly) {
      const f = this.fly; f.t += dt / f.dur; const s = f.t >= 1 ? 1 : 1 - Math.pow(1 - f.t, 3);
      this.camera.position.lerpVectors(f.p0, f.p1, s); this.controls.target.lerpVectors(f.t0, f.t1, s);
      if (f.t >= 1) this.fly = null;
    }
    if (this.cameraMode === "fly") {                                       // WASD / QE glide
      const v = new THREE.Vector3(), fwd = new THREE.Vector3(); this.camera.getWorldDirection(fwd);
      const right = new THREE.Vector3().crossVectors(fwd, this.camera.up).normalize();
      if (this.keys.KeyW) v.add(fwd); if (this.keys.KeyS) v.sub(fwd);
      if (this.keys.KeyD) v.add(right); if (this.keys.KeyA) v.sub(right);
      if (this.keys.KeyQ) v.y += 1; if (this.keys.KeyE) v.y -= 1;
      if (v.lengthSq()) { const sp = (this.mode === "survey" ? 18 : 12) * dt; v.normalize().multiplyScalar(sp); this.camera.position.add(v); this.controls.target.add(v); }
    }
    if (this.play && this.trackPts && this.trackPts.length > 1) {
      this.boatT = Math.min(1, this.boatT + dt / 14);
      const n = this.trackPts.length - 1, f = this.boatT * n, i = Math.min(n - 1, Math.floor(f));
      const a = this.trackPts[i], b = this.trackPts[i + 1];
      this.boat.position.lerpVectors(a, b, f - i);
      this.boat.lookAt(b.x, b.y, b.z);
      this.objMeshes.forEach(m => { if (m.along <= this.boatT) m.meshes.forEach(x => { if (!x.visible) { x.visible = true; x.scale.setScalar(0.01); } }); });
      if (this.boatT >= 1) { this.play = false; if (this.opts.onReplayEnd) this.opts.onReplayEnd(); }
    }
    this.objMeshes && this.objMeshes.forEach(m => m.meshes.forEach(x => { if (x.visible && x.scale.x < 1) x.scale.setScalar(Math.min(1, x.scale.x + dt * 3)); }));
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    const w = this.el.clientWidth, h = this.el.clientHeight;
    this.labels.forEach(l => {
      const p = l.pos.clone().project(this.camera);
      l.el.style.display = p.z < 1 ? "" : "none";
      l.el.style.transform = `translate(${(p.x * 0.5 + 0.5) * w}px, ${(-p.y * 0.5 + 0.5) * h}px)`;
    });
  }
}

window.DepthTwin = { create: (el, opts) => new TwinViewer(el, opts) };
window.dispatchEvent(new Event("depthtwin-ready"));
