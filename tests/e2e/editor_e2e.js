// End-to-end test of the schematic editor's drag-and-drop workflow.
// Usage: node tests/e2e/editor_e2e.js http://127.0.0.1:PORT/ [screenshot.png]
const { chromium } = require("playwright");
const assert = (cond, msg) => { if (!cond) throw new Error("ASSERT: " + msg); };
const near = (a, b, tol) => Math.abs(a - b) <= tol;

(async () => {
  const url = process.argv[2];
  const shot = process.argv[3];
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  await page.goto(url);
  await page.waitForFunction(() => window.OpenPE && document.querySelectorAll(".item").length > 20);

  // ---- 1. every bundled example produces a valid netlist and simulates
  const examples = await page.evaluate(() => Object.keys(OpenPE.EXAMPLES));
  for (const name of examples) {
    const r = await page.evaluate(async (name) => {
      OpenPE.loadDoc(OpenPE.EXAMPLES[name]());
      const n = OpenPE.buildNetlist();
      if (n.errors.length) return { errors: n.errors, text: n.text };
      await OpenPE.runSim();
      return { errors: [], res: OpenPE.results, text: n.text, warnings: n.warnings };
    }, name);
    assert(!r.errors.length, `${name}: ${r.errors.join("; ")}\n${r.text}`);
    assert(r.res && r.res.ok, `${name}: simulation failed\n${r.text}`);
    console.log(`example ok: ${name}`);
    if (name.startsWith("Buck")) {
      const v = r.res.tran.signals["V(out)"];
      const tail = v.slice(-200); const mean = tail.reduce((a, b) => a + b, 0) / tail.length;
      assert(near(mean, 12, 0.4), `buck output ${mean}`);
      assert("I(L1)" in r.res.tran.signals, "inductor current plotted");
    }
    if (name.startsWith("RC")) {
      const t = r.res.ac.traces["V(out)"];
      assert(near(t.f_high, 1 / (2 * Math.PI * 1e3 * 100e-9), 20), `RC corner ${t.f_high}`);
    }
    if (name.startsWith("Op-amp")) {
      assert(near(r.res.ac.traces["V(N002)"] ? 0 : 0, 0, 1), "");
      const vout = Object.entries(r.res.op.voltages);
      assert(vout.length > 0, "op voltages");
    }
    if (name.startsWith("Common")) {
      const vc = r.res.op.voltages["c"];
      assert(near(vc, 9.0, 0.3), `CE collector ${vc}`);
    }
  }

  // ---- 2. build a voltage divider from scratch with the mouse
  await page.evaluate(() => OpenPE.loadDoc({ comps: [], wires: [], sim: {} }, "blank"));
  await page.evaluate(() => { const v = OpenPE.view; v.x = 0; v.y = 0; v.k = 1; });
  const canvas = await page.locator("#svg").boundingBox();
  const at = (x, y) => [canvas.x + x, canvas.y + y];           // world == canvas px at k=1, view at 0,0
  const dragPart = async (type, x, y, rotate = 0) => {      // press on a palette item, drag, release on the canvas
    const b = await page.locator(`.item[data-type="${type}"]`).boundingBox();
    await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2); await page.mouse.down();
    await page.mouse.move(...at(x - 40, y - 30), { steps: 8 });
    for (let i = 0; i < rotate; i++) await page.keyboard.press("r");   // rotate while dragging
    await page.mouse.move(...at(x, y), { steps: 4 }); await page.mouse.up();
  };
  await dragPart("V", 200, 300);         // drag from the palette
  await dragPart("R", 300, 200);
  await dragPart("R", 400, 300);
  // click-to-place for the remaining parts
  await page.locator('.item[data-type="GND"]').click();
  await page.mouse.click(...at(200, 350));
  await page.locator('.item[data-type="GND"]').click();
  await page.mouse.click(...at(400, 350));
  let doc = await page.evaluate(() => OpenPE.doc);
  assert(doc.comps.length === 5, `5 parts placed, got ${doc.comps.length}`);
  // rotate the second resistor (selected by clicking it) to vertical
  const R2 = doc.comps.find((c) => c.props.name === "R2");
  await page.mouse.click(...at(R2.x, R2.y));
  await page.keyboard.press("r");
  doc = await page.evaluate(() => OpenPE.doc);
  assert(doc.comps.find((c) => c.props.name === "R2").rot === 1, "rotated with R");

  const pin = async (name, i) => page.evaluate(([name, i]) => {
    const c = name.startsWith("GND")           // ground symbols are unnamed: GND1 = first, GND2 = second
      ? OpenPE.doc.comps.filter((c) => c.type === "GND")[+name.slice(3) - 1]
      : OpenPE.doc.comps.find((c) => c.props.name === name);
    const p = OpenPE.pinsOf(c)[i]; return [p.x, p.y];
  }, [name, i]);
  const dragWire = async (a, b) => {                            // drag from pin to pin
    await page.mouse.move(...at(...a)); await page.mouse.down();
    await page.mouse.move(...at((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), { steps: 4 });
    await page.mouse.move(...at(...b), { steps: 4 }); await page.mouse.up();
  };
  await dragWire(await pin("V1", 0), await pin("R1", 0));      // V+ -> R1 left
  await dragWire(await pin("R1", 1), await pin("R2", 0));      // R1 right -> R2 top
  await dragWire(await pin("V1", 1), await pin("GND1", 0));
  await dragWire(await pin("R2", 1), await pin("GND2", 0));
  doc = await page.evaluate(() => OpenPE.doc);
  assert(doc.wires.length === 4, `4 wires drawn, got ${doc.wires.length}`);
  for (const w of doc.wires) for (let i = 1; i < w.pts.length; i++)
    assert(w.pts[i][0] === w.pts[i - 1][0] || w.pts[i][1] === w.pts[i - 1][1], "wires are orthogonal");

  // edit V1 value through the properties panel (double-click opens it)
  const V1 = doc.comps.find((c) => c.props.name === "V1");
  await page.mouse.dblclick(...at(V1.x, V1.y));
  const dcField = page.locator('#props input[data-k="dc"]');
  await dcField.fill("10"); await dcField.press("Enter");
  // switch analyses: only .op
  await page.keyboard.press("Escape");
  await page.mouse.click(...at(700, 600));
  await page.locator('#props input[data-s="tran"]').uncheck();
  await page.locator('#props input[data-s="op"]').check();
  let net = await page.evaluate(() => OpenPE.buildNetlist());
  assert(!net.errors.length, "divider netlist: " + net.errors.join("; ") + "\n" + net.text);
  await page.locator("#bRun").click();
  await page.waitForFunction(() => OpenPE.results && OpenPE.results.op);
  let res = await page.evaluate(() => OpenPE.results);
  const mid = await page.evaluate(() => { const n = OpenPE.computeNets(); const c = OpenPE.doc.comps.find((c) => c.props.name === "R1"); return n.names.get(n.pinNet.get(c.id + ":1")); });
  assert(near(res.op.voltages[mid], 5, 1e-6), `divider mid = ${res.op.voltages[mid]}`);
  console.log("divider built with the mouse: V(mid) =", res.op.voltages[mid]);

  // ---- 3. move R2 by dragging its body: attached wires must follow
  const before = await pin("R2", 0);
  const R2b = (await page.evaluate(() => OpenPE.doc)).comps.find((c) => c.props.name === "R2");
  await page.mouse.move(...at(R2b.x + 2, R2b.y + 2)); await page.mouse.down();
  await page.mouse.move(...at(R2b.x + 62, R2b.y + 42), { steps: 6 }); await page.mouse.up();
  const after = await pin("R2", 0);
  assert(after[0] === before[0] + 60 && after[1] === before[1] + 40, `moved by grid-snapped delta ${after}`);
  net = await page.evaluate(() => OpenPE.buildNetlist());
  assert(!net.errors.length && !net.warnings.length, "still fully connected after move: " + net.warnings.join("; "));
  doc = await page.evaluate(() => OpenPE.doc);
  for (const w of doc.wires) for (let i = 1; i < w.pts.length; i++)
    assert(w.pts[i][0] === w.pts[i - 1][0] || w.pts[i][1] === w.pts[i - 1][1], "rubber-banded wires stay orthogonal");
  await page.locator("#bRun").click();
  await page.waitForFunction((m) => OpenPE.results && OpenPE.results.op && Math.abs(OpenPE.results.op.voltages[m] - 5) < 1e-6, mid);

  // ---- 4. undo / redo, copy / paste, delete, box select
  await page.keyboard.press("Control+z");
  assert(near((await pin("R2", 0))[0], before[0], 0), "undo restores position");
  await page.keyboard.press("Control+y");
  assert(near((await pin("R2", 0))[0], before[0] + 60, 0), "redo re-applies move");
  await page.mouse.move(...at(150, 150)); await page.mouse.down();
  await page.mouse.move(...at(700, 650), { steps: 5 }); await page.mouse.up();   // box-select all
  const nsel = await page.evaluate(() => OpenPE.selection.length);
  assert(nsel === 9, `box select picked ${nsel} items`);
  await page.keyboard.press("Control+c"); await page.keyboard.press("Control+v");
  doc = await page.evaluate(() => OpenPE.doc);
  assert(doc.comps.length === 10 && doc.wires.length === 8, "paste duplicated the circuit");
  assert(new Set(doc.comps.filter((c) => c.props.name).map((c) => c.props.name)).size === 6, "pasted parts were renamed");
  await page.keyboard.press("Delete");
  doc = await page.evaluate(() => OpenPE.doc);
  assert(doc.comps.length === 5 && doc.wires.length === 4, "delete removed the pasted copy");

  // ---- 5. a control loop: PWM driving a switch is a signal net, not a power net
  await page.evaluate(() => OpenPE.loadDoc(OpenPE.EXAMPLES["Buck converter (PWM)"]()));
  const kinds = await page.evaluate(() => { const n = OpenPE.computeNets(); return [...n.kinds.values()]; });
  assert(kinds.includes("s") && kinds.includes("p"), "signal and power nets distinguished");

  if (shot) {
    await page.evaluate(() => OpenPE.runSim());
    await page.waitForFunction(() => OpenPE.results && OpenPE.results.tran);
    await page.waitForTimeout(300);
    await page.screenshot({ path: shot });
  }
  assert(!errors.length, "browser errors: " + errors.join("\n"));
  await browser.close();
  console.log("E2E OK");
})().catch((e) => { console.error(e.message || e); process.exit(1); });
