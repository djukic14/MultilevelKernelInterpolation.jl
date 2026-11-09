/* Browser illustrations; docs/check-explorers.jl checks these against MultilevelKernelInterpolation. */
(function (root) {
    "use strict";
    const linspace = (a, b, n) => Array.from({length: n}, (_, i) => a + (b - a) * i / (n - 1));

    function basis(family, p, count, buffer = 0, nodes = "equidistant") {
        if (![p, count, buffer].every(Number.isSafeInteger) || p < 1 || count <= p || buffer < 0)
            throw new Error("Use integer degree >= 1, basis count > degree, and buffer >= 0.");
        if (family === "lagrange") {
            const xs = linspace(0, 1, p + 1).map((x, i) => {
                if (nodes === "chebyshev1") return (1 - Math.cos((2 * i + 1) * Math.PI / (2 * (p + 1)))) / 2;
                if (nodes === "chebyshev2") return (1 - Math.cos(i * Math.PI / p)) / 2;
                return x;
            });
            const logs = xs.map((x, i) => -xs.reduce((s, y, j) => s + (i === j ? 0 : Math.log(Math.abs(x - y))), 0));
            const scale = Math.max(...logs);
            const weights = logs.map((x, i) => (i % 2 ? -1 : 1) * Math.exp(x - scale));
            return {family, p, count: p + 1, buffer: 0, nodes, n: p + 1, points: xs, lo: 0, hi: 1, eval(x) {
                const hit = xs.findIndex(t => Math.abs(t - x) < 4 * Number.EPSILON);
                if (hit >= 0) return xs.map((_, i) => Number(i === hit));
                const terms = xs.map((t, i) => weights[i] / (x - t));
                const sum = terms.reduce((a, b) => a + b, 0);
                return terms.map(t => t / sum);
            }};
        }
        const n = count + 2 * buffer, lo = -buffer / (count - p), hi = 1 - lo;
        const knots = [...Array(p).fill(lo), ...linspace(lo, hi, n - p + 1), ...Array(p).fill(hi)];
        const points = Array.from({length: n}, (_, i) => i === 0 ? lo : i === n - 1 ? hi :
            knots.slice(i + 1, i + p + 1).reduce((a, b) => a + b, 0) / p);
        return {family, p, count, buffer, nodes, n, points, knots, lo, hi, eval(x) {
            if (x === hi) return Array.from({length: n}, (_, i) => Number(i === n - 1));
            let values = knots.slice(0, -1).map((t, i) => Number(t <= x && x < knots[i + 1]));
            for (let d = 1; d <= p; d++) {
                values = values.slice(0, -1).map((_, i) => {
                    const a = knots[i + d] - knots[i], b = knots[i + d + 1] - knots[i + 1];
                    return (a ? (x - knots[i]) * values[i] / a : 0) +
                        (b ? (knots[i + d + 1] - x) * values[i + 1] / b : 0);
                });
            }
            return values;
        }};
    }

    // Partial-pivoting solve with all parent functions as right-hand sides.
    function solve(a, b) {
        const n = a.length, m = b[0].length, rows = a.map((r, i) => [...r, ...b[i]]);
        for (let k = 0; k < n; k++) {
            let pivot = k;
            for (let i = k + 1; i < n; i++) if (Math.abs(rows[i][k]) > Math.abs(rows[pivot][k])) pivot = i;
            if (Math.abs(rows[pivot][k]) < 1e-14) throw new Error("The collocation solve is numerically singular. Try a lower degree.");
            [rows[k], rows[pivot]] = [rows[pivot], rows[k]];
            for (let i = k + 1; i < n; i++) {
                const f = rows[i][k] / rows[k][k];
                for (let j = k; j < n + m; j++) rows[i][j] -= f * rows[k][j];
            }
        }
        const x = Array.from({length: n}, () => Array(m).fill(0));
        for (let i = n - 1; i >= 0; i--) for (let j = 0; j < m; j++) {
            let v = rows[i][n + j];
            for (let k = i + 1; k < n; k++) v -= rows[i][k] * x[k][j];
            x[i][j] = v / rows[i][i];
        }
        return Array.from({length: m}, (_, j) => x.map(r => r[j]));
    }

    function transfer(child, parent, side) {
        const rhs = child.points.map(t => parent.eval((t + side) / 2));
        if (child.family === "lagrange") return Array.from({length: parent.n}, (_, i) => rhs.map(r => r[i]));
        return solve(child.points.map(t => child.eval(t)), rhs);
    }
    // Match increase/noincrease with an ExplicitBuffer policy.
    function parentBasis(child, mode) {
        if (!["increase", "noincrease"].includes(mode)) throw new Error("Choose increase or noincrease.");
        const growing = mode === "increase";
        const p = growing && child.family === "lagrange" ? 2 * child.p : child.p;
        const count = growing && child.family === "bspline" ? 2 * child.count - child.p : child.count;
        return basis(child.family, p, child.family === "lagrange" ? p + 1 : count, child.buffer, child.nodes);
    }
    const api = {basis, transfer, parentBasis};
    if (typeof module !== "undefined") module.exports = api;
    root.MKIExplorer = api;
    if (typeof document === "undefined") return;

    let plotlyReady;
    function loadPlotly() {
        if (root.Plotly) return Promise.resolve(root.Plotly);
        if (!plotlyReady) plotlyReady = new Promise((resolve, reject) => {
            // Documenter uses AMD; loading Plotly as a bare script leaves no global.
            if (root.requirejs && root.define && root.define.amd) {
                root.requirejs.config({paths: {"mki-plotly": "https://cdn.plot.ly/plotly-2.33.0.min"}});
                root.requirejs(["mki-plotly"], resolve, () => reject(new Error("Could not load Plotly. Check your internet connection and reload.")));
                return;
            }
            const script = document.createElement("script");
            script.src = "https://cdn.plot.ly/plotly-2.33.0.min.js";
            script.onload = () => resolve(root.Plotly);
            script.onerror = () => reject(new Error("Could not load Plotly. Check your internet connection and reload."));
            document.head.append(script);
        });
        return plotlyReady;
    }
    const colors = ["#4063d8", "#389826", "#9558b2", "#cb3c33"];
    const trace = (x, y, name, i, dash = "solid") => ({x, y, name, type: "scatter", mode: "lines",
        line: {color: colors[i % colors.length], dash, width: 2}});

    function mount(host) {
        if (host.dataset.ready) return;
        host.dataset.ready = "true";
        const isTransfer = host.dataset.kind === "transfer";
        const fields = {}, form = document.createElement("form");
        form.className = "explorer-controls";
        function field(key, label, value, options) {
            const wrap = document.createElement("label"), input = document.createElement(options ? "select" : "input");
            wrap.textContent = label;
            input.setAttribute("aria-label", label);
            if (options) for (const [v, text] of options) input.add(new Option(text, v));
            else { input.type = "number"; input.step = "1"; input.min = key.includes("buffer") ? "0" : "1"; }
            input.value = value; fields[key] = input; wrap.append(input); form.append(wrap);
        }
        field("family", "Basis", host.dataset.family || "bspline", [["bspline", "B-spline"], ["lagrange", "Lagrange"]]);
        if (isTransfer) { fields.family.disabled = true; fields.family.parentElement.hidden = true; }
        field("nodes", "Lagrange nodes", "equidistant", [["equidistant", "Equidistant"], ["chebyshev1", "Chebyshev I"], ["chebyshev2", "Chebyshev II"]]);
        const lagrange = host.dataset.family === "lagrange";
        field("degree", isTransfer ? "Child degree" : "Degree", lagrange ? 4 : 3);
        field("count", isTransfer ? "Child basis count (before buffer)" : "Basis count (before buffer)", isTransfer ? 6 : 9);
        field("buffer", isTransfer ? "Child buffer per side" : "Buffer per side", 0);
        if (isTransfer) {
            field("parentMode", "Build parent", "noincrease", [["noincrease", "noincrease(child)"], ["increase", "increase(child)"]]);
            field("selected", "Parent function", 3);
        }
        const submit = document.createElement("button"); submit.textContent = "Update"; submit.type = "submit"; form.append(submit);
        const status = document.createElement("p"); status.setAttribute("role", "status"); status.setAttribute("aria-live", "polite");
        const parentSummary = document.createElement("p");
        parentSummary.setAttribute("aria-live", "polite");
        host.append(form);
        if (isTransfer) host.append(parentSummary);
        host.append(status);
        const plots = Array.from({length: isTransfer ? 4 : 1}, () => {
            const div = document.createElement("div"); div.className = "explorer-plot"; host.append(div); return div;
        });
        function toggle() {
            const spline = fields.family.value === "bspline";
            for (const k of ["count", "buffer"]) if (fields[k]) {
                fields[k].disabled = !spline;
                fields[k].parentElement.hidden = !spline;
            }
            fields.nodes.disabled = spline;
            fields.nodes.parentElement.hidden = spline;
        }
        fields.family.addEventListener("change", toggle); toggle();
        async function draw(event) {
            if (event) event.preventDefault();
            submit.disabled = true;
            try {
                const get = key => {
                    const value = Number(fields[key].value);
                    if (fields[key].value.trim() === "" || !Number.isSafeInteger(value)) throw new Error("Enter whole numbers in every active numeric field.");
                    return value;
                };
                const family = fields.family.value, spline = family === "bspline";
                const p = get("degree"), count = spline ? get("count") : p + 1, buffer = spline ? get("buffer") : 0;
                const growing = isTransfer && fields.parentMode.value === "increase";
                const pp = growing && !spline ? 2 * p : p;
                const pc = growing && spline ? 2 * count - p : count;
                const pb = buffer;
                const size = Math.max(spline ? count + 2 * buffer : p + 1, isTransfer ? (spline ? pc + 2 * pb : pp + 1) : 0);
                // No parameter presets or hard order cap; ask before expensive browser work.
                if (size > 100 && !root.confirm("This choice may be slow and numerically unstable. Continue?")) { status.textContent = "Update cancelled."; return; }
                status.textContent = "Computing...";
                await new Promise(resolve => setTimeout(resolve, 30));
                const child = basis(family, p, count, buffer, fields.nodes.value);
                const Plotly = await loadPlotly();
                const dark = document.documentElement.classList.contains("theme--documenter-dark") ||
                    (document.documentElement.dataset.theme === "dark");
                const foreground = dark ? "#ededed" : "#252525";
                const render = (index, data, title, shapes = []) => Plotly.react(plots[index], data, {
                    title: {text: title, font: {size: 15}}, height: 320, autosize: true,
                    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
                    font: {color: foreground}, margin: {t: 45, l: 55, r: 15, b: 45},
                    xaxis: {title: isTransfer ? "x (parent coordinates)" : "x", zeroline: false}, yaxis: {zeroline: false},
                    showlegend: index === 2, legend: {orientation: "h", y: -0.25}, shapes,
                }, {responsive: true, displaylogo: false});
                const shade = (a, b) => ({type: "rect", x0: a, x1: b, y0: 0, y1: 1,
                    yref: "paper", fillcolor: "rgba(149,88,178,0.10)", line: {width: 0}, layer: "below"});
                if (!isTransfer) {
                    const xs = linspace(child.lo, child.hi, Math.max(401, child.n * 12 + 1));
                    const ys = xs.map(x => child.eval(x));
                    await render(0, Array.from({length: child.n}, (_, i) => trace(xs, ys.map(r => r[i]), `Basis ${i + 1}`, i)),
                        `${spline ? "B-spline" : "Lagrange"} basis`, spline && buffer ? [shade(child.lo, 0), shade(1, child.hi)] : []);
                    status.textContent = `${child.n} basis functions; degree ${p}.`;
                } else {
                    const parent = parentBasis(child, fields.parentMode.value);
                    parentSummary.textContent = spline ?
                        `Parent: degree ${parent.p}, ${parent.count} basis functions before buffer, ${parent.buffer} buffer points per side (${parent.n} functions total).` :
                        `Parent: degree ${parent.p}, ${parent.n} basis functions.`;
                    const selected = get("selected") - 1;
                    if (selected < 0 || selected >= parent.n) throw new Error(`Choose a parent function from 1 to ${parent.n}.`);
                    const px = linspace(parent.lo, parent.hi, Math.max(401, parent.n * 12 + 1));
                    const py = px.map(x => parent.eval(x));
                    const u = linspace(child.lo, child.hi, Math.max(401, child.n * 12 + 1));
                    const cv = u.map(x => child.eval(x));
                    await render(0, Array.from({length: parent.n}, (_, i) => trace(px, py.map(r => r[i]), `Parent ${i + 1}`, i)), "Parent basis", pb ? [shade(parent.lo, 0), shade(1, parent.hi)] : []);
                    const children = [], comparison = [], errors = []; let maxError = 0;
                    for (const side of [0, 1]) {
                        const x = u.map(t => (t + side) / 2), c = transfer(child, parent, side)[selected];
                        const truth = x.map(t => parent.eval(t)[selected]);
                        const approx = cv.map(r => r.reduce((sum, v, j) => sum + v * c[j], 0));
                        const residual = approx.map((v, i) => v - truth[i]);
                        for (const v of residual) maxError = Math.max(maxError, Math.abs(v));
                        for (let i = 0; i < child.n; i++) children.push(trace(x, cv.map(r => r[i]), `Child ${side + 1}, basis ${i + 1}`, i, side ? "dash" : "solid"));
                        comparison.push(trace(x, truth, `Parent on ${side ? "right" : "left"}`, 0), trace(x, approx, `From ${side ? "right" : "left"} child`, side + 1, "dash"));
                        errors.push(trace(x, residual, `${side ? "Right" : "Left"} residual`, side));
                    }
                    if (![...comparison, ...errors].every(t => t.y.every(Number.isFinite))) throw new Error("Numerical overflow. Try a lower degree or fewer functions.");
                    await render(1, children, "Left and right child bases", buffer ? [shade(child.lo / 2, 0), shade(1, (child.hi + 1) / 2)] : []);
                    await render(2, comparison, `Parent function ${selected + 1} and reconstruction`);
                    await render(3, errors, "Reconstruction minus parent");
                    status.textContent = `Maximum sampled absolute residual: ${maxError.toExponential(3)}. ` +
                        (spline ? "Browser collocation illustration; MultilevelKernelInterpolation constructs compatible transfers by knot insertion." : "Transfer coefficients are parent values at the child nodes.");
                }
            } catch (err) { status.textContent = err.message + " Plots retain the last completed configuration."; }
            finally { submit.disabled = false; }
        }
        form.addEventListener("submit", draw);
        draw();
    }
    const init = () => document.querySelectorAll(".basis-explorer").forEach(mount);
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})(typeof globalThis === "undefined" ? this : globalThis);
