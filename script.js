/* =====================================================================
   CreditAI MLOps — script.js
   1. Motor do modelo: Random Forest exportado do scikit-learn
   2. Réplica da API FastAPI (rotas, Pydantic 422, regras 400, JSON)
   3. Cliente para a API real
   4. Gráficos SVG
   5. Seções da página
   ===================================================================== */
(function (G) {
  "use strict";

  /* ================= 1. MOTOR ================= */
  var Engine = (function () {
    function M() { return G.CREDIT_MODEL; }
    function miss(v) { return v === null || v === undefined || v === "" || (typeof v === "number" && isNaN(v)); }
    function vec(r) {
      var x = [];
      M().numeric.forEach(function (f) {
        var v = miss(r[f.name]) ? f.median : Number(r[f.name]);
        x.push(Math.fround((v - f.mean) / f.scale)); // o scikit-learn compara em float32
      });
      M().categorical.forEach(function (f) {
        var v = miss(r[f.name]) ? f.mode : String(r[f.name]);
        f.categories.forEach(function (c) { x.push(c === v ? 1 : 0); });
      });
      return x;
    }
    /** [P(reprovado), P(aprovado)] = média das 200 árvores, como predict_proba */
    function probas(r) {
      var x = vec(r), trees = M().trees, s0 = 0, s1 = 0;
      for (var i = 0; i < trees.length; i++) {
        var t = trees[i], n = 0;
        while (t.f[n] !== -1) n = x[t.f[n]] <= t.t[n] ? t.l[n] : t.r[n];
        s0 += t.q[n]; s1 += t.p[n];
      }
      return [s0 / trees.length, s1 / trees.length];
    }
    function proba(r) { return probas(r)[1]; }
    function faixa(p) { return p >= 0.75 ? "Baixo Risco" : p >= 0.45 ? "Risco Moderado" : "Alto Risco"; }
    return { proba: proba, probas: probas, faixa: faixa };
  })();

  /* ================= 2. RÉPLICA DA API ================= */
  var MORADIA = ["PROPRIA", "ALUGADA", "FINANCIADA", "OUTROS"];
  var FINALIDADE = ["EDUCACAO", "SAUDE", "NEGOCIOS", "PESSOAL", "QUITACAO_DIVIDAS", "IMOVEL"];
  // mesma ordem e regras de DadosClienteInput (app/schemas/cliente.py)
  var SCHEMA = [
    ["nome_cliente", { type: "str", opt: true, def: "Cliente Padrao", min: 2, max: 80 }],
    ["idade", { type: "int", ge: 18, le: 100 }],
    ["pontuacao_credito", { type: "int", ge: 300, le: 850 }],
    ["renda_mensal", { type: "float", def: 5000.0, gt: 0, le: 1000000 }],
    ["valor_emprestimo", { type: "float", def: 12000.0, ge: 500, le: 500000 }],
    ["tempo_emprego_anos", { type: "float", def: 3.0, ge: 0, le: 60 }],
    ["inadimplencia_previa", { type: "int", def: 0, ge: 0, le: 1 }],
    ["tipo_moradia", { type: "enum", def: "PROPRIA", values: MORADIA }],
    ["finalidade_emprestimo", { type: "enum", def: "PESSOAL", values: FINALIDADE }]
  ];
  var FIELDS = SCHEMA.map(function (s) { return s[0]; });

  function pyNum(v) { return Number.isInteger(v) ? v.toFixed(1) : String(v); }
  function enumMsg(vals) {
    var q = vals.map(function (v) { return "'" + v + "'"; });
    return q.slice(0, -1).join(", ") + " or " + q[q.length - 1];
  }
  function checkField(name, s, raw, errors) {
    function err(type, msg, ctx) { var e = { type: type, loc: ["body", name], msg: msg, input: raw }; if (ctx) e.ctx = ctx; errors.push(e); }
    var v = raw;
    if (s.type === "str") {
      if (raw === null && s.opt) return null;
      if (typeof raw !== "string") { err("string_type", "Input should be a valid string"); return; }
      v = raw.trim();
      if (v.length < s.min) { err("string_too_short", "String should have at least " + s.min + " characters", { min_length: s.min }); return; }
      if (v.length > s.max) { err("string_too_long", "String should have at most " + s.max + " characters", { max_length: s.max }); return; }
      return v;
    }
    if (s.type === "enum") {
      if (s.values.indexOf(raw) === -1) { err("enum", "Input should be " + enumMsg(s.values), { expected: enumMsg(s.values) }); return; }
      return raw;
    }
    if (s.type === "int") {
      if (typeof raw === "number") {
        if (!isFinite(raw)) { err("finite_number", "Input should be a finite number"); return; }
        if (Math.floor(raw) !== raw) { err("int_from_float", "Input should be a valid integer, got a number with a fractional part"); return; }
      } else if (typeof raw === "string") {
        if (!/^\s*[-+]?\d+\s*$/.test(raw)) { err("int_parsing", "Input should be a valid integer, unable to parse string as an integer"); return; }
        v = parseInt(raw, 10);
      } else if (typeof raw === "boolean") { v = raw ? 1 : 0; }
      else { err("int_type", "Input should be a valid integer"); return; }
    } else {
      if (typeof raw === "string") {
        if (raw.trim() === "" || isNaN(Number(raw))) { err("float_parsing", "Input should be a valid number, unable to parse string as a number"); return; }
        v = Number(raw);
      } else if (typeof raw === "boolean") { v = raw ? 1 : 0; }
      else if (typeof raw !== "number") { err("float_type", "Input should be a valid number"); return; }
    }
    if (s.gt !== undefined && !(v > s.gt)) { err("greater_than", "Input should be greater than " + s.gt, { gt: s.gt }); return; }
    if (s.ge !== undefined && !(v >= s.ge)) { err("greater_than_equal", "Input should be greater than or equal to " + s.ge, { ge: s.ge }); return; }
    if (s.le !== undefined && !(v <= s.le)) { err("less_than_equal", "Input should be less than or equal to " + s.le, { le: s.le }); return; }
    return v;
  }
  function validate(obj, errors) {
    if (obj === null || typeof obj !== "object" || Array.isArray(obj)) {
      errors.push({ type: "model_attributes_type", loc: ["body"], msg: "Input should be a valid dictionary or object to extract fields from", input: obj });
      return null;
    }
    var c = {};
    SCHEMA.forEach(function (pair) {
      var name = pair[0], s = pair[1];
      if (!Object.prototype.hasOwnProperty.call(obj, name)) {
        if (s.def === undefined) errors.push({ type: "missing", loc: ["body", name], msg: "Field required", input: obj });
        else c[name] = s.def;
      } else c[name] = checkField(name, s, obj[name], errors);
    });
    Object.keys(obj).forEach(function (k) {
      if (FIELDS.indexOf(k) === -1) errors.push({ type: "extra_forbidden", loc: ["body", k], msg: "Extra inputs are not permitted", input: obj[k] });
    });
    return c;
  }
  /** realizar_predicao() de credito_service.py */
  function predictService(c) {
    var idadeLaboral = Math.max(0, c.idade - 14);
    if (c.tempo_emprego_anos > idadeLaboral)
      return { status: 400, data: { detail: "Inconsistencia cadastral: tempo_emprego_anos (" + pyNum(c.tempo_emprego_anos) + ") excede a idade laboral possivel (" + idadeLaboral + " anos)." } };
    var indice = c.valor_emprestimo / (c.renda_mensal * 12);
    if (indice > 20) return { status: 400, data: { detail: "Valor solicitado excede 20x a renda anual declarada." } };
    var pp = Engine.probas(c), p = pp[1], cls = pp[1] > pp[0] ? 1 : 0; // argmax, como o predict()
    return { status: 200, data: {
      nome_cliente: c.nome_cliente, idade_informada: c.idade, pontuacao_informada: c.pontuacao_credito,
      decisao: cls ? "Aprovado" : "Reprovado", codigo_decisao: cls,
      probabilidade_aprovacao: round4(p), probabilidade_risco_inadimplencia: round4(pp[0]),
      faixa_risco: Engine.faixa(p), indice_comprometimento_anual: round4(indice),
      versao_modelo: G.CREDIT_MODEL.versao_modelo
    } };
  }
  function round4(v) { return Number(v.toFixed(4)); } // arredonda o valor exato, como o round() do Python

  var LocalApi = {
    name: "navegador",
    request: function (method, path, body) {
      var t0 = performance.now(), r = LocalApi.handle(String(method).toUpperCase(), (path || "/").split("?")[0], body);
      r.ms = Math.max(0.01, performance.now() - t0);
      r.ok = r.status < 300; r.error = r.ok ? null : fmtDetail(r.data.detail);
      return Promise.resolve(r);
    },
    handle: function (method, path, body) {
      var routes = { "/health": "GET", "/predict": "POST", "/v1/predict": "POST" };
      if (!routes[path]) return { status: 404, data: { detail: "Not Found" } };
      if (routes[path] !== method) return { status: 405, data: { detail: "Method Not Allowed" } };
      var m = G.CREDIT_MODEL;
      if (path === "/health") return { status: 200, data: { status_servico: "online", modelo_carregado_em_ram: true, algoritmo: m.algoritmo, versao_modelo: m.versao_modelo, metricas_referencia_teste: m.metricas_teste } };
      if (typeof body === "string") {
        try { body = JSON.parse(body); }
        catch (e) { return { status: 422, data: { detail: [{ type: "json_invalid", loc: ["body", 0], msg: "JSON decode error", input: {}, ctx: { error: e.message } }] } }; }
      }
      var errors = [], c = validate(body, errors);
      if (errors.length) return { status: 422, data: { detail: errors } };
      return predictService(c);
    }
  };
  function fmtDetail(d) {
    if (Array.isArray(d)) return d.map(function (e) { var l = (e.loc || []).filter(function (p) { return p !== "body"; }).join("."); return (l ? l + ": " : "") + e.msg; }).join(" | ");
    return d === undefined ? "Erro desconhecido" : String(d);
  }

  /* ================= 3. API REAL ================= */
  function RemoteApi(base) { this.base = String(base || "").replace(/\/+$/, ""); this.name = "API real"; }
  RemoteApi.prototype.request = function (method, path, body) {
    var self = this, t0 = performance.now(), o = { method: method, headers: { Accept: "application/json" } };
    if (body !== undefined && body !== null && method !== "GET") { o.headers["Content-Type"] = "application/json"; o.body = typeof body === "string" ? body : JSON.stringify(body); }
    return fetch(self.base + path, o).then(function (res) {
      return res.text().then(function (t) {
        var d; try { d = JSON.parse(t); } catch (e) { d = { detail: t }; }
        return { ok: res.ok, status: res.status, data: d, ms: performance.now() - t0, error: res.ok ? null : fmtDetail(d && d.detail) };
      });
    }).catch(function () {
      return { ok: false, status: null, data: null, ms: performance.now() - t0, error: "Sem resposta de " + self.base + ". A API está rodando e com CORS liberado?" };
    });
  };

  if (typeof module !== "undefined" && module.exports) { module.exports = { Engine: Engine, LocalApi: LocalApi }; }
  if (typeof document === "undefined") return;

  /* ================= 4. GRÁFICOS SVG ================= */
  var NS = "http://www.w3.org/2000/svg";
  function el(tag, a, parent) { var n = document.createElementNS(NS, tag); for (var k in a) n.setAttribute(k, a[k]); if (parent) parent.appendChild(n); return n; }
  function svg(w, h, label) { return el("svg", { viewBox: "0 0 " + w + " " + h, role: "img", "aria-label": label, "class": "chart" }); }
  function text(p, x, y, s, cls, anchor) { var t = el("text", { x: x, y: y, "class": cls || "ch-lbl", "text-anchor": anchor || "middle" }, p); t.textContent = s; return t; }
  function tip(n, s) { el("title", {}, n).textContent = s; }
  function fitW(box, max, min) { var w = box && box.clientWidth ? box.clientWidth : max; return Math.max(min || 320, Math.min(max, w)); }
  function nice(v) { if (v <= 0) return 1; var s = Math.pow(10, Math.floor(Math.log10(v))), m = v / s; return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * s; }
  function pct(v, d) { return (v * 100).toLocaleString("pt-BR", { minimumFractionDigits: d === undefined ? 1 : d, maximumFractionDigits: d === undefined ? 1 : d }) + "%"; }
  function num(v, d) { return Number(v).toLocaleString("pt-BR", { minimumFractionDigits: d || 0, maximumFractionDigits: d || 0 }); }
  function brl(v) { return "R$ " + num(v, 0); }

  /** Velocímetro semicircular da probabilidade de aprovação */
  function Gauge(p) {
    var s = svg(300, 176, "Probabilidade de aprovação " + pct(p)), r = 116, cx = 150, cy = 150, len = Math.PI * r;
    var d = "M" + (cx - r) + " " + cy + " A" + r + " " + r + " 0 0 1 " + (cx + r) + " " + cy;
    el("path", { d: d, "class": "g-track" }, s);
    var v = el("path", { d: d, "class": "g-val", style: "stroke:" + riskColor(p) + ";stroke-dasharray:" + len + ";stroke-dashoffset:" + (len * (1 - p)) }, s);
    tip(v, "Probabilidade de aprovação: " + pct(p));
    // marcas de 45% e 75% (faixas de risco)
    [0.45, 0.75].forEach(function (m) {
      var a = Math.PI * (1 - m), x1 = cx + (r - 14) * Math.cos(a), y1 = cy - (r - 14) * Math.sin(a), x2 = cx + (r + 14) * Math.cos(a), y2 = cy - (r + 14) * Math.sin(a);
      el("line", { x1: x1, y1: y1, x2: x2, y2: y2, "class": "ch-mark" }, s);
      text(s, cx + (r + 26) * Math.cos(a), cy - (r + 26) * Math.sin(a) + 4, Math.round(m * 100) + "%", "ch-lbl sm");
    });
    text(s, cx, cy - 18, pct(p), "g-num");
    text(s, cx, cy + 4, "chance de aprovação", "ch-lbl");
    return s;
  }
  function riskColor(p) { return p >= 0.75 ? "var(--ok)" : p >= 0.45 ? "var(--warn)" : "var(--bad)"; }
  function riskCls(p) { return p >= 0.75 ? "ch-ok" : p >= 0.45 ? "ch-warn" : "ch-bad"; }

  function frame(w, h, m) { return { w: w, h: h, m: m, iw: w - m.l - m.r, ih: h - m.t - m.b }; }
  function ticksFor(max) { var m = max / Math.pow(10, Math.floor(Math.log10(max))); return m === 2 ? 4 : 5; }
  function yGrid(s, f, max, ticks, fmt) {
    if (ticks === "auto") ticks = ticksFor(max);
    for (var i = 0; i <= ticks; i++) {
      var v = max * i / ticks, y = f.m.t + f.ih - f.ih * i / ticks;
      el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: y, y2: y, "class": "ch-grid" }, s);
      text(s, f.m.l - 8, y + 4, fmt(v), "ch-lbl", "end");
    }
  }

  /** Linha genérica 0..1 no eixo y */
  function LineChart(xs, ys, o) {
    var f = frame(o.w || 640, o.h || 280, { t: 16, r: 18, b: 46, l: 48 }), s = svg(f.w, f.h, o.label || "");
    var x0 = o.x0 !== undefined ? o.x0 : xs[0], x1 = o.x1 !== undefined ? o.x1 : xs[xs.length - 1];
    function X(v) { return f.m.l + (v - x0) / (x1 - x0) * f.iw; }
    function Y(v) { return f.m.t + f.ih - v * f.ih; }
    yGrid(s, f, 1, 4, o.fmtY || function (v) { return Math.round(v * 100) + "%"; });
    (o.xticks || []).forEach(function (t) { text(s, X(t), f.h - 26, o.fmtX ? o.fmtX(t) : t, "ch-lbl"); });
    if (o.xTitle) text(s, f.m.l + f.iw / 2, f.h - 4, o.xTitle, "ch-ttl");
    if (o.diag) el("line", { x1: X(x0), y1: Y(0), x2: X(x1), y2: Y(1), "class": "ch-diag" }, s);
    if (o.hline !== undefined) { el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: Y(o.hline), y2: Y(o.hline), "class": "ch-mark" }, s); text(s, f.m.l + 6, Y(o.hline) - 6, o.hlineLabel || "", "ch-lbl sm", "start"); }
    var pts = xs.map(function (x, i) { return X(x) + " " + Y(ys[i]); });
    if (o.area) el("path", { d: "M" + X(xs[0]) + " " + Y(0) + " L" + pts.join(" L") + " L" + X(xs[xs.length - 1]) + " " + Y(0) + " Z", "class": "ch-area" }, s);
    el("path", { d: "M" + pts.join(" L"), "class": "ch-line" }, s);
    if (o.mark !== undefined) {
      var i = o.markIndex, mx = X(o.mark);
      el("line", { x1: mx, x2: mx, y1: f.m.t, y2: f.m.t + f.ih, "class": "ch-mark" }, s);
      var c = el("circle", { cx: mx, cy: Y(ys[i]), r: 6, "class": "ch-dot hot" }, s);
      tip(c, o.markLabel || "");
      var right = mx + 8 > f.m.l + f.iw - 110;
      text(s, right ? mx - 8 : mx + 8, f.m.t + f.ih - 8, o.markLabel || "", "ch-lbl sm", right ? "end" : "start");
    }
    return s;
  }

  /** Barras horizontais: [{k, v, cls, label}] */
  function HBars(items, o) {
    var rowH = o.rowH || 34, lw = o.labelW || 150, w = o.w || 560, h = items.length * rowH + 10, s = svg(w, h, o.label || "");
    var max = o.max || nice(Math.max.apply(null, items.map(function (d) { return d.v; })) * 1.1), iw = w - lw - 70;
    items.forEach(function (d, i) {
      var y = i * rowH + 6, bw = Math.max(2, iw * d.v / max);
      text(s, lw - 10, y + rowH / 2 + 1, d.k, "ch-lbl", "end");
      el("rect", { x: lw, y: y + 5, width: iw, height: rowH - 14, rx: 5, "class": "ch-muted", style: "opacity:.15" }, s);
      var r = el("rect", { x: lw, y: y + 5, width: bw, height: rowH - 14, rx: 5, "class": d.cls || "ch-brand" }, s);
      tip(r, d.k + ": " + (d.label || d.v));
      text(s, lw + bw + 8, y + rowH / 2 + 1, d.label || d.v, "ch-val", "start");
    });
    return s;
  }

  /* ================= 5. PÁGINA ================= */
  var D = G.CREDIT_DATA;
  function $(s, r) { return (r || document).querySelector(s); }
  function $$(s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); }
  function esc(s) { return String(s).replace(/[&<>"']/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]; }); }
  function jsonHtml(o) {
    return esc(JSON.stringify(o, null, 2)).replace(/(&quot;[^&]*?&quot;)(\s*:)/g, '<span class="k">$1</span>$2').replace(/:\s(&quot;.*?&quot;)/g, ': <span class="s">$1</span>');
  }
  function stCls(s) { return !s ? "s5xx" : s < 300 ? "s200" : s < 500 ? "s4xx" : "s5xx"; }
  function stTxt(s) { return { 200: "200 OK", 400: "400 Bad Request", 404: "404 Not Found", 405: "405 Method Not Allowed", 422: "422 Unprocessable Entity", 500: "500 Internal Server Error" }[s] || (s ? String(s) : "sem resposta"); }
  function fmtMs(ms) { return ms < 1 ? ms.toLocaleString("pt-BR", { maximumFractionDigits: 2 }) + " ms" : num(ms, ms < 100 ? 1 : 0) + " ms"; }
  var LBL = { PROPRIA: "Própria", ALUGADA: "Alugada", FINANCIADA: "Financiada", OUTROS: "Outros", EDUCACAO: "Educação", SAUDE: "Saúde", NEGOCIOS: "Negócios", PESSOAL: "Pessoal", QUITACAO_DIVIDAS: "Quitar dívidas", IMOVEL: "Imóvel" };
  function L(v) { return LBL[v] || v; }
  function curl(method, path, body) {
    var base = state.api === LocalApi ? "http://127.0.0.1:8000" : state.api.base;
    return "curl -X " + method + " " + base + path + (body && method !== "GET" ? " \\\n  -H 'Content-Type: application/json' \\\n  -d '" + (typeof body === "string" ? body : JSON.stringify(body)) + "'" : "");
  }

  var PRESETS = {
    carlos: { nome_cliente: "Carlos Eduardo Santos", idade: 35, pontuacao_credito: 720, renda_mensal: 7500, valor_emprestimo: 18000, tempo_emprego_anos: 5, inadimplencia_previa: 0, tipo_moradia: "PROPRIA", finalidade_emprestimo: "NEGOCIOS" },
    roberto: { nome_cliente: "Roberto Silva", idade: 21, pontuacao_credito: 360, renda_mensal: 1800, valor_emprestimo: 45000, tempo_emprego_anos: 0.5, inadimplencia_previa: 1, tipo_moradia: "ALUGADA", finalidade_emprestimo: "QUITACAO_DIVIDAS" },
    ana: { nome_cliente: "Ana Beatriz Souza", idade: 38, pontuacao_credito: 780, renda_mensal: 12000, valor_emprestimo: 15000, tempo_emprego_anos: 8, inadimplencia_previa: 0, tipo_moradia: "PROPRIA", finalidade_emprestimo: "NEGOCIOS" }
  };
  var state = { api: LocalApi, cliente: Object.assign({}, PRESETS.carlos) };

  /* ---------- hero ---------- */
  function initHero() {
    var t = D.metricas.teste, L50 = D.latency.p50;
    $("#heroKpis").innerHTML = [
      ["good", pct(t.acuracia), "acurácia em 450 clientes inéditos"],
      ["", num(t.roc_auc, 3).replace(".", ","), "ROC-AUC no teste"],
      ["", "3.000", "clientes, 8 variáveis"],
      ["good", "6/6", "testes pytest passando"],
      ["", num(L50, 0) + "<small> ms</small>", "por predição na API, 5 ms otimizada"]
    ].map(function (k) { return '<div class="kpi ' + k[0] + '"><div class="v">' + k[1] + '</div><div class="l">' + k[2] + "</div></div>"; }).join("");

    var people = [PRESETS.carlos, PRESETS.roberto, PRESETS.ana], idx = 0, dots = $("#parDots");
    dots.innerHTML = people.map(function (p, i) { return '<button type="button" aria-label="Parecer de ' + esc(p.nome_cliente) + '"></button>'; }).join("");
    function show(i) {
      idx = i; var c = people[i], r = predictService(c), d = r.data;
      $("#parId").textContent = "nº " + String(1027 + i).padStart(4, "0");
      $("#parRows").innerHTML = [
        ["Solicitante", c.nome_cliente], ["Idade", c.idade + " anos"], ["Renda mensal", brl(c.renda_mensal)], ["Valor pedido", brl(c.valor_emprestimo)],
        ["Finalidade", L(c.finalidade_emprestimo)], ["Moradia", L(c.tipo_moradia)], ["Emprego", num(c.tempo_emprego_anos, c.tempo_emprego_anos % 1 ? 1 : 0) + " anos"], ["Inadimplência", c.inadimplencia_previa ? "Sim" : "Não"]
      ].map(function (r) { return "<div><dt>" + r[0] + "</dt><dd>" + esc(r[1]) + "</dd></div>"; }).join("");
      var mk = $("#parMk"); mk.style.left = ((c.pontuacao_credito - 300) / 550 * 100) + "%"; mk.querySelector("span").textContent = c.pontuacao_credito;
      var st = $("#parStamp"); st.className = "stamp " + (d.codigo_decisao ? "aprovado" : "reprovado");
      st.innerHTML = esc(d.decisao) + "<small>" + esc(d.faixa_risco) + "</small>";
      $("#parProb").textContent = pct(d.probabilidade_aprovacao);
      void st.offsetWidth; st.classList.add("pop");
      $("#parReq").innerHTML = '<span class="dim">POST</span> /predict <span class="dim">→</span> <span class="ok">200 OK</span> <span class="dim">· decisao:</span> "' + esc(d.decisao) + '" <span class="dim">· probabilidade_aprovacao:</span> ' + d.probabilidade_aprovacao;
      $$("button", dots).forEach(function (b, j) { b.setAttribute("aria-current", j === i ? "true" : "false"); });
    }
    $$("button", dots).forEach(function (b, i) { b.addEventListener("click", function () { stop(); show(i); }); });
    show(0);
    var timer = null;
    function stop() { if (timer) { clearInterval(timer); timer = null; } }
    if (!matchMedia("(prefers-reduced-motion: reduce)").matches) timer = setInterval(function () { show((idx + 1) % people.length); }, 4200);
    $("#parecer").addEventListener("mouseenter", stop);
  }

  /* ---------- fluxo da API ---------- */
  var FLOWS = {
    aprovado: { body: PRESETS.carlos, path: "/predict" },
    reprovado: { body: PRESETS.roberto, path: "/v1/predict" },
    invalido: { body: { idade: 15, pontuacao_credito: 990 }, path: "/predict" },
    regra: { body: { idade: 20, pontuacao_credito: 700, renda_mensal: 4000, valor_emprestimo: 10000, tempo_emprego_anos: 15 }, path: "/predict" }
  };
  var flowTimers = [];
  function runFlow(key, instant) {
    flowTimers.forEach(clearTimeout); flowTimers = [];
    var f = FLOWS[key], r = LocalApi.handle("POST", f.path, f.body);
    var stopAt = r.status === 422 ? 2 : r.status === 400 ? 3 : 5;
    var nodes = $$("#pipe .node"), packet = $("#packet"), pipe = $("#pipe");
    nodes.forEach(function (n) { n.className = "node"; });
    packet.className = "packet";
    $("#flowMethod").textContent = "POST " + f.path;
    $("#flowReq").innerHTML = jsonHtml(f.body);
    $("#flowRes").innerHTML = "";
    var st = $("#flowStatus"); st.textContent = "…"; st.className = "pill";
    var narr = {
      0: "O cliente envia o JSON para a API.",
      1: "O roteador do FastAPI recebe <code>POST " + f.path + "</code> e entrega o corpo ao schema.",
      2: r.status === 422 ? "<b>Barrado pelo Pydantic.</b> Idade 15 viola <code>ge=18</code> e score 990 viola <code>le=850</code>. O modelo nem é chamado." : "O Pydantic confere tipos e limites de cada campo. Tudo certo.",
      3: r.status === 400 ? "<b>Barrado pela regra de negócio.</b> 15 anos de emprego aos 20 anos de idade não fecha a conta. Os tipos estavam certos, então é 400 e não 422." : "O serviço confere as regras: tempo de emprego compatível com a idade e empréstimo até 20× a renda anual.",
      4: "O Random Forest, já na memória desde o startup, calcula a probabilidade com as 200 árvores.",
      5: "A resposta volta tipada pelo <code>PredicaoCreditoOutput</code>: <b>" + (r.data.decisao || "") + "</b>, " + (r.data.faixa_risco || "") + "."
    };
    var reduce = instant || matchMedia("(prefers-reduced-motion: reduce)").matches, step = reduce ? 0 : 650;
    function place(i) {
      var n = nodes[i], pr = pipe.getBoundingClientRect(), nr = n.getBoundingClientRect();
      packet.style.left = (nr.left - pr.left + nr.width / 2 - 9) + "px";
    }
    for (var i = 0; i <= stopAt; i++) (function (i) {
      flowTimers.push(setTimeout(function () {
        packet.classList.add("on"); place(i);
        nodes.forEach(function (n, j) { if (j < i) n.className = "node pass"; });
        var last = i === stopAt;
        nodes[i].className = "node " + (last && r.status !== 200 ? "stop-" + r.status : last ? "pass" : "active");
        if (last && r.status !== 200) packet.classList.add("err");
        $("#flowNarr").innerHTML = narr[i];
        if (last) {
          $("#flowRes").innerHTML = jsonHtml(r.data);
          st.textContent = stTxt(r.status); st.className = "pill " + stCls(r.status);
          if (r.status === 200) tickLifespan();
        }
      }, i * step));
    })(i);
  }
  var lsCount = 0;
  function tickLifespan() {
    var box = $("#lsReqs"); if (box.children.length >= 24) box.removeChild(box.firstChild);
    var b = document.createElement("i"); b.className = "hit"; b.title = "Requisição " + (++lsCount); box.appendChild(b);
  }
  function initFlow() {
    var box = $("#lsReqs"); for (var i = 0; i < 8; i++) { var b = document.createElement("i"); b.className = "hit"; box.appendChild(b); }
    $$("[data-flow]").forEach(function (b) {
      b.addEventListener("click", function () {
        $$("[data-flow]").forEach(function (o) { o.setAttribute("aria-pressed", o === b ? "true" : "false"); });
        runFlow(b.dataset.flow);
      });
    });
    runFlow("aprovado", true); // estado final já visível ao abrir; a animação roda ao clicar
    window.addEventListener("resize", function () { var on = $("[data-flow][aria-pressed=true]"); if (on) runFlow(on.dataset.flow, true); });
  }

  /* ---------- simulador ---------- */
  function fillSelect(sel, vals, v) { sel.innerHTML = vals.map(function (x) { return '<option value="' + x + '"' + (x === v ? " selected" : "") + ">" + L(x) + "</option>"; }).join(""); }
  function syncForm() {
    var c = state.cliente;
    $("#f_nome").value = c.nome_cliente; $("#f_score").value = c.pontuacao_credito; $("#f_idade").value = c.idade;
    $("#f_tempo").value = c.tempo_emprego_anos; $("#f_renda").value = c.renda_mensal; $("#f_valor").value = c.valor_emprestimo;
    fillSelect($("#f_moradia"), MORADIA, c.tipo_moradia); fillSelect($("#f_final"), FINALIDADE, c.finalidade_emprestimo);
    $$("[data-inad]").forEach(function (b) { b.setAttribute("aria-pressed", String(+b.dataset.inad === c.inadimplencia_previa)); });
    outs();
  }
  function outs() {
    var c = state.cliente;
    $("#o_score").textContent = c.pontuacao_credito; $("#o_idade").textContent = c.idade + " anos";
    $("#o_tempo").textContent = num(c.tempo_emprego_anos, c.tempo_emprego_anos % 1 ? 1 : 0) + (c.tempo_emprego_anos === 1 ? " ano" : " anos");
  }
  function readForm() {
    var c = state.cliente;
    c.nome_cliente = $("#f_nome").value || "Cliente Padrao";
    c.pontuacao_credito = parseInt($("#f_score").value, 10); c.idade = parseInt($("#f_idade").value, 10);
    c.tempo_emprego_anos = parseFloat($("#f_tempo").value);
    var r = parseFloat($("#f_renda").value), v = parseFloat($("#f_valor").value);
    c.renda_mensal = isNaN(r) ? 0 : r; c.valor_emprestimo = isNaN(v) ? 0 : v;
    c.tipo_moradia = $("#f_moradia").value; c.finalidade_emprestimo = $("#f_final").value;
  }
  var simTimer = null;
  function initSim() {
    var f = $("#simForm");
    f.addEventListener("input", function () { readForm(); outs(); clearTimeout(simTimer); simTimer = setTimeout(runSim, 90); });
    f.addEventListener("submit", function (e) { e.preventDefault(); runSim(); });
    $$("[data-inad]").forEach(function (b) { b.addEventListener("click", function () { state.cliente.inadimplencia_previa = +b.dataset.inad; syncForm(); runSim(); }); });
    $$("[data-preset]").forEach(function (b) {
      b.addEventListener("click", function () {
        var k = b.dataset.preset;
        if (k === "random") { var s = D.amostra[Math.floor(Math.random() * D.amostra.length)]; state.cliente = Object.assign({ nome_cliente: "Cliente da base" }, s); }
        else state.cliente = Object.assign({}, PRESETS[k]);
        syncForm(); runSim();
      });
    });
    syncForm();
  }
  var lastDecision = null;
  function runSim() {
    var c = Object.assign({}, state.cliente), out = $("#simResult");
    state.api.request("POST", "/predict", c).then(function (r) {
      if (!r.ok) {
        out.innerHTML = '<div class="alert ' + (r.status === 400 || r.status === 422 ? "warn" : "bad") + '"><b>' + esc(stTxt(r.status)) + "</b><br>" + esc(r.error) + "</div>" + restBlock(c, r);
        bindCopy(out); lastDecision = null; return;
      }
      var d = r.data, p = d.probabilidade_aprovacao, ind = d.indice_comprometimento_anual;
      out.innerHTML =
        '<div class="panel"><div class="res-top"><div class="gauge" id="gaugeBox"></div><div class="stamp res-stamp ' + (d.codigo_decisao ? "aprovado" : "reprovado") + '" id="resStamp">' + esc(d.decisao) + "<small>" + esc(d.faixa_risco) + "</small></div></div>" +
        '<div class="scorebar"><div class="mk" style="left:' + ((c.pontuacao_credito - 300) / 550 * 100) + '%"><span>' + c.pontuacao_credito + '</span></div></div><div class="scale-l"><span>300</span><span>Score de crédito</span><span>850</span></div></div>' +
        '<div class="facts">' +
          '<div class="fact"><div class="l">Risco de inadimplência</div><div class="v">' + pct(d.probabilidade_risco_inadimplencia) + "</div></div>" +
          '<div class="fact"><div class="l">Empréstimo ÷ renda anual</div><div class="v">' + num(ind * 100, 0) + '%</div><div class="meter" title="Limite da regra de negócio: 2.000% (20× a renda anual)"><span style="width:' + Math.max(1.5, Math.min(100, ind / 20 * 100)) + "%;background:" + (ind > 1 ? "var(--warn)" : "var(--brand)") + '"></span></div><div class="l">limite: 20× (2.000%)</div></div>' +
          '<div class="fact"><div class="l">Resposta</div><div class="v">' + fmtMs(r.ms) + '</div><div class="l">' + esc(state.api.name) + "</div></div>" +
        "</div>" +
        '<div class="panel"><h3>E se o score fosse outro?</h3><p class="sub">Mesmo cliente, score de 300 a 850. A linha marca o ponto de virada da decisão.</p><div id="sweep"></div></div>' +
        restBlock(c, r);
      $("#gaugeBox").appendChild(Gauge(p));
      var st = $("#resStamp"); if (lastDecision !== d.decisao) { st.classList.add("pop"); lastDecision = d.decisao; }
      drawSweep(c);
      bindCopy(out);
    });
  }
  function restBlock(c, r) {
    return '<details class="rest"><summary>Requisição e resposta REST</summary>' +
      '<div class="lbl"><span class="pill s200">POST</span> /predict</div><pre class="code"><code>' + jsonHtml(c) + "</code></pre>" +
      '<div class="lbl"><span class="pill ' + stCls(r.status) + '">' + esc(stTxt(r.status)) + "</span></div>" + (r.data ? '<pre class="code"><code>' + jsonHtml(r.data) + "</code></pre>" : "") +
      '<div class="lbl">cURL</div><pre class="code"><button class="copy" type="button">Copiar</button><code>' + esc(curl("POST", "/predict", c)) + "</code></pre></details>";
  }
  function drawSweep(c) {
    var xs = [], ys = [];
    for (var s = 300; s <= 850; s += 10) { xs.push(s); ys.push(Engine.proba(Object.assign({}, c, { pontuacao_credito: s }))); }
    var box = $("#sweep"), cur = Math.round((c.pontuacao_credito - 300) / 10);
    var flip = null; for (var i = 1; i < ys.length; i++) if (ys[i - 1] <= 0.5 && ys[i] > 0.5) { flip = xs[i]; break; }
    box.innerHTML = "";
    box.appendChild(LineChart(xs, ys, { w: fitW(box, 620), h: 240, x0: 300, x1: 850, xticks: [300, 400, 500, 600, 700, 800, 850], xTitle: "Score de crédito",
      hline: 0.5, hlineLabel: "aprova acima de 50%", area: true, mark: c.pontuacao_credito, markIndex: Math.min(xs.length - 1, cur),
      markLabel: "score atual: " + c.pontuacao_credito, label: "Probabilidade de aprovação por score" }));
    box.insertAdjacentHTML("beforeend", '<p class="sub" style="margin:8px 0 0">' + (flip ? "Para este perfil, a aprovação começa por volta de <b>" + flip + " pontos</b>." : ys[ys.length - 1] <= 0.5 ? "Para este perfil, nem o score máximo aprova: outros fatores pesam mais." : "Para este perfil, qualquer score aprova.") + "</p>");
  }

  /* ---------- motor (navegador × API real) ---------- */
  function setStatus(kind, t) { var s = $("#apiStatus"); s.className = "status-dot " + kind; s.querySelector("span").textContent = t; }
  function initEngine() {
    var box = $("#apiUrlBox"), url = $("#apiUrl");
    try { var sv = localStorage.getItem("creditApiUrl"); if (sv) url.value = sv; } catch (e) {}
    var hint = document.createElement("details"); hint.className = "rest"; hint.hidden = true; hint.style.flexBasis = "100%";
    hint.innerHTML = '<summary>Como liberar o acesso do navegador à API (CORS)</summary><p class="sub" style="margin:10px 0">O navegador só aceita respostas de outra origem se a API permitir. Adicione ao <code>main.py</code>, logo depois de criar o <code>app</code>:</p>' +
      '<pre class="code"><button class="copy" type="button">Copiar</button><code>from fastapi.middleware.cors import CORSMiddleware\n\napp.add_middleware(\n    CORSMiddleware,\n    allow_origins=["*"],\n    allow_methods=["GET", "POST"],\n    allow_headers=["Content-Type"],\n)</code></pre>';
    $(".engine").appendChild(hint); bindCopy(hint);
    function apply() {
      var remote = $('input[name="engine"]:checked').value === "remote";
      box.hidden = !remote; hint.hidden = !remote;
      state.api = remote ? new RemoteApi(url.value) : LocalApi;
      setStatus("", "Verificando…");
      state.api.request("GET", "/health").then(function (r) {
        if (r.ok && r.data.modelo_carregado_em_ram) setStatus("ok", (remote ? "API online" : "Modelo no navegador") + " · " + fmtMs(r.ms));
        else setStatus("err", r.error || "Modelo indisponível");
        runSim();
      });
    }
    $$('input[name="engine"]').forEach(function (r) { r.addEventListener("change", apply); });
    $("#apiConnect").addEventListener("click", function () { try { localStorage.setItem("creditApiUrl", url.value); } catch (e) {} apply(); });
    apply();
  }

  /* ---------- laboratório ---------- */
  function initLab() {
    var sc = D.api_real, sel = $("#labPreset");
    sel.innerHTML = sc.map(function (s, i) { return '<option value="' + i + '">' + esc(s.name) + " · " + s.expect + "</option>"; }).join("");
    function load(i) {
      var s = sc[i]; $("#labMethod").value = s.method; $("#labPath").value = s.path;
      sel.value = String(i); $("#labBody").value = s.body ? JSON.stringify(s.body, null, 2) : ""; toggleBody();
    }
    function toggleBody() { $("#labBodyWrap").hidden = $("#labMethod").value === "GET"; }
    sel.addEventListener("change", function () { load(+sel.value); });
    $("#labMethod").addEventListener("change", toggleBody);
    $("#labForm").addEventListener("submit", function (e) { e.preventDefault(); sendLab(); });
    $("#labFormat").addEventListener("click", function () {
      try { $("#labBody").value = JSON.stringify(JSON.parse($("#labBody").value), null, 2); }
      catch (e) { $("#labOut").innerHTML = '<div class="alert warn">O corpo não é um JSON válido: ' + esc(e.message) + "</div>"; }
    });
    $("#labAll").addEventListener("click", runAll);
    load(3);
  }
  function sendLab() {
    var m = $("#labMethod").value, path = $("#labPath").value.trim() || "/", body = m === "GET" ? undefined : $("#labBody").value;
    var preset = D.api_real[+$("#labPreset").value], same = preset && preset.method === m && preset.path === path && (m === "GET" || sameJson(body, preset.body));
    state.api.request(m, path, body).then(function (r) {
      $("#labOut").innerHTML = '<div class="resp-head"><span class="pill ' + stCls(r.status) + '" style="font-size:14px;padding:5px 12px">' + esc(stTxt(r.status)) + '</span><span class="muted mono" style="font-size:13px">' + fmtMs(r.ms) + " · " + esc(state.api.name) + "</span></div>" +
        (r.data ? '<pre class="code"><code>' + jsonHtml(r.data) + "</code></pre>" : '<div class="alert bad">' + esc(r.error) + "</div>") +
        (same ? '<p class="sub" style="margin:10px 0 0">Na API real, este cenário devolveu <b>' + esc(stTxt(preset.status)) + "</b>" + (preset.status === r.status ? " também." : ".") + "</p>" : "") +
        '<p class="sub" style="margin:12px 0 6px">cURL equivalente</p><pre class="code"><button class="copy" type="button">Copiar</button><code>' + esc(curl(m, path, body)) + "</code></pre>";
      bindCopy($("#labOut"));
    });
  }
  function sameJson(a, b) { try { return JSON.stringify(JSON.parse(a)) === JSON.stringify(b); } catch (e) { return false; } }
  function runAll() {
    var sc = D.api_real, rows = [], chain = Promise.resolve();
    sc.forEach(function (s) { chain = chain.then(function () { return state.api.request(s.method, s.path, s.body === null ? undefined : s.body).then(function (r) { rows.push({ s: s, r: r }); }); }); });
    chain.then(function () {
      var ok = rows.filter(function (x) { return x.r.status === x.s.expect; }).length;
      var same = rows.filter(function (x) { return x.r.status === x.s.status && JSON.stringify(x.r.data) === JSON.stringify(x.s.response); }).length;
      $("#labScore").innerHTML = '<span class="tag ok">' + ok + "/" + rows.length + " com o status esperado</span>" +
        (state.api === LocalApi ? '<span class="tag brand">' + same + "/" + rows.length + " respostas idênticas às da API real</span>" : "");
      $("#labTable").innerHTML = "<thead><tr><th>Cenário</th><th>Rota</th><th>Esperado</th><th>" + esc(state.api === LocalApi ? "Navegador" : "API real agora") + "</th><th>API real (gravado)</th><th></th><th>Motivo</th></tr></thead><tbody>" +
        rows.map(function (x) {
          var good = x.r.status === x.s.expect;
          return "<tr><td>" + esc(x.s.name) + '</td><td class="m">' + x.s.method + " " + esc(x.s.path) + '</td><td class="m">' + x.s.expect +
            '</td><td><span class="pill ' + stCls(x.r.status) + '">' + (x.r.status || "—") + '</span></td><td><span class="pill ' + stCls(x.s.status) + '">' + x.s.status + "</span></td><td>" +
            (good ? '<span class="yes" aria-label="aprovado">✓</span>' : '<span class="no" aria-label="reprovado">✗</span>') + '</td><td class="why">' + esc(x.r.ok ? (x.r.data.decisao ? x.r.data.decisao + " · " + pct(x.r.data.probabilidade_aprovacao) : "online") : x.r.error) + "</td></tr>";
        }).join("") + "</tbody>";
    });
  }

  /* ---------- desempenho ---------- */
  function initPerf() {
    var M = D.metricas, parts = [["treino", "Treino (70%)"], ["validacao", "Validação (15%)"], ["teste", "Teste (15%)"]];
    $("#metricsTbl").innerHTML = "<thead><tr><th>Partição</th><th>Acurácia</th><th>Precisão</th><th>Recall</th><th>F1</th><th>ROC-AUC</th></tr></thead><tbody>" +
      parts.map(function (p) { var m = M[p[0]]; return "<tr" + (p[0] === "teste" ? ' class="hl"' : "") + "><td>" + p[1] + "</td><td>" + pct(m.acuracia) + "</td><td>" + pct(m.precisao) + "</td><td>" + pct(m.recall) + "</td><td>" + num(m.f1_score, 3) + "</td><td>" + num(m.roc_auc, 3) + "</td></tr>"; }).join("") + "</tbody>";
    var gap = (M.treino.acuracia - M.teste.acuracia) * 100;
    $("#gapNote").innerHTML = "<b>Leitura:</b> o modelo acerta " + pct(M.treino.acuracia) + " no treino e " + pct(M.teste.acuracia) + " no teste, uma diferença de " + num(gap, 1) + " pontos. É um sobreajuste leve: validação e teste concordam, então o número de teste é confiável.";

    var roc = D.roc, rb = $("#rocChart");
    rb.appendChild(LineChart(roc.fpr, roc.tpr, { w: fitW(rb, 560), h: 300, x0: 0, x1: 1, xticks: [0, 0.25, 0.5, 0.75, 1], fmtX: function (v) { return Math.round(v * 100) + "%"; },
      xTitle: "Falsos aprovados (taxa de falso positivo)", area: true, diag: true, label: "Curva ROC, AUC " + num(roc.auc, 3) }));
    rb.insertAdjacentHTML("beforeend", '<p class="sub" style="margin:8px 0 0">Área sob a curva: <b>' + num(roc.auc, 3) + "</b>. A linha tracejada é um modelo que chuta (0,5).</p>");

    var thr = $("#thr");
    function upd() {
      var t = parseFloat(thr.value), y = D.test.y, p = D.test.p, tp = 0, fp = 0, tn = 0, fn = 0;
      for (var i = 0; i < y.length; i++) { var a = p[i] > t; if (a && y[i]) tp++; else if (a) fp++; else if (y[i]) fn++; else tn++; }
      var n = y.length, prec = tp / Math.max(1, tp + fp), rec = tp / Math.max(1, tp + fn), appr = (tp + fp) / n;
      $("#thrOut").textContent = pct(t, 0);
      $("#thrCm").innerHTML = '<div></div><div class="ax">Modelo aprova</div><div class="ax">Modelo reprova</div>' +
        '<div class="ay">Bom pagador</div><div class="c tp">' + tp + '<small>aprovados certos</small></div><div class="c fn">' + fn + '<small>negócio perdido</small></div>' +
        '<div class="ay">Mau pagador</div><div class="c fp">' + fp + '<small>risco aceito</small></div><div class="c tn">' + tn + "<small>reprovados certos</small></div>";
      $("#thrStats").innerHTML = [["Taxa de aprovação", pct(appr)], ["Precisão das aprovações", pct(prec)], ["Bons pagadores atendidos", pct(rec)], ["Acurácia", pct((tp + tn) / n)]]
        .map(function (s) { return '<div class="fact"><div class="l">' + s[0] + '</div><div class="v">' + s[1] + "</div></div>"; }).join("");
      $("#thrNote").innerHTML = t > 0.5 ? "Política <b>conservadora</b>: menos risco aceito (" + fp + " maus pagadores aprovados), mas " + fn + " bons clientes ficam sem crédito." :
        t < 0.5 ? "Política <b>agressiva</b>: mais negócio fechado, com " + fp + " maus pagadores aprovados." : "Limiar padrão da API (50%): equilíbrio entre risco aceito e negócio perdido.";
    }
    thr.addEventListener("input", upd); upd();

    var NAMES = { pontuacao_credito: "Score de crédito", inadimplencia_previa: "Inadimplência anterior", valor_emprestimo: "Valor do empréstimo", renda_mensal: "Renda mensal", tempo_emprego_anos: "Tempo de emprego", tipo_moradia: "Moradia", idade: "Idade", finalidade_emprestimo: "Finalidade" };
    var ib = $("#impChart");
    ib.appendChild(HBars(D.importance.map(function (d, i) { return { k: NAMES[d.feature], v: d.mean, cls: i === 0 ? "ch-brand" : "ch-gold", label: "−" + num(d.mean, 3).replace(".", ",") }; }),
      { w: fitW(ib, 900, 340), labelW: 170, label: "Importância por permutação" }));
    ib.insertAdjacentHTML("beforeend", '<p class="sub" style="margin:8px 0 0">O score sozinho responde pela maior parte do poder do modelo. Embaralhá-lo derruba a ROC-AUC em ' + num(D.importance[0].mean, 2) + " ponto.</p>");
  }

  /* ---------- dados ---------- */
  var RATE_TITLES = { faixa_score: "Aprovação por faixa de score", comprometimento: "Aprovação pelo peso do empréstimo na renda anual", inadimplencia: "Aprovação por inadimplência anterior", moradia: "Aprovação por tipo de moradia", finalidade: "Aprovação por finalidade", faixa_idade: "Aprovação por idade" };
  function rateNote(key, rows) {
    var best = rows.reduce(function (a, b) { return b.rate > a.rate ? b : a; }), worst = rows.reduce(function (a, b) { return b.rate < a.rate ? b : a; });
    var lb = function (r) { return key === "inadimplencia" ? (r.k === "1" ? "com inadimplência" : "sem inadimplência") : L(r.k); };
    return "<b>" + esc(lb(best)) + "</b> tem " + pct(best.rate, 0) + " de aprovação, contra " + pct(worst.rate, 0) + " em <b>" + esc(lb(worst)) + "</b>.";
  }
  function showRate(key) {
    var rows = D.rates[key], box = $("#rateChart");
    $("#rateTitle").textContent = RATE_TITLES[key];
    box.innerHTML = "";
    box.appendChild(HBars(rows.map(function (r) {
      var k = key === "inadimplencia" ? (r.k === "1" ? "Teve" : "Nunca teve") : L(r.k);
      return { k: k, v: r.rate, cls: r.rate >= 0.6 ? "ch-ok" : r.rate >= 0.35 ? "ch-gold" : "ch-bad", label: pct(r.rate, 0) + " · " + num(r.n) };
    }), { w: fitW(box, 560), max: 1, labelW: 130, rowH: 46, label: RATE_TITLES[key] }));
    $("#rateNote").innerHTML = rateNote(key, rows);
    $$("#rateTabs .tab").forEach(function (t) { t.setAttribute("aria-selected", String(t.dataset.rate === key)); });
  }
  function scoreHist() {
    var h = D.score_hist, box = $("#scoreHist"), f = frame(fitW(box, 560), 260, { t: 12, r: 12, b: 42, l: 40 }), s = svg(f.w, f.h, "Distribuição do score por decisão");
    var tot = h.aprovado.map(function (a, i) { return a + h.reprovado[i]; }), max = nice(Math.max.apply(null, tot) * 1.1), bw = f.iw / tot.length;
    yGrid(s, f, max, "auto", function (v) { return Math.round(v); });
    tot.forEach(function (_, i) {
      var x = f.m.l + i * bw + 1, ha = f.ih * h.aprovado[i] / max, hr = f.ih * h.reprovado[i] / max, base = f.m.t + f.ih;
      var r1 = el("rect", { x: x, y: base - hr, width: bw - 2, height: hr, "class": "ch-bad" }, s); tip(r1, h.bins[i] + "–" + h.bins[i + 1] + ": " + h.reprovado[i] + " reprovados");
      var r2 = el("rect", { x: x, y: base - hr - ha, width: bw - 2, height: ha, "class": "ch-brand" }, s); tip(r2, h.bins[i] + "–" + h.bins[i + 1] + ": " + h.aprovado[i] + " aprovados");
    });
    [300, 400, 500, 600, 700, 800].forEach(function (v) { text(s, f.m.l + (v - 300) / 550 * f.iw, f.h - 22, v, "ch-lbl"); });
    text(s, f.m.l + f.iw / 2, f.h - 2, "Score de crédito", "ch-ttl");
    box.appendChild(s);
  }
  function initData() {
    $$("#rateTabs .tab").forEach(function (t) { t.addEventListener("click", function () { showRate(t.dataset.rate); }); });
    showRate("faixa_score"); scoreHist();
    $("#nullChips").innerHTML = Object.keys(D.dataset.nulls).map(function (k) { return '<span class="chip">' + k + ": " + D.dataset.nulls[k] + " vazios</span>"; }).join("");
  }

  /* ---------- latência ---------- */
  function initLatency() {
    var L0 = D.latency, box = $("#latChart"), smp = L0.samples, lo = 40, hi = 150, nb = 22, bins = new Array(nb).fill(0);
    smp.forEach(function (v) { bins[Math.max(0, Math.min(nb - 1, Math.floor((v - lo) / (hi - lo) * nb)))]++; });
    var f = frame(fitW(box, 560), 250, { t: 22, r: 14, b: 42, l: 40 }), s = svg(f.w, f.h, "Histograma de latência"), max = nice(Math.max.apply(null, bins) * 1.1), bw = f.iw / nb;
    yGrid(s, f, max, "auto", function (v) { return Math.round(v); });
    bins.forEach(function (n, i) { var hh = f.ih * n / max; var r = el("rect", { x: f.m.l + i * bw + 1, y: f.m.t + f.ih - hh, width: bw - 2, height: hh, rx: 2, "class": "ch-brand" }, s); tip(r, num(lo + i * 5) + "–" + num(lo + i * 5 + 5) + " ms: " + n); });
    function X(v) { return f.m.l + (v - lo) / (hi - lo) * f.iw; }
    [["p50", L0.p50], ["p95", L0.p95]].forEach(function (q) { el("line", { x1: X(q[1]), x2: X(q[1]), y1: f.m.t - 6, y2: f.m.t + f.ih, "class": "ch-mark" }, s); text(s, X(q[1]) + 4, f.m.t + 4, q[0], "ch-lbl sm", "start"); });
    for (var t = 40; t <= 150; t += 20) text(s, X(t), f.h - 22, t, "ch-lbl");
    text(s, f.m.l + f.iw / 2, f.h - 2, "Milissegundos por requisição", "ch-ttl");
    box.appendChild(s);
    $("#latStats").innerHTML = [["Mediana", L0.p50], ["p95", L0.p95], ["p99", L0.p99], ["GET /health", L0.health_p50]].map(function (q) { return '<span class="tag brand">' + q[0] + ": " + fmtMs(q[1]) + "</span>"; }).join("");
    var ob = $("#optChart");
    ob.appendChild(HBars(D.otimizacao.map(function (o, i) { return { k: ["Hoje", "n_jobs=1", "+ 1 chamada"][i], v: o.ms, cls: i === 0 ? "ch-bad" : i === 1 ? "ch-gold" : "ch-ok", label: num(o.ms, 1) + " ms" }; }),
      { w: fitW(ob, 560), labelW: 110, max: 80, label: "Tempo por predição" }));
  }

  /* ---------- Streamlit ---------- */
  function initShots() {
    var shots = [["st_simulador", "Simulador: Carlos aprovado com 95,7%, faixa Baixo Risco. O formulário do Streamlit vem com a finalidade Educação; com Negócios, como no README, a chance é 95,0%."],
      ["st_lab", "Laboratório: idade 15 e score 990 barrados pelo Pydantic com 422 Unprocessable Entity."],
      ["st_metricas", "Métricas do pipeline lidas do relatorio_metricas.json gerado na Fase 1."]];
    function show(i) { $("#shotImg").src = "assets/" + shots[i][0] + ".webp"; $("#shotImg").alt = "Painel Streamlit. " + shots[i][1]; $("#shotCap").textContent = shots[i][1];
      $$("[data-shot]").forEach(function (t) { t.setAttribute("aria-selected", String(+t.dataset.shot === i)); }); }
    $$("[data-shot]").forEach(function (t) { t.addEventListener("click", function () { show(+t.dataset.shot); }); });
    show(0);
  }

  /* ---------- tema, copiar, apresentação ---------- */
  function bindCopy(root) {
    $$(".copy", root).forEach(function (b) {
      if (b.dataset.bound) return; b.dataset.bound = "1";
      b.addEventListener("click", function () {
        var code = b.parentElement.querySelector("code"), t = code.innerText;
        function done(m) { b.textContent = m; setTimeout(function () { b.textContent = "Copiar"; }, 1500); }
        function sel() { var r = document.createRange(); r.selectNodeContents(code); var s = getSelection(); s.removeAllRanges(); s.addRange(r); done("Selecionado"); }
        try { navigator.clipboard.writeText(t).then(function () { done("Copiado"); }, sel); } catch (e) { sel(); }
      });
    });
  }
  function initTheme() {
    var root = document.documentElement;
    try { var t = localStorage.getItem("creditTheme"); if (t) root.setAttribute("data-theme", t); } catch (e) {}
    $("#themeBtn").addEventListener("click", function () {
      var dark = root.getAttribute("data-theme") === "dark" || (!root.getAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches);
      root.setAttribute("data-theme", dark ? "light" : "dark");
      try { localStorage.setItem("creditTheme", dark ? "light" : "dark"); } catch (e) {}
    });
  }
  function initPresent() {
    var root = document.documentElement, slides = $$("[data-slide]");
    function cur() { var y = scrollY + innerHeight / 3, idx = 0; slides.forEach(function (s, i) { if (s.offsetTop <= y) idx = i; }); return idx; }
    function go(i) { i = Math.max(0, Math.min(slides.length - 1, i)); slides[i].scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" }); }
    function count() { $("#hudCount").textContent = (cur() + 1) + " / " + slides.length; }
    function enter() {
      root.classList.add("present"); count();
      try { if (root.requestFullscreen) root.requestFullscreen().catch(function () {}); } catch (e) {}
      setTimeout(function () { go(cur()); }, 50);
    }
    function exit() {
      root.classList.remove("present");
      try { if (document.fullscreenElement && document.exitFullscreen) document.exitFullscreen().catch(function () {}); } catch (e) {}
    }
    $("#presentBtn").addEventListener("click", function () { root.classList.contains("present") ? exit() : enter(); });
    // seções mais altas que a tela: rola dentro dela antes de pular para a próxima
    function smooth() { return matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth"; }
    function next() {
      var s = slides[cur()], r = s.getBoundingClientRect();
      if (r.bottom > innerHeight + 24) scrollBy({ top: Math.min(innerHeight * 0.85, r.bottom - innerHeight), behavior: smooth() }); else go(cur() + 1);
    }
    function prev() {
      var s = slides[cur()], r = s.getBoundingClientRect();
      if (r.top < -24) scrollBy({ top: Math.max(-innerHeight * 0.85, r.top), behavior: smooth() }); else go(cur() - 1);
    }
    $("#hudPrev").addEventListener("click", prev);
    $("#hudNext").addEventListener("click", next);
    $("#hudExit").addEventListener("click", exit);
    addEventListener("scroll", function () { if (root.classList.contains("present")) count(); }, { passive: true });
    document.addEventListener("keydown", function (e) {
      var tag = (e.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;
      if (e.key === "p" || e.key === "P") { root.classList.contains("present") ? exit() : enter(); return; }
      if (!root.classList.contains("present")) return;
      if (["ArrowRight", "ArrowDown", "PageDown", " "].indexOf(e.key) !== -1) { e.preventDefault(); next(); }
      else if (["ArrowLeft", "ArrowUp", "PageUp"].indexOf(e.key) !== -1) { e.preventDefault(); prev(); }
      else if (e.key === "Escape") exit();
    });
    document.addEventListener("fullscreenchange", function () { if (!document.fullscreenElement && root.classList.contains("present")) root.classList.remove("present"); });
    // destaca o capítulo atual no menu
    var links = $$(".nav-links a");
    addEventListener("scroll", function () { var id = slides[cur()].id; links.forEach(function (a) { a.classList.toggle("on", a.getAttribute("href") === "#" + id); }); }, { passive: true });
  }

  function boot() {
    initTheme();
    if (!G.CREDIT_MODEL || !G.CREDIT_DATA) { document.body.insertAdjacentHTML("afterbegin", '<div class="alert bad" style="margin:16px">Não foi possível carregar assets/model.js ou assets/data.js.</div>'); return; }
    initHero(); initFlow(); initSim(); initEngine(); initLab(); initPerf(); initData(); initLatency(); initShots(); initPresent(); bindCopy(document);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})(typeof window !== "undefined" ? window : globalThis);
