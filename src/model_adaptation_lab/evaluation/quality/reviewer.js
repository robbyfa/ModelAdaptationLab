/* No dependencies or network calls. All source/model text enters the DOM as text. */
"use strict";
const cp = (s) => Array.from(s);
const slice = (s, a, b) => cp(s).slice(a, b).join("");
const clone = (x) => JSON.parse(JSON.stringify(x));
const support = ["supported", "contradicted", "not_supported", "unverifiable"];
function freshClaim(text, start, end) {
  return {
    start,
    end,
    text,
    kind: null,
    reference_support: null,
    retrieved_support: null,
    reference_evidence: [],
    retrieved_evidence: [],
    cited_document_ids: [],
    citation_support: null,
    rationale: "",
  };
}
function splitClaim(r, index, position) {
  const c = r.claims[index];
  if (
    !Number.isInteger(position) ||
    position <= 0 ||
    position >= cp(c.text).length
  )
    throw Error("Place the caret inside the claim.");
  r.claims.splice(
    index,
    1,
    freshClaim(slice(c.text, 0, position), c.start, c.start + position),
    freshClaim(slice(c.text, position), c.start + position, c.end),
  );
}
function mergeClaim(r, index, answer) {
  const a = r.claims[index],
    b = r.claims[index + 1];
  if (!b) throw Error("No following claim.");
  r.claims.splice(
    index,
    2,
    freshClaim(slice(answer, a.start, b.end), a.start, b.end),
  );
}
function validate(r, c, docs) {
  const errors = [],
    check = (ok, msg) => {
      if (!ok) errors.push(msg);
    };
  const failed = c.status === "failed",
    empty = !c.answer.trim(),
    present = !failed && !empty;
  check(
    ["answer", "abstain", "mixed", "empty", "failed"].includes(r.behavior),
    "Choose response behavior.",
  );
  check(!!r.rationale?.trim(), "Give a response rationale.");
  check(
    failed
      ? r.behavior === "failed"
      : empty
        ? r.behavior === "empty"
        : !["empty", "failed"].includes(r.behavior),
    "Behavior conflicts with generation availability.",
  );
  check(
    r.correctness === null ||
      (Number.isInteger(r.correctness) &&
        r.correctness >= 0 &&
        r.correctness <= 3),
    "Correctness must be 0–3 or unavailable.",
  );
  check(
    (present && c.answerable) || r.correctness === null,
    "Correctness unavailable for missing output/reference.",
  );
  check(
    r.correctness !== null || !!r.correctness_unavailable_reason?.trim(),
    "Explain unavailable correctness.",
  );
  check(
    !(r.behavior === "abstain" && c.answerable) ||
      r.correctness === 0 ||
      r.correctness === null,
    "Answerable abstention cannot have positive correctness.",
  );
  const reviewed = r.reviewed_document_ids;
  check(
    new Set(reviewed).size === reviewed.length &&
      reviewed.every((id) => Object.hasOwn(docs, id)),
    "Unknown or repeated reviewed document.",
  );
  for (const scope of ["reference", "retrieved"]) {
    const applicable =
        present && (scope === "reference" || c.retrieval_applicable),
      value = r[scope + "_sufficiency"];
    const available =
      scope === "reference" ? reviewed.length : c.retrieved_passages.length;
    check(
      applicable
        ? ["sufficient", "insufficient", "uncertain"].includes(value)
        : value === null,
      scope + ": choose applicable sufficiency.",
    );
    check(
      !applicable || available || value === "uncertain",
      scope + ": absent evidence requires uncertain sufficiency.",
    );
  }
  let cursor = 0;
  r.claims.forEach((claim, i) => {
    const tag = "Claim " + (i + 1) + ": ",
      factual = claim.kind === "factual";
    check(
      Number.isInteger(claim.start) &&
        Number.isInteger(claim.end) &&
        cursor <= claim.start &&
        claim.start < claim.end &&
        claim.end <= cp(c.answer).length &&
        !slice(c.answer, cursor, claim.start).trim() &&
        slice(c.answer, claim.start, claim.end) === claim.text,
      tag + "invalid span coverage.",
    );
    cursor = claim.end;
    check(
      ["factual", "nonfactual"].includes(claim.kind),
      tag + "choose claim kind.",
    );
    check(!!claim.rationale?.trim(), tag + "give a rationale.");
    for (const scope of ["reference", "retrieved"]) {
      const applicable =
          factual && (scope === "reference" || c.retrieval_applicable),
        label = claim[scope + "_support"],
        quotes = claim[scope + "_evidence"];
      check(
        applicable ? support.includes(label) : label === null,
        tag + scope + " support missing or inapplicable.",
      );
      check(applicable || !quotes.length, tag + "inapplicable quotations.");
      check(
        !["supported", "contradicted"].includes(label) || quotes.length > 0,
        tag + scope + " requires a quote.",
      );
      const available =
        scope === "reference" ? reviewed.length : c.retrieved_passages.length;
      check(
        !applicable || available || label === "unverifiable",
        tag + scope + " absent evidence requires unverifiable.",
      );
      for (const q of quotes) {
        const d = docs[q.document_id];
        check(
          Object.keys(q).sort().join(",") === "document_id,end,quote,start" &&
            d &&
            Number.isInteger(q.start) &&
            Number.isInteger(q.end) &&
            0 <= q.start &&
            q.start < q.end &&
            q.end <= cp(d.text).length &&
            !!q.quote.trim() &&
            slice(d.text, q.start, q.end) === q.quote,
          tag + "quote must match exact frozen source offsets.",
        );
        check(
          scope === "reference"
            ? reviewed.includes(q.document_id)
            : c.retrieved_passages.some(
                (p) =>
                  p.document_id === q.document_id &&
                  p.start <= q.start &&
                  p.end >= q.end,
              ),
          tag + "quote outside declared evidence scope.",
        );
      }
    }
    const cited = claim.cited_document_ids,
      label = claim.citation_support;
    const ids = Array.from(
      c.answer.matchAll(/\[([\p{L}\p{N}_.:-]+)\]/gu),
      (m) => m[1],
    );
    check(
      new Set(cited).size === cited.length &&
        cited.every((id) => ids.includes(id)),
      tag + "citation IDs must occur in answer.",
    );
    if (!factual || !c.retrieval_applicable)
      check(
        label === null && !cited.length,
        tag + "citation assessment not applicable.",
      );
    else {
      check(
        [...support, "missing"].includes(label),
        tag + "choose citation assessment.",
      );
      check(
        (cited.length === 0) === (label === "missing"),
        tag + "missing citation label conflicts with linked IDs.",
      );
      check(
        !["supported", "contradicted"].includes(label) ||
          (claim.retrieved_evidence.length > 0 &&
            cited.every((id) =>
              claim.retrieved_evidence.some((q) => q.document_id === id),
            )),
        tag + "citation support needs quoted cited passages.",
      );
    }
  });
  check(!slice(c.answer, cursor).trim(), "Claims omit trailing answer text.");
  check(
    !failed || !r.claims.length,
    "Failed output cannot have claim judgments.",
  );
  check(
    r.behavior !== "abstain" || !r.claims.some((c) => c.kind === "factual"),
    "Abstention with factual advice must be mixed.",
  );
  return errors;
}
// Export pure operations for synthetic Node/Python compatibility tests.
if (typeof module !== "undefined")
  module.exports = { cp, slice, splitClaim, mergeClaim, validate };
if (typeof document !== "undefined") boot();
function boot() {
  const data = JSON.parse(document.getElementById("payload").textContent);
  const $ = (id) => document.getElementById(id),
    cases = data.cases,
    docs = data.documents;
  $("rubric").textContent = data.rubric;
  let review = clone(data.template),
    done = new Set(),
    index = 0,
    active = 0,
    key = "",
    source = null,
    findAt = 0;
  const current = () => review.items[index],
    ccase = () => cases[index];
  function message(text) {
    $("message").textContent = text;
  }
  function element(tag, text, parent) {
    const el = document.createElement(tag);
    if (text !== undefined) el.textContent = text;
    if (parent) parent.append(el);
    return el;
  }
  function button(text, action, parent) {
    const b = element("button", text, parent);
    b.type = "button";
    b.onclick = () => {
      try {
        action();
      } catch (e) {
        message(e.message);
      }
    };
    return b;
  }
  function draft() {
    return {
      format: "model-lab-review-draft-v1",
      review,
      completed: [...done],
      index,
    };
  }
  function save() {
    try {
      localStorage.setItem(key, JSON.stringify(draft()));
      message("Saved locally. Download a draft backup before closing.");
    } catch (_) {
      message(
        "Browser storage unavailable. Download a draft backup to keep your work.",
      );
    }
  }
  function changed() {
    done.delete(ccase().item_id);
    save();
    progress();
  }
  function progress() {
    $("progress").textContent = done.size + " / " + cases.length + " completed";
    $("prev").disabled = index === 0;
    $("next").disabled = index === cases.length - 1;
    $("jump").replaceChildren();
    cases.forEach((c, i) => {
      const o = element(
        "option",
        i +
          1 +
          " / " +
          cases.length +
          (done.has(c.item_id) ? " · complete" : " · draft"),
        $("jump"),
      );
      o.value = i;
    });
    $("jump").value = index;
  }
  function select(label, obj, field, choices, parent, callback) {
    const l = element("label", label + " ", parent),
      s = element("select", undefined, l);
    for (const [value, text] of [
      [null, "— Choose / unavailable —"],
      ...choices.map((x) =>
        Array.isArray(x) ? x : [x, x.replaceAll("_", " ")],
      ),
    ]) {
      const o = element("option", text, s);
      o.value = value === null ? "" : String(value);
    }
    s.value = obj[field] === null ? "" : String(obj[field]);
    s.onchange = () => {
      obj[field] =
        s.value === ""
          ? null
          : field === "correctness"
            ? Number(s.value)
            : s.value;
      changed();
      if (callback) callback();
    };
    return s;
  }
  function textField(label, obj, field, parent) {
    const l = element("label", label, parent),
      t = element("textarea", undefined, l);
    t.value = obj[field];
    t.oninput = () => {
      obj[field] = t.value;
      changed();
    };
    return t;
  }
  function draw() {
    const c = ccase(),
      r = current();
    $("case").replaceChildren();
    element(
      "p",
      "Response " + (index + 1) + " · " + c.item_id,
      $("case"),
    ).className = "muted";
    for (const [label, value] of [
      ["Question", c.question],
      ["Generated answer", c.answer],
      ["Reference answer", c.reference_answer],
    ]) {
      element("h2", label, $("case"));
      element("pre", value || "Unavailable", $("case"));
    }
    element(
      "p",
      "Annotation: " +
        (c.answerable ? "answerable" : "unanswerable") +
        " · Generation: " +
        c.status +
        " · Finish: " +
        c.finish_reason,
      $("case"),
    ).className = c.finish_reason === "length" ? "notice" : "muted";
    $("rating").replaceChildren();
    select(
      "Correctness",
      r,
      "correctness",
      [
        [0, "0 — Incorrect"],
        [1, "1 — Relevant fragments; material error / missing necessary step"],
        [2, "2 — Mostly correct; minor error / secondary omission"],
        [3, "3 — Correct"],
      ],
      $("rating"),
    );
    textField(
      "Reason if correctness unavailable",
      r,
      "correctness_unavailable_reason",
      $("rating"),
    );
    select(
      "Response behavior",
      r,
      "behavior",
      ["answer", "abstain", "mixed", "empty", "failed"],
      $("rating"),
    );
    for (const scope of ["reference", "retrieved"])
      if (scope === "reference" || c.retrieval_applicable)
        select(
          scope + " evidence sufficiency",
          r,
          scope + "_sufficiency",
          ["sufficient", "insufficient", "uncertain"],
          $("rating"),
        );
    textField(
      "Response rationale / reference defects",
      r,
      "rationale",
      $("rating"),
    );
    $("errors").textContent = "";
    active = Math.min(active, Math.max(0, r.claims.length - 1));
    drawClaims();
    drawDocs();
    progress();
  }
  function drawClaims() {
    const r = current(),
      c = ccase();
    $("claims").replaceChildren();
    r.claims.forEach((claim, i) => {
      const box = element("div", undefined, $("claims"));
      box.className = "claim" + (i === active ? " active" : "");
      button(
        "Use claim " + (i + 1) + " for evidence",
        () => {
          active = i;
          drawClaims();
        },
        box,
      );
      element(
        "p",
        "Source span [" + claim.start + ":" + claim.end + "]",
        box,
      ).className = "muted";
      const text = element("textarea", undefined, box);
      text.value = claim.text;
      text.readOnly = true;
      text.setAttribute(
        "aria-label",
        "Claim " + (i + 1) + " text; place caret to split",
      );
      button(
        "Split at caret",
        () => {
          splitClaim(
            r,
            i,
            cp(text.value.substring(0, text.selectionStart)).length,
          );
          active = i;
          changed();
          drawClaims();
        },
        box,
      );
      const merge = button(
        "Merge with next",
        () => {
          mergeClaim(r, i, c.answer);
          active = i;
          changed();
          drawClaims();
        },
        box,
      );
      merge.disabled = i === r.claims.length - 1;
      select(
        "Claim kind",
        claim,
        "kind",
        ["factual", "nonfactual"],
        box,
        () => {
          claim.reference_support = null;
          claim.retrieved_support = null;
          claim.reference_evidence = [];
          claim.retrieved_evidence = [];
          claim.cited_document_ids = [];
          claim.citation_support = null;
          changed();
          drawClaims();
        },
      );
      if (claim.kind === "factual") {
        for (const scope of ["reference", "retrieved"])
          if (scope === "reference" || c.retrieval_applicable)
            select(scope + " support", claim, scope + "_support", support, box);
        if (c.retrieval_applicable) {
          select(
            "Citation support",
            claim,
            "citation_support",
            [...support, "missing"],
            box,
          );
          const ids = [
            ...new Set(
              Array.from(
                c.answer.matchAll(/\[([\p{L}\p{N}_.:-]+)\]/gu),
                (m) => m[1],
              ),
            ),
          ];
          element("p", "Link citations actually used for this claim:", box);
          ids.forEach((id) => {
            const l = element("label", undefined, box),
              cb = element("input", undefined, l);
            cb.type = "checkbox";
            cb.checked = claim.cited_document_ids.includes(id);
            l.append(document.createTextNode(id));
            cb.onchange = () => {
              claim.cited_document_ids = cb.checked
                ? [...claim.cited_document_ids, id]
                : claim.cited_document_ids.filter((v) => v !== id);
              changed();
            };
          });
        }
      }
      for (const scope of ["reference", "retrieved"])
        claim[scope + "_evidence"].forEach((q, j) => {
          const item = element("div", undefined, box);
          item.className = "quote";
          element(
            "p",
            scope + " · " + q.document_id + " [" + q.start + ":" + q.end + "]",
            item,
          );
          element("pre", q.quote, item);
          button(
            "Remove quote",
            () => {
              claim[scope + "_evidence"].splice(j, 1);
              changed();
              drawClaims();
            },
            item,
          );
        });
      textField("Claim rationale", claim, "rationale", box);
    });
    $("active").textContent = r.claims.length
      ? "Active claim: " + (active + 1)
      : "No claims in this response.";
  }
  function options() {
    return $("scope").value === "reference"
      ? ccase().reference_documents.map((id) => ({
          ...docs[id],
          start: 0,
          end: cp(docs[id].text).length,
        }))
      : ccase().retrieved_passages;
  }
  function drawDocs() {
    const entries = options();
    $("document").replaceChildren();
    entries.forEach((d, i) => {
      const o = element(
        "option",
        d.document_id +
          " · " +
          (d.title || "") +
          ($("scope").value === "retrieved"
            ? " [" + d.start + ":" + d.end + "]"
            : ""),
        $("document"),
      );
      o.value = i;
    });
    showDoc();
  }
  function showDoc() {
    source = options()[Number($("document").value)] || null;
    findAt = 0;
    $("source").textContent = source
      ? source.text
      : "No evidence available in this scope.";
    $("sourceinfo").textContent = source
      ? "Source offsets [" +
        source.start +
        ":" +
        source.end +
        "] · Select exact text below"
      : "";
    $("reviewed").disabled = !source;
    $("reviewed").checked =
      !!source && current().reviewed_document_ids.includes(source.document_id);
  }
  $("scope").onchange = drawDocs;
  $("document").onchange = showDoc;
  $("reviewed").onchange = () => {
    if (!source) return;
    const id = source.document_id;
    current().reviewed_document_ids = $("reviewed").checked
      ? [...new Set([...current().reviewed_document_ids, id])]
      : current().reviewed_document_ids.filter((x) => x !== id);
    changed();
  };
  $("find").onclick = () => {
    if (!source || !$("search").value) return;
    const text = source.text,
      term = $("search").value;
    let at = text.indexOf(term, findAt);
    if (at < 0) at = text.indexOf(term);
    if (at < 0) {
      message("No exact match in this document / passage.");
      return;
    }
    findAt = at + 1;
    const range = document.createRange();
    range.setStart($("source").firstChild, at);
    range.setEnd($("source").firstChild, at + term.length);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
    $("source").focus();
    const rect = range.getBoundingClientRect(),
      container = $("source").getBoundingClientRect();
    $("source").scrollTop += rect.top - container.top - 40;
    message("Exact match selected. Inspect context before attaching.");
  };
  // Prevent button focus from discarding the source selection.
  $("attach").onmousedown = (e) => e.preventDefault();
  $("attach").onclick = () => {
    const claim = current().claims[active],
      sel = window.getSelection();
    if (!source || !claim || claim.kind !== "factual" || !sel.rangeCount) {
      message("Select a factual claim and exact source text first.");
      return;
    }
    const range = sel.getRangeAt(0);
    if (
      range.startContainer !== $("source").firstChild ||
      range.endContainer !== $("source").firstChild ||
      !range.toString().trim()
    ) {
      message("Select text entirely inside this source passage.");
      return;
    }
    const q = {
      document_id: source.document_id,
      start:
        source.start + cp(source.text.substring(0, range.startOffset)).length,
      end: source.start + cp(source.text.substring(0, range.endOffset)).length,
      quote: range.toString(),
    };
    const scope = $("scope").value;
    if (
      scope === "reference" &&
      !current().reviewed_document_ids.includes(source.document_id)
    ) {
      message(
        "Declare this document reviewed before attaching reference evidence.",
      );
      return;
    }
    if (
      !claim[scope + "_evidence"].some(
        (x) => JSON.stringify(x) === JSON.stringify(q),
      )
    )
      claim[scope + "_evidence"].push(q);
    changed();
    drawClaims();
    message(
      "Quote attached with exact source offsets. Choose the support judgment yourself.",
    );
  };
  function navigate(i) {
    index = i;
    active = 0;
    save();
    draw();
  }
  $("prev").onclick = () => navigate(index - 1);
  $("next").onclick = () => navigate(index + 1);
  $("jump").onchange = () => navigate(Number($("jump").value));
  $("complete").onclick = () => {
    const errors = validate(current(), ccase(), docs);
    $("errors").textContent = errors.join("\n");
    if (!errors.length) {
      done.add(ccase().item_id);
      save();
      progress();
      message("Response complete. You may revise it; edits clear completion.");
    } else {
      done.delete(ccase().item_id);
      save();
      progress();
      message("Resolve the listed fields before marking complete.");
    }
  };
  $("independent").onchange = () => {
    review.independent = $("independent").checked;
    save();
  };
  function download(value, name) {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(value, null, 2) + "\n"], {
        type: "application/json",
      }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  $("backup").onclick = () => download(draft(), "review-draft.json");
  $("export").onclick = () => {
    const items = review.items.filter((r) => done.has(r.item_id));
    const errors = items.flatMap((r) =>
      validate(
        r,
        cases.find((c) => c.item_id === r.item_id),
        docs,
      ),
    );
    if (!items.length || errors.length) {
      message(
        "No valid completed ratings to export. Validate each response first.",
      );
      return;
    }
    download({ ...review, items }, "ratings.json");
    message(
      "Exported " +
        items.length +
        " completed ratings. Unfinished responses remain in your draft. Coordinator must run quality score for authoritative validation.",
    );
  };
  function restore(value) {
    if (
      value.format !== "model-lab-review-draft-v1" ||
      value.review?.packet_sha256 !== data.template.packet_sha256 ||
      value.review?.rubric_sha256 !== data.template.rubric_sha256 ||
      value.review?.reviewer_id !== review.reviewer_id ||
      value.review?.kind !== "human" ||
      value.review?.model_revision !== null ||
      value.review?.judge_prompt_sha256 !== null ||
      typeof value.review?.independent !== "boolean" ||
      value.review?.schema_version !== 1
    )
      throw Error("Draft belongs to a different packet, reviewer or schema.");
    if (
      !Array.isArray(value.review.items) ||
      value.review.items.length !== cases.length ||
      !Array.isArray(value.completed)
    )
      throw Error("Incomplete draft structure.");
    value.review.items.forEach((r, i) => {
      if (
        r.item_id !== cases[i].item_id ||
        !Array.isArray(r.claims) ||
        !Array.isArray(r.reviewed_document_ids) ||
        !["rationale", "correctness_unavailable_reason"].every(
          (k) => typeof r[k] === "string",
        )
      )
        throw Error("Invalid draft item.");
      for (const c of r.claims)
        if (
          typeof c.text !== "string" ||
          typeof c.rationale !== "string" ||
          ![
            "reference_evidence",
            "retrieved_evidence",
            "cited_document_ids",
          ].every((k) => Array.isArray(c[k]))
        )
          throw Error("Invalid draft claim.");
      validate(r, cases[i], docs);
    });
    const completed = new Set(value.completed);
    for (const id of completed) {
      const i = cases.findIndex((c) => c.item_id === id);
      if (i < 0 || validate(value.review.items[i], cases[i], docs).length)
        throw Error("Draft marks an invalid response complete.");
    }
    review = clone(value.review);
    done = completed;
    index =
      Number.isInteger(value.index) &&
      value.index >= 0 &&
      value.index < cases.length
        ? value.index
        : 0;
  }
  $("restore").onchange = async () => {
    try {
      const file = $("restore").files[0];
      if (!file) return;
      const value = JSON.parse(await file.text());
      if (
        !confirm(
          "Replace this workspace with your selected draft? Download a backup first if needed.",
        )
      )
        return;
      restore(value);
      $("independent").checked = review.independent;
      save();
      draw();
    } catch (e) {
      message("Draft not loaded: " + e.message);
    } finally {
      $("restore").value = "";
    }
  };
  $("start").onclick = () => {
    const id = $("identity").value.trim();
    if (!id) {
      message("Enter your reviewer ID.");
      return;
    }
    review.reviewer_id = id;
    key =
      "model-lab-review-v1:" +
      data.workspace +
      ":" +
      review.packet_sha256 +
      ":" +
      id;
    try {
      const saved = localStorage.getItem(key);
      if (saved) restore(JSON.parse(saved));
    } catch (e) {
      message(
        "Saved progress unavailable: " +
          e.message +
          ". You can restore a downloaded draft.",
      );
    }
    $("login").hidden = true;
    $("app").hidden = false;
    $("who").textContent = "Reviewer: " + id;
    $("independent").checked = review.independent;
    draw();
  };
}
