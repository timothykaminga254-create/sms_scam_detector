/* =========================================================
   predictor.js
   Uses the model trained by train_model.py (model.js) to
   score SMS messages in the browser.
   Preprocessing here MUST match train_model.py exactly.
   ========================================================= */
(function () {
  const STOP = new Set(
    ("a an the and or but if is are was were be been am i me my you your we our they them he she it its " +
     "to of in on at for with as by from this that these those so do does did have has had will would can " +
     "could should just not no yes").split(" ")
  );

  function tokenize(text) {
    let t = text.toLowerCase();
    t = t.replace(/https?:\/\/\S+|www\.\S+|\b\S+\.(?:com|biz|xyz|info|co|net)\b\S*/g, " urltoken ");
    t = t.replace(/[$£€]\s?\d[\d,.]*|\d[\d,.]*\s?(?:usd|dollars|pounds)/g, " moneytoken ");
    t = t.replace(/\b\d{5,}\b/g, " longnumtoken ");
    t = t.replace(/\b\d+\b/g, " numtoken ");
    t = t.replace(/[^a-z\s]/g, " ");
    const toks = t.split(/\s+/).filter(w => w.length > 1 && !STOP.has(w));
    if (/[A-Z]{4,}/.test(text)) toks.push("allcapstoken");
    if (/!{2,}/.test(text)) toks.push("excitetoken");
    return toks;
  }

  function analyzer(text) {
    const toks = tokenize(text);
    const out = toks.slice();
    for (let i = 0; i < toks.length - 1; i++) out.push(toks[i] + " " + toks[i + 1]);
    return out;
  }

  // TF-IDF (sublinear tf, L2 norm) followed by logistic regression
  window.predictTrained = function (text) {
    const M = window.TRAINED_MODEL;
    if (!M) return null;

    const counts = {};
    for (const term of analyzer(text)) {
      if (M.terms[term]) counts[term] = (counts[term] || 0) + 1;
    }

    const weights = {};
    let norm = 0;
    for (const [term, c] of Object.entries(counts)) {
      const w = (1 + Math.log(c)) * M.terms[term][0];
      weights[term] = w;
      norm += w * w;
    }
    norm = Math.sqrt(norm) || 1;

    let score = M.intercept;
    const words = [];
    for (const [term, w] of Object.entries(weights)) {
      const contribution = (M.terms[term][1] * w) / norm;
      score += contribution;
      words.push({ w: term, weight: contribution });
    }

    const pSpam = 1 / (1 + Math.exp(-score));
    return {
      label: pSpam >= (M.threshold != null ? M.threshold : 0.5) ? "spam" : "ham",
      pSpam,
      words: words.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight)),
      tokens: Object.keys(weights)
    };
  };
})();
