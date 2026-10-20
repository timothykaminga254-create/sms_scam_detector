/* =========================================================
   SMS SCAM DETECTION - Supervised Learning (Naive Bayes)
   Modules:
   1. Dataset
   2. Preprocessing
   3. Naive Bayes model
   4. Train/test split + evaluation
   5. UI controller
   ========================================================= */

/* ---------- 1. DATASET ---------- */
const DATASET = [
  // ---- SCAM (spam) ----
  ["spam", "URGENT! You have won a $1000 cash prize. Call 09061701461 now to claim your reward"],
  ["spam", "Congratulations! You've been selected for a free iPhone. Click http://free-gift.biz to claim"],
  ["spam", "Your bank account has been suspended. Verify your PIN at http://secure-bank-login.com immediately"],
  ["spam", "WINNER!! As a valued customer you have been selected to receive a £900 prize reward. Call now"],
  ["spam", "Dear customer, your ATM card is blocked. Update your details here http://bit.ly/update-card"],
  ["spam", "FREE entry in a weekly competition to win cash. Text WIN to 80086 now. T&Cs apply"],
  ["spam", "You have a package waiting. Pay the delivery fee of $2.99 at http://track-parcel.info to release it"],
  ["spam", "Final notice: your loan of $5000 is approved. Reply YES to receive funds instantly, no credit check"],
  ["spam", "Claim your free ringtone now! Text TONE to 87077. Subscription 3 pounds per week"],
  ["spam", "Security alert: unusual login detected on your account. Confirm your password at http://verify-now.co"],
  ["spam", "Mobile money alert: you received 50,000. Send back the wrong transfer or your account will be closed"],
  ["spam", "Earn $500 daily working from home. No experience needed. Click the link to register today"],
  ["spam", "Your number has won the lottery of 2,000,000. Contact our agent to claim your winning now"],
  ["spam", "URGENT your account will be deactivated in 24 hours. Click here to verify identity immediately"],
  ["spam", "Hot singles in your area want to meet you. Reply now for free chat, 18+ only"],
  ["spam", "Get cheap meds online no prescription needed. Order now and save 80 percent discount"],
  ["spam", "IRS notice: you owe back taxes. Pay immediately by gift card to avoid arrest and legal action"],
  ["spam", "You have been pre-approved for a credit card with a 5000 limit. Apply now at http://easy-credit.xyz"],
  ["spam", "Congratulations you are the lucky winner of a free holiday voucher. Call 0800 now to claim prize"],
  ["spam", "Your Netflix payment failed. Update billing information at http://netflix-billing-update.com to avoid suspension"],
  ["spam", "Last chance! Exclusive offer, win a brand new car. Text CAR to 88600 free entry"],
  ["spam", "Dear user your SIM will be blocked today. Send your PIN and ID number to keep your line active"],
  ["spam", "Investment opportunity: double your bitcoin in 48 hours guaranteed. Send deposit now"],
  ["spam", "Hi, I am a prince and need your help to transfer 10 million. You will receive 30 percent commission"],
  ["spam", "Claim your COVID relief payment of $1200 now. Verify your social security number at the link"],
  ["spam", "Free msg: your mobile has won a bonus cash award. Reply CLAIM to collect your prize today"],
  ["spam", "Alert: your PayPal account is limited. Login at http://paypal-secure-check.com to restore access"],
  ["spam", "You are selected for a government grant of 10000. Pay a small processing fee to receive it"],
  ["spam", "Cheap loans available, bad credit accepted. Call now for instant cash approval"],
  ["spam", "Winner! Your phone number was drawn in our promo. Send your bank details to receive the money"],

  // ---- SAFE (ham) ----
  ["ham", "Hey, are we still meeting for lunch at 1pm today?"],
  ["ham", "Mum, I will be home late tonight, don't wait up for dinner"],
  ["ham", "Can you send me the notes from yesterday's lecture please"],
  ["ham", "Happy birthday! Hope you have a wonderful day with your family"],
  ["ham", "I'll call you when I get out of the meeting, should be around 3"],
  ["ham", "Don't forget to buy milk and bread on your way home"],
  ["ham", "The project deadline has moved to Friday, let's discuss in the morning"],
  ["ham", "Thanks for helping me move yesterday, I owe you one"],
  ["ham", "What time does the football match start tonight?"],
  ["ham", "Ok see you at the station at 6. I will bring the tickets"],
  ["ham", "Sorry I missed your call, I was in class. Call me back later"],
  ["ham", "Dad says the car is fixed and you can pick it up tomorrow"],
  ["ham", "Let's go to the cinema this weekend, there's a new movie out"],
  ["ham", "I am on my way, traffic is bad so I will be about ten minutes late"],
  ["ham", "Please remember to submit your assignment before midnight"],
  ["ham", "Good morning! Hope you slept well. Talk to you later"],
  ["ham", "Can we reschedule our appointment to next Tuesday afternoon?"],
  ["ham", "Your table is booked for four people at 7:30 tonight"],
  ["ham", "I just reached home safely. Thanks for the ride"],
  ["ham", "Did you see the game last night? It was amazing"],
  ["ham", "The doctor said I should rest for a few days, nothing serious"],
  ["ham", "Reminder: team meeting tomorrow at 10am in the conference room"],
  ["ham", "Grandma wants to know if you're coming for Sunday lunch"],
  ["ham", "I left my keys at your place, can I pick them up later today?"],
  ["ham", "Congrats on the new job! Let's celebrate this weekend"],
  ["ham", "Can you help me with the maths homework tonight?"],
  ["ham", "I'm at the shop, do you need anything else from here?"],
  ["ham", "Thanks, I got the money you sent for the rent. Much appreciated"],
  ["ham", "The delivery arrived this morning, I put the package in the kitchen"],
  ["ham", "Let me know when you're free and we can go over the report together"]
];

/* ---------- 2. PREPROCESSING ---------- */
const STOPWORDS = new Set(
  ("a an the and or but if is are was were be been am i me my you your we our they them he she it its " +
   "to of in on at for with as by from this that these those so do does did have has had will would can " +
   "could should just not no yes").split(" ")
);

function tokenize(text) {
  let t = text.toLowerCase();
  t = t.replace(/https?:\/\/\S+|www\.\S+|\b\S+\.(com|biz|xyz|info|co|net)\b\S*/g, " urltoken ");
  t = t.replace(/[$£€]\s?\d[\d,.]*|\d[\d,.]*\s?(usd|dollars|pounds)/g, " moneytoken ");
  t = t.replace(/\b\d{5,}\b/g, " longnumtoken ");
  t = t.replace(/\b\d+\b/g, " numtoken ");
  t = t.replace(/[^a-z\s]/g, " ");
  const tokens = t.split(/\s+/).filter(w => w.length > 1 && !STOPWORDS.has(w));
  if (/[A-Z]{4,}/.test(text)) tokens.push("allcapstoken");   // shouting is a strong feature
  if (/!{2,}/.test(text)) tokens.push("excitetoken");
  return tokens;
}

/* ---------- 3. NAIVE BAYES MODEL ---------- */
class NaiveBayes {
  constructor(alpha = 1) {
    this.alpha = alpha;           // Laplace smoothing
    this.reset();
  }
  reset() {
    this.classCounts = { spam: 0, ham: 0 };
    this.wordCounts = { spam: {}, ham: {} };
    this.totalWords = { spam: 0, ham: 0 };
    this.vocab = new Set();
    this.docs = 0;
  }
  train(data) {
    this.reset();
    for (const [label, text] of data) {
      this.classCounts[label]++;
      this.docs++;
      for (const w of tokenize(text)) {
        this.wordCounts[label][w] = (this.wordCounts[label][w] || 0) + 1;
        this.totalWords[label]++;
        this.vocab.add(w);
      }
    }
  }
  wordLogProb(w, label) {
    const c = this.wordCounts[label][w] || 0;
    return Math.log((c + this.alpha) / (this.totalWords[label] + this.alpha * this.vocab.size));
  }
  predict(text) {
    const tokens = tokenize(text);
    const scores = {};
    for (const label of ["spam", "ham"]) {
      let s = Math.log((this.classCounts[label] + 1) / (this.docs + 2));
      for (const w of tokens) if (this.vocab.has(w)) s += this.wordLogProb(w, label);
      scores[label] = s;
    }
    // convert log scores to probability (softmax)
    const m = Math.max(scores.spam, scores.ham);
    const es = Math.exp(scores.spam - m), eh = Math.exp(scores.ham - m);
    const pSpam = es / (es + eh);
    return {
      label: pSpam >= 0.5 ? "spam" : "ham",
      pSpam,
      tokens: [...new Set(tokens)].filter(w => this.vocab.has(w))
    };
  }
  // log-odds contribution of a word: positive = scam, negative = safe
  wordWeight(w) {
    return this.wordLogProb(w, "spam") - this.wordLogProb(w, "ham");
  }
  topSpamWords(n = 15) {
    return [...this.vocab]
      .filter(w => (this.wordCounts.spam[w] || 0) >= 2)
      .map(w => ({ w, weight: this.wordWeight(w) }))
      .sort((a, b) => b.weight - a.weight)
      .slice(0, n);
  }
}

/* ---------- 4. SPLIT + EVALUATION ---------- */
function seededShuffle(arr, seed = 42) {
  const a = arr.slice();
  let s = seed;
  const rand = () => (s = (s * 1664525 + 1013904223) % 4294967296) / 4294967296;
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function stratifiedSplit(data, testFraction) {
  const spam = seededShuffle(data.filter(d => d[0] === "spam"));
  const ham = seededShuffle(data.filter(d => d[0] === "ham"));
  const cut = arr => Math.max(1, Math.round(arr.length * testFraction));
  const test = [...spam.slice(0, cut(spam)), ...ham.slice(0, cut(ham))];
  const train = [...spam.slice(cut(spam)), ...ham.slice(cut(ham))];
  return { train, test };
}

function evaluate(model, test) {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (const [label, text] of test) {
    const pred = model.predict(text).label;
    if (label === "spam" && pred === "spam") tp++;
    else if (label === "ham" && pred === "ham") tn++;
    else if (label === "ham" && pred === "spam") fp++;
    else fn++;
  }
  const accuracy = (tp + tn) / test.length;
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
  return { tp, tn, fp, fn, accuracy, precision, recall, f1 };
}

/* ---------- 5. UI CONTROLLER ---------- */
const $ = id => document.getElementById(id);
const model = new NaiveBayes();   // used for evaluation (trained on train split)
const finalModel = new NaiveBayes(); // used for live predictions (trained on ALL data)
let lastMessage = "";

function pct(x) { return (x * 100).toFixed(1) + "%"; }

function retrain() {
  const testFraction = Number($("splitRange").value) / 100;
  const { train, test } = stratifiedSplit(DATASET, testFraction);
  model.train(train);
  const r = evaluate(model, test);
  finalModel.train(DATASET);

  $("metrics").innerHTML = [
    ["Accuracy", pct(r.accuracy)],
    ["Precision", pct(r.precision)],
    ["Recall", pct(r.recall)],
    ["F1 score", pct(r.f1)]
  ].map(([n, v]) => `<div class="metric"><b>${v}</b><span>${n}</span></div>`).join("");

  $("confusion").querySelector("tbody").innerHTML = `
    <tr><td>Actually Safe</td><td>${r.tn}</td><td>${r.fp}</td></tr>
    <tr><td>Actually Scam</td><td>${r.fn}</td><td>${r.tp}</td></tr>`;

  const nSpam = DATASET.filter(d => d[0] === "spam").length;
  $("datasetInfo").textContent =
    `Dataset: ${DATASET.length} messages (${nSpam} scam, ${DATASET.length - nSpam} safe). ` +
    `Evaluated on ${test.length} held-out messages; live predictions use a model trained on all data. ` +
    `Vocabulary: ${finalModel.vocab.size} words.`;

  renderSpamWords();
}

function renderSpamWords() {
  $("spamWords").innerHTML = finalModel.topSpamWords(18)
    .map(x => `<span class="chip spam">${x.w}</span>`).join("");
}

function analyze() {
  const text = $("smsInput").value.trim();
  if (!text) return;
  lastMessage = text;
  // Use the model trained by train_model.py when model.js is present
  const r = (window.predictTrained && window.predictTrained(text)) || finalModel.predict(text);

  const v = $("verdict");
  v.textContent = r.label === "spam" ? "⚠️ Likely SCAM" : "✅ Looks SAFE";
  v.className = "verdict " + r.label;
  $("barFill").style.width = (r.pSpam * 100).toFixed(1) + "%";
  $("probText").textContent = `Scam probability: ${pct(r.pSpam)}`;

  const words = (r.words || r.tokens
    .map(w => ({ w, weight: finalModel.wordWeight(w) }))
    .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight)))
    .slice(0, 10);
  $("topWords").innerHTML = words.length
    ? words.map(x => `<span class="chip ${x.weight > 0 ? "spam" : "ham"}">${x.w}</span>`).join("")
    : '<span class="muted">No known words found in this message.</span>';

  $("result").classList.remove("hidden");
}

function addSample(label, text) {
  if (!text.trim()) return;
  DATASET.push([label, text.trim()]);
  retrain();
  if ($("smsInput").value.trim()) analyze();
}

$("checkBtn").addEventListener("click", analyze);
$("clearBtn").addEventListener("click", () => {
  $("smsInput").value = "";
  $("result").classList.add("hidden");
});
$("examples").addEventListener("change", e => {
  if (e.target.value) { $("smsInput").value = e.target.value; analyze(); }
  e.target.selectedIndex = 0;
});
$("splitRange").addEventListener("input", e => { $("splitVal").textContent = e.target.value; });
$("splitRange").addEventListener("change", retrain);
$("retrainBtn").addEventListener("click", retrain);
$("markSpam").addEventListener("click", () => addSample("spam", lastMessage));
$("markHam").addEventListener("click", () => addSample("ham", lastMessage));
$("addBtn").addEventListener("click", () => {
  addSample($("newLabel").value, $("newText").value);
  $("newText").value = "";
});
$("smsInput").addEventListener("keydown", e => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) analyze();
});

retrain();
