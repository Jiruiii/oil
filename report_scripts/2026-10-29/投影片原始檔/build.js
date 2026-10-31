// 10/29 weekly deck: background-inclusive Oil IoU, Boundary IoU, R ring8, look-alike analysis.
const pptxgen = require("pptxgenjs");
// apply_theme.js comes from the pptx skill used to build the deck; set APPLY_THEME to its path.
const { applyTheme } = require(process.env.APPLY_THEME);

const OUT = process.argv[2];
const IMG = require("path").resolve(__dirname, "../../../../reports/2026-10-29/投影片原始檔/img") + "/";   // images stay outside git
const THEME = {
  name: "Oil Weekly",
  headFontFace: "Microsoft JhengHei",
  bodyFontFace: "Microsoft JhengHei",
  colors: {
    dk1: "14213D", lt1: "FFFFFF", dk2: "2B3445", lt2: "F7F6F2",
    accent1: "A3561A", accent2: "2F6FB0", accent3: "E3A869", accent4: "5B6472",
    accent5: "ECEFF3", accent6: "F4E6DA", hlink: "2F6FB0", folHlink: "5B6472",
  },
};
const HX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "週進度報告 10/29：HNM 與 look-alike";
const C = pres.SchemeColor;
const NAVY = C.text1, INK = C.text2, PAPER = C.background2, WHITE = C.background1;
const TERRA = C.accent1, BLUE = C.accent2, AMBER = C.accent3, SLATE = C.accent4;
const GREY = C.accent5, PEACH = C.accent6;

const W = 13.333, M = 0.6;
const FOOT = "單一 seed 42 · 三折平均 · [95% CI]＝2,000 次場景配對 bootstrap";

pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: NAVY },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: M + 0.4, y: 1.3, w: 9, h: 0.5,
        fontSize: 18, color: AMBER, bold: true, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: M + 0.4, y: 1.9, w: 11.5, h: 2.0,
        fontSize: 44, bold: true, color: PAPER, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: PAPER },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: M, y: 0.35, w: 9, h: 0.4,
        fontSize: 14, color: TERRA, bold: true, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.75, w: W - 2 * M, h: 0.8,
        fontSize: 32, bold: true, color: NAVY, valign: "top", align: "left", margin: 0 }, text: "" } },
    { text: { text: FOOT, options: { x: M, y: 7.0, w: 9, h: 0.3, fontSize: 10, color: SLATE, margin: 0 } } },
  ],
  slideNumber: { x: W - M - 0.8, y: 7.0, w: 0.8, h: 0.3, fontSize: 10, color: SLATE, align: "right", margin: 0 },
});
pres.defineSlideMaster({
  title: "CLOSING_DARK",
  background: { color: NAVY },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: M, y: 0.35, w: 9, h: 0.4,
        fontSize: 14, color: AMBER, bold: true, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.75, w: W - 2 * M, h: 0.8,
        fontSize: 32, bold: true, color: PAPER, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: W - M - 0.8, y: 7.0, w: 0.8, h: 0.3, fontSize: 10, color: SLATE, align: "right", margin: 0 },
});

let section = null;
function slide(master, kicker, title, notes, sectionTitle) {
  if (sectionTitle && sectionTitle !== section) { pres.addSection({ title: sectionTitle }); section = sectionTitle; }
  const s = pres.addSlide({ masterName: master, sectionTitle: section });
  s.addText(kicker, { placeholder: "kicker" });
  s.addText(title, { placeholder: "title" });
  s.addNotes(notes);
  return s;
}
const txt = (s, text, o) => s.addText(text, Object.assign({ isTextBox: true, margin: 0, color: INK, fontSize: 16, valign: "top" }, o));
const card = (s, x, y, w, h, fill, name) =>
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { type: "none" }, objectName: name });
const bullets = (items, o = {}) => items.map((t, i) => ({
  text: t, options: Object.assign({ bullet: { indent: 18 }, paraSpaceAfter: 8, breakLine: i < items.length - 1 }, o) }));

// table helpers
const th = (t, o = {}) => ({ text: t, options: Object.assign({ bold: true, color: WHITE, fill: { color: NAVY }, align: "center", valign: "middle" }, o) });
const td = (t, o = {}) => ({ text: t, options: Object.assign({ color: INK, valign: "middle" }, o) });
const TBL = { fontSize: 14, border: { type: "solid", pt: 0.75, color: "D5D9DF" }, margin: [0.04, 0.08, 0.04, 0.08] };

const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
  legendFontFace: "+mn-lt", titleFontFace: "+mn-lt", catAxisLabelColor: HX.dk2, valAxisLabelColor: HX.accent4,
  dataLabelColor: HX.dk2, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, dataLabelFontSize: 11, legendFontSize: 12,
  valGridLine: { color: "D5D9DF", size: 0.75 }, catGridLine: { style: "none" }, catAxisLineShow: false, valAxisLineShow: false };

const ARMS = ["A 基準", "B 隨機（同池）", "C HNM-A", "D HNM-A+SI", "E HNM-EffNet", "F HNM-EffNet+SI", "R ring8"];

// ---------------------------------------------------------------- 1 cover
{
  pres.addSection({ title: "總覽" }); section = "總覽";
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: section });
  s.addText("週進度報告 · 2026-10-29", { placeholder: "kicker" });
  s.addText("評估指標、邊界實驗\n與 HNM 漏報油汙分析", { placeholder: "title" });
  const rows = [["評估指標", "含背景 Oil IoU、Boundary IoU"], ["邊界實驗", "R ring8：訓練時忽略油汙外圈 8 px"], ["look-alike", "相同 recall 下的背景誤報；HNM 漏報油汙的特性"]];
  rows.forEach(([k, v], i) => {
    s.addShape(pres.shapes.OVAL, { x: M + 0.4, y: 4.32 + i * 0.6, w: 0.16, h: 0.16, fill: { color: AMBER }, line: { type: "none" }, objectName: `cover-dot-${i}` });
    txt(s, [{ text: k + "　", options: { bold: true, color: AMBER } }, { text: v, options: { color: "C9D2E0" } }],
      { x: M + 0.75, y: 4.2 + i * 0.6, w: 10, h: 0.4, fontSize: 20, objectName: `cover-item-${i}` });
  });
  txt(s, "Prithvi-EO-2.0-300M + DeepLabV3 · GT-expand 三折 · 單一 seed", { x: M + 0.4, y: 6.6, w: 9, h: 0.35, fontSize: 14, color: "9AA8BF", objectName: "cover-foot" });
  s.addNotes("這週的進度分三個部分。第一是評估指標，主指標加入背景小圖的誤報，另外加了一個只看邊界的 Boundary IoU。第二是邊界的實驗 ring8。第三是跟 look-alike 有關的分析：在 recall 相同的條件下比較背景誤報，然後看 HNM 漏掉的油汙有什麼特性。");
}

// ---------------------------------------------------------------- 2 arms (table layout, 10/29)
{
  const s = slide("CONTENT", "實驗組別", "七組實驗的設定",
    "先說明後面會一直出現的七組。表上每一列是一組，每一欄是可能不一樣的地方，橫的看就知道每組改了什麼；沒有改的地方是一槓。所有組的油汙小圖、val 和 test、模型和訓練設定都一樣，每折訓練背景也都是 1,860、1,860、1,870 張。先看上面兩列：A 是原本的基準，背景是原名冊裡每個場景隨機抽 10 張；R 是這週的邊界實驗，背景跟 A 一樣，只在訓練時讓油汙外圈 8 像素的未標註像素不計分，所以 R 跟 A 比。下面五組用另一個候選池，在訓練場景上滑窗取小圖，要求確認背景至少 50%、未標註不超過 30%。B 從這個池子隨機抽；C 到 F 四組都是 HNM，從同一個池子只挑被判成油的背景小圖，也就是硬負樣本。四組只差兩件事：一是由誰判斷，C、D 用基準 A 自己的誤報，A 看過這些訓練場景；E、F 用 EfficientNet 分類器，每個場景都由沒看過它的分類器判斷。二是要不要再用三個光譜指數排除最像油的 5%，有的話名稱加 +SI，也就是 D 和 F，這是學長 HNM_SI 的做法。最後一欄是比較對象：B 到 F 用同一個池子、同樣張數、同樣依場景平均抽，只差挑法，所以 HNM 各組都跟 B 比，加 SI 的組另外跟沒加的比；A 的背景池不同，不直接拿來比。",
    "總覽");
  txt(s, "每一列是一組、每一欄是可能不同的地方；「—」＝沒有改，同欄相同的格子合併。",
    { x: M, y: 1.48, w: W - 2 * M, h: 0.35, fontSize: 15, color: SLATE, objectName: "arms-sub" });
  const NONE = "—";
  const name = (l, n, col) => ({ text: [{ text: l + "  ", options: { bold: true, color: col, fontSize: 20 } }, { text: n, options: { bold: true, color: NAVY } }],
    options: { valign: "middle", fill: { color: WHITE } } });
  const v = (t, o = {}) => td(t, Object.assign({ align: "center", fill: { color: WHITE } }, o));
  const no = (o = {}) => v(NONE, Object.assign({ color: "B4BBC6" }, o));
  const hnm = (t, o = {}) => v(t, Object.assign({ fill: { color: PEACH }, bold: true, color: NAVY }, o));
  const cmp = (t) => v(t, { bold: true, color: TERRA });
  const rows = [
    [th("組別"), th("背景候選池"), th("背景怎麼挑"), th("由誰判斷像油"), th("光譜篩選 SI"), th("訓練 loss"), th("和誰比")],
    [name("A", "基準", SLATE), v("原背景名冊", { rowspan: 2 }), v("每場景隨機 10 張", { rowspan: 2 }), no(), no(), no(), no()],
    [name("R", "ring8", SLATE), no(), no(), hnm("油汙外圈 8 px\n未標註不計分", { fontSize: 12 }), cmp("A")],
    [name("B", "隨機（同池）", SLATE), v("滑窗候選池\n確認背景 ≥ 50%\n未標註 ≤ 30%", { rowspan: 5, fontSize: 13 }), v("每場景隨機"), no(), no(), no(), cmp("對照組")],
    [name("C", "HNM-A", TERRA), hnm("只挑被判成油的\n（硬負樣本）", { rowspan: 4 }), hnm("基準 A\n（看過訓練場景）", { rowspan: 2, fontSize: 13 }), no(), no(), cmp("B")],
    [name("D", "HNM-A+SI", TERRA), hnm("排除最像油\n的 5%", { fontSize: 12 }), no(), cmp("B、C")],
    [name("E", "HNM-EffNet", BLUE), hnm("EfficientNet\n（沒看過該場景）", { rowspan: 2, fontSize: 13 }), no(), no(), cmp("B")],
    [name("F", "HNM-EffNet+SI", BLUE), hnm("排除最像油\n的 5%", { fontSize: 12 }), no(), cmp("B、E")],
  ];
  s.addTable(rows, Object.assign({}, TBL, { x: M, y: 1.9, w: W - 2 * M, colW: [2.3, 1.85, 1.95, 1.95, 1.4, 1.5, 1.18],
    rowH: [0.42, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48], objectName: "arms-table" }));
  card(s, M, 6.3, W - 2 * M, 0.62, WHITE, "arms-note-card");
  txt(s, [{ text: "固定不變：", options: { bold: true, color: TERRA } },
          { text: "油汙小圖、val／test、模型與訓練設定、每折背景張數（1,860／1,860／1,870）。B～F 同池同張數，HNM 各組都和 B 比；A 的背景池不同，不直接比。SI＝MNDWI、NDI、SR 三個光譜指數（學長 HNM_SI）。" }],
    { x: M + 0.3, y: 6.33, w: W - 2 * M - 0.6, h: 0.56, fontSize: 13, valign: "middle", objectName: "arms-note" });
}

// ---------------------------------------------------------------- 3 summary
{
  const s = slide("CONTENT", "本週結果摘要", "HNM 降低背景誤報，但會漏掉部分弱對比油汙",
    "先講這週的結果。第一，主指標加入背景小圖的誤報之後，C 跟 B 的差異變得不顯著，D、E、F 還是顯著比 B 低，整體排序跟上週一樣。第二，ring8 會讓模型在油汙外圍多預測一圈，Boundary IoU 下降 0.134；標註是 NOAA 的專家標註，多預測的部分就是誤報，所以這個設定不採用。第三，在 recall 相同的條件下比較，C 在 test 背景小圖上的警報率是 2.5%，A 是 4.8%，所以 HNM 的誤報下降不是單純因為模型整體變保守。第四，HNM 會漏掉一部分對比比較弱的油汙，這些油汙在模型特徵上比較接近 HNM 選出的硬負樣本，用近鄰裡硬負樣本的比例去預測哪些油汙會被漏掉，AUC 是 0.78。");
  const cards = [
    ["−0.019", "C − B 含背景 Oil IoU", "加入背景誤報後差異不顯著；D、E、F 仍低於 B", BLUE],
    ["−0.134", "R − A Boundary IoU", "ring8 在油汙外圍多預測一圈，precision 下降，不採用", TERRA],
    ["2.5%", "相同 recall 下 C 的背景警報率", "A 為 4.8%、B 為 5.2%", BLUE],
    ["0.78", "以特徵近鄰預測漏報的 AUC", "C 漏掉的油汙在特徵上較接近硬負樣本", TERRA],
  ];
  const gap = 0.3, cw = (W - 2 * M - 3 * gap) / 4;
  cards.forEach(([big, label, desc, col], i) => {
    const x = M + i * (cw + gap);
    card(s, x, 2.0, cw, 4.0, WHITE, `sum-card-${i}`);
    txt(s, big, { x: x + 0.3, y: 2.3, w: cw - 0.6, h: 1.0, fontSize: 48, bold: true, color: col, objectName: `sum-big-${i}` });
    txt(s, label, { x: x + 0.3, y: 3.45, w: cw - 0.6, h: 0.9, fontSize: 17, bold: true, color: NAVY, objectName: `sum-label-${i}` });
    txt(s, desc, { x: x + 0.3, y: 4.45, w: cw - 0.6, h: 1.4, fontSize: 14, color: INK, objectName: `sum-desc-${i}` });
  });
}

// ---------------------------------------------------------------- 4 recap
{
  const s = slide("CONTENT", "上週回顧", "上週結果與本週工作",
    "上週的結果有三點：C 到 F 四種背景選取方法的 Oil IoU 都比同池隨機背景 B 低；HNM 讓 test 背景小圖的誤報大約減半，但主指標只評油汙周邊，所以反映不出來；主指標的誤報有 88% 在油汙邊界附近。根據上週的討論，這週先處理兩件事：一是評估要不要納入背景誤報，在第 5、6 頁；二是油汙邊界的問題，在第 7 到 10 頁。第 11 到 15 頁是跟論文題目 look-alike 有關的分析。",
    "總覽");
  txt(s, "上週（10/22）結果", { x: M, y: 1.9, w: 5.6, h: 0.4, fontSize: 20, bold: true, color: NAVY, objectName: "recap-left-h" });
  txt(s, bullets([
    "C～F 四種背景選取方法的 Oil IoU 都低於同池隨機背景 B（−0.026～−0.053，CI 不含 0）",
    "HNM 讓 test 背景小圖的誤報約減半，但主指標只評油汙周邊，反映不出來",
    "主指標的誤報有 88% 在油汙邊界附近；油旁未標註區的誤判率是確認背景的 3.8 倍",
  ]), { x: M, y: 2.5, w: 5.6, h: 3.6, fontSize: 16, objectName: "recap-left" });
  const items = [["1", "評估是否納入背景誤報", "含背景 Oil IoU（第 5–6 頁）"],
                 ["2", "油汙邊界與漏報", "Boundary IoU、R ring8 實驗（第 7–10 頁）"],
                 ["3", "look-alike 誤報分析", "門檻掃描、漏報油汙分析（第 11–15 頁）"]];
  items.forEach(([n, q, a], i) => {
    const y = 1.9 + i * 1.62;
    card(s, 7.0, y, W - M - 7.0, 1.45, WHITE, `recap-card-${i}`);
    s.addShape(pres.shapes.OVAL, { x: 7.3, y: y + 0.4, w: 0.65, h: 0.65, fill: { color: i < 2 ? TERRA : BLUE }, line: { type: "none" }, objectName: `recap-num-bg-${i}` });
    txt(s, n, { x: 7.3, y: y + 0.4, w: 0.65, h: 0.65, fontSize: 22, bold: true, color: WHITE, align: "center", valign: "middle", objectName: `recap-num-${i}` });
    txt(s, q, { x: 8.2, y: y + 0.25, w: W - M - 8.45, h: 0.5, fontSize: 19, bold: true, color: NAVY, objectName: `recap-q-${i}` });
    txt(s, a, { x: 8.2, y: y + 0.8, w: W - M - 8.45, h: 0.5, fontSize: 14, color: INK, objectName: `recap-a-${i}` });
  });
}

// ---------------------------------------------------------------- 5 metric 1 definition
{
  const s = slide("CONTENT", "新指標一", "含背景 Oil IoU：把背景小圖的誤報加入分母",
    "新指標的定義很簡單：原本 Oil IoU 的分母是抓對的油、漏掉的油、油汙周邊的誤報；現在再加上 test 背景小圖上的誤報。背景小圖是每折凍結名冊裡屬於 test 場景的那 1,170 張，大約 7,668 萬像素，都不含油、以確認背景為主，所以在上面判成油的像素全部算誤報。為什麼不直接整景推論？因為整景大部分是未標註區，分數會被未標註區主導；背景小圖是最乾淨的背景。原主指標保留，新指標並列。驗證方面，fold 1 基準算出來和上週診斷完全一樣；ring8 訓練時自動算的值也和事後診斷工具算的一致。",
    "新指標一：含背景 Oil IoU");
  card(s, M, 1.9, W - 2 * M, 1.5, NAVY, "def1-formula-card");
  txt(s, [{ text: "含背景 Oil IoU ＝ TP ÷ ( TP ＋ FN ＋ FP", options: {} }, { text: "油汙周邊", options: { fontSize: 16 } },
          { text: " ＋ ", options: {} }, { text: "FP", options: { color: AMBER } }, { text: "背景小圖", options: { color: AMBER, fontSize: 16 } }, { text: " )", options: {} }],
    { x: M + 0.4, y: 2.15, w: W - 2 * M - 0.8, h: 0.6, fontSize: 26, bold: true, color: PAPER, objectName: "def1-formula" });
  txt(s, "原主指標 ＝ TP ÷ (TP ＋ FN ＋ FP油汙周邊)，保留不變；新指標並列", { x: M + 0.4, y: 2.85, w: 10, h: 0.4, fontSize: 16, color: "C9D2E0", objectName: "def1-orig" });
  const cols = [
    ["油汙周邊小圖", "原主指標的範圍（GT-aware 重組）", ["TP 抓對的油、FN 漏掉的油", "FP 油汙周邊的誤報", "未標註區當背景計分"], BLUE],
    ["test 背景小圖", "每折 1,170 張 · 約 7,668 萬像素", ["凍結名冊中屬於 test 場景者，不含油", "同模型、同 TTA 推論", "判成油的像素全部算誤報"], AMBER],
    ["驗證", "兩條計算路徑結果一致", ["fold 1 基準：1,170 張、171,638 誤報像素，與上週診斷相同", "R ring8：訓練流程自動評估＝事後診斷（三折相同）"], SLATE],
  ];
  const cw = (W - 2 * M - 2 * 0.3) / 3;
  cols.forEach(([h, sub, items, col], i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, 3.7, cw, 3.05, WHITE, `def1-card-${i}`);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 3.98, w: 0.22, h: 0.22, fill: { color: col }, line: { type: "none" }, objectName: `def1-dot-${i}` });
    txt(s, h, { x: x + 0.65, y: 3.88, w: cw - 0.9, h: 0.45, fontSize: 18, bold: true, color: NAVY, objectName: `def1-h-${i}` });
    txt(s, sub, { x: x + 0.3, y: 4.4, w: cw - 0.6, h: 0.4, fontSize: 14, color: SLATE, objectName: `def1-sub-${i}` });
    txt(s, bullets(items), { x: x + 0.3, y: 4.9, w: cw - 0.6, h: 1.8, fontSize: 15, objectName: `def1-list-${i}` });
  });
}

// ---------------------------------------------------------------- 6 metric 1 results
{
  const s = slide("CONTENT", "新指標一　結果", "加入背景誤報後，只有 C − B 變為不顯著",
    "左圖每組兩根柱子，深色是原主指標，淺色是含背景版本。每組都往下掉 0.01 到 0.02，掉最少的是背景誤報最少的 D。右表是對 B 的配對差：C 從負 0.026 顯著，變成負 0.019、CI 跨過 0；D、E、F 的差距縮小，但 CI 仍不含 0。原因是背景誤報只占含背景聯集的 2 到 5%，D 只有 1.9%，就算 HNM 把它壓到 0 也扣不了多少；D 的 recall 比 B 少 0.044，影響大得多。所以 HNM 減少背景誤報的效果有反映出來，但幅度不足以抵消 recall 的下降。",
    "新指標一：含背景 Oil IoU");
  s.addChart(pres.charts.BAR, [
    { name: "Oil IoU（原）", labels: ARMS.map(a => a.split(" ")[0]), values: [0.4332, 0.4469, 0.4205, 0.3993, 0.4015, 0.3935, 0.4129] },
    { name: "含背景 Oil IoU", labels: ARMS.map(a => a.split(" ")[0]), values: [0.4145, 0.4255, 0.4069, 0.3893, 0.3865, 0.3822, 0.3905] },
  ], Object.assign({ x: M, y: 1.8, w: 6.4, h: 4.9, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
    chartColors: [HX.dk1, HX.accent3], valAxisMinVal: 0.3, valAxisMaxVal: 0.47, valAxisMajorUnit: 0.05, valAxisLabelFormatCode: "0.00",
    showValue: false, showLegend: true, legendPos: "t", showTitle: false, objectName: "res1-chart" }, chartText));
  const P = (t, sig) => td(t, { align: "center", bold: sig, color: sig ? TERRA : INK });
  s.addTable([
    [th("對 B 的配對差"), th("原 Oil IoU"), th("含背景 Oil IoU")],
    [td("C HNM-A"), P("−0.026\n[−0.048, −0.006]", true), td("−0.019\n[−0.041, +0.005]", { align: "center", bold: true, color: BLUE, fill: { color: "E3ECF6" } })],
    [td("D HNM-A+SI"), P("−0.048\n[−0.072, −0.025]", true), P("−0.036\n[−0.060, −0.012]", true)],
    [td("E HNM-EffNet"), P("−0.045\n[−0.068, −0.024]", true), P("−0.039\n[−0.061, −0.017]", true)],
    [td("F HNM-EffNet+SI"), P("−0.053\n[−0.081, −0.029]", true), P("−0.043\n[−0.069, −0.018]", true)],
  ], Object.assign({ x: 7.3, y: 1.9, w: W - M - 7.3, colW: [1.95, 1.7, 1.78], objectName: "res1-table" }, TBL));
  card(s, 7.3, 5.25, W - M - 7.3, 1.45, PEACH, "res1-callout");
  txt(s, [{ text: "背景誤報只占含背景聯集 2～5%", options: { bold: true, color: NAVY, breakLine: true } },
          { text: "D 只有 1.9%，影響小於 recall 的下降（−0.044）", options: {} }],
    { x: 7.55, y: 5.4, w: W - M - 7.8, h: 1.2, fontSize: 15, objectName: "res1-callout-text" });
}

// ---------------------------------------------------------------- 7 metric 2 definition
{
  const s = slide("CONTENT", "新指標二", "Boundary IoU：只比較油汙邊緣那一圈",
    "第二個指標參考 Boundary IoU 這篇論文實作。做法是：GT 和預測各自取「油汙內、離輪廓 8 像素以內」的一圈，再算這兩圈的 IoU。左圖是一個例子：預測只比 GT 稍微大一點、偏一點，一般的 mask IoU 還有 0.83，但兩圈邊界帶重疊很少，Boundary IoU 只剩 0.26。也就是說，它對大油汙的邊界誤差比 mask IoU 敏感。寬度 d 取 8 像素：論文用影像對角線的 2%，256 小圖大約是 7 像素，取 8 跟上週診斷的 near_oil 一致。我們只推論油汙周邊的小圖，所以多做一個調整：沒推論的區域當成未知，不算輪廓，避免小圖邊緣被當成邊界。",
    "新指標二：Boundary IoU");
  const iw = 3.3, ih = iw * 420 / 560;
  s.addImage({ path: IMG + "biou_masks.png", x: M, y: 2.0, w: iw, h: ih, altText: "GT 油汙（藍）與預測油汙輪廓（棕）只差一點位移", objectName: "def2-img-masks" });
  s.addImage({ path: IMG + "biou_bands.png", x: M + iw + 0.3, y: 2.0, w: iw, h: ih, altText: "兩者的內側邊界帶：藍為只在 GT、棕為只在預測、深藍為重疊", objectName: "def2-img-bands" });
  txt(s, "整塊比：mask IoU 0.83", { x: M, y: 2.1 + ih, w: iw, h: 0.4, fontSize: 16, bold: true, color: NAVY, align: "center", objectName: "def2-cap-1" });
  txt(s, "邊界帶：Boundary IoU 0.26", { x: M + iw + 0.3, y: 2.1 + ih, w: iw, h: 0.4, fontSize: 16, bold: true, color: TERRA, align: "center", objectName: "def2-cap-2" });
  const legend = [["只在 GT 邊界帶", BLUE], ["只在預測邊界帶", TERRA], ["重疊", NAVY]];
  legend.forEach(([t, col], i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: M + i * 2.2, y: 2.75 + ih, w: 0.2, h: 0.2, fill: { color: col }, line: { type: "none" }, objectName: `def2-leg-sw-${i}` });
    txt(s, t, { x: M + 0.3 + i * 2.2, y: 2.7 + ih, w: 1.85, h: 0.3, fontSize: 12, color: SLATE, objectName: `def2-leg-${i}` });
  });
  const x0 = 7.7;
  card(s, x0, 1.9, W - M - x0, 4.85, WHITE, "def2-card");
  txt(s, [
    { text: "定義", options: { bold: true, color: NAVY, fontSize: 18, breakLine: true } },
    { text: "G_d、P_d：GT／預測油汙中，距自身輪廓 ≤ d 像素的內側邊界帶", options: { breakLine: true } },
    { text: "Boundary IoU ＝ |G_d ∩ P_d| ÷ |G_d ∪ P_d|，每折所有 test 場景合併", options: { breakLine: true } },
    { text: " ", options: { fontSize: 8, breakLine: true } },
    { text: "d ＝ 8 px", options: { bold: true, color: NAVY, fontSize: 18, breakLine: true } },
    { text: "論文取影像對角線 2%；256 小圖約 7 px，取 8 與上週 near_oil 一致", options: { breakLine: true } },
    { text: " ", options: { fontSize: 8, breakLine: true } },
    { text: "本專案的調整", options: { bold: true, color: NAVY, fontSize: 18, breakLine: true } },
    { text: "只推論油汙周邊小圖：未推論區視為未知、不當輪廓，小圖邊緣不會被算成邊界", options: {} },
  ], { x: x0 + 0.35, y: 2.15, w: W - M - x0 - 0.7, h: 4.0, fontSize: 15, paraSpaceAfter: 4, objectName: "def2-text" });
  txt(s, "Cheng, Girshick, Dollár, Berg, Kirillov. Boundary IoU. CVPR 2021 (arXiv:2103.16562)",
    { x: x0 + 0.35, y: 6.25, w: W - M - x0 - 0.7, h: 0.4, fontSize: 10, color: SLATE, objectName: "def2-cite" });
}

// ---------------------------------------------------------------- 8 metric 2 results
{
  const s = slide("CONTENT", "新指標二　結果", "Boundary IoU：B 最高，各組都偏低",
    "這是七組的 Boundary IoU。所有組都只有 0.16 到 0.34，遠低於 Oil IoU 的 0.39 到 0.45，所以邊界是模型目前最弱的地方，這跟上週誤報 88% 在邊界一致。B 最高，0.335，比 A 高 0.042，CI 不含 0，而且三折方向一致；B 的 precision 也是最高的，代表隨機背景讓模型邊界畫得比較收斂。HNM 四組都低於 B。R ring8 最低，原因下一節講。上週備用頁的正負 8 像素容忍帶是「把邊界排除」，Boundary IoU 是「只看邊界」，兩者互補。",
    "新指標二：Boundary IoU");
  const vals = [0.2929, 0.3349, 0.2861, 0.2885, 0.2452, 0.2715, 0.1585];
  s.addChart(pres.charts.BAR, [{ name: "Boundary IoU", labels: ARMS, values: vals }],
    Object.assign({ x: M, y: 1.8, w: 7.4, h: 4.95, barDir: "bar", barGapWidthPct: 45,
      chartColors: [HX.accent4, HX.accent2, HX.accent4, HX.accent4, HX.accent4, HX.accent4, HX.accent1],
      catAxisOrientation: "maxMin", valAxisMinVal: 0, valAxisMaxVal: 0.4, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0",
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", showLegend: false, showTitle: false, objectName: "res2-chart" }, chartText));
  const x0 = 8.4, w = W - M - x0;
  const facts = [
    ["0.16～0.34", "七組範圍；Oil IoU 為 0.39～0.45", NAVY],
    ["+0.042", "B − A [+0.032, +0.053]，三折方向一致", BLUE],
    ["−0.047～−0.090", "C～F − B，CI 皆不含 0", SLATE],
  ];
  facts.forEach(([big, d, col], i) => {
    const y = 1.9 + i * 1.65;
    card(s, x0, y, w, 1.45, WHITE, `res2-card-${i}`);
    txt(s, big, { x: x0 + 0.3, y: y + 0.15, w: w - 0.6, h: 0.6, fontSize: 28, bold: true, color: col, objectName: `res2-big-${i}` });
    txt(s, d, { x: x0 + 0.3, y: y + 0.8, w: w - 0.6, h: 0.6, fontSize: 14, color: INK, objectName: `res2-desc-${i}` });
  });
}

// ---------------------------------------------------------------- 9 ring8 method
{
  const s = slide("CONTENT", "實驗　R ring8", "R ring8：訓練時，油汙外圈 8 px 的未標註區不計分",
    "接著是邊界的實驗。上週診斷看到，主指標的誤報 55.6% 在油汙 8 像素內，而且油旁未標註區被判成油的比例是確認背景的 3.8 倍。我的假設是：這一圈對模型來說比較難學，直接當背景訓練，可能讓模型在邊緣比較保守而漏掉油。所以 R 只改一件事：訓練時，油汙外圈 8 像素內、原本沒標註的像素，loss 不計分。手標背景、船、遠處未標註都照舊當背景，val 和 test 的評分方式完全不變，所以可以直接跟 A 比。開跑前檢查過，fold 1 同樣 40 張訓練小圖，油像素數不變，多出約 21.6 萬個不計分像素。",
    "實驗：R ring8");
  const iw = 5.2, ih = iw * 420 / 640;
  s.addImage({ path: IMG + "ring_diagram.png", x: M, y: 1.95, w: iw, h: ih, altText: "示意：深藍油汙、外圈斜線為不計分區、外側為當背景的未標註區、右側框為手標背景", objectName: "ring-img" });
  const leg = [["油汙（標註 0）→ 油", NAVY], ["外圈 8 px 未標註 → 不計分（只改這裡）", "B8BEC8"], ["遠處未標註（255）→ 背景，照舊", "E9E4DD"], ["手標背景框（1）、船（2）→ 背景，照舊", "D8C6B3"]];
  leg.forEach(([t, col], i) => {
    const y = 2.1 + ih + i * 0.36;
    s.addShape(pres.shapes.RECTANGLE, { x: M, y: y + 0.04, w: 0.24, h: 0.24, fill: { color: col }, line: { color: "5B6472", width: 0.75 }, objectName: `ring-leg-sw-${i}` });
    txt(s, t, { x: M + 0.4, y, w: iw - 0.4, h: 0.32, fontSize: 14, objectName: `ring-leg-${i}` });
  });
  const x0 = 7.1, w = W - M - x0;
  card(s, x0, 1.95, w, 4.8, WHITE, "ring-card");
  txt(s, [
    { text: "動機（上週診斷）", options: { bold: true, color: NAVY, fontSize: 18, breakLine: true } },
    { text: "主指標誤報 55.6% 在油汙 8 px 內", options: { bullet: { indent: 18 }, breakLine: true } },
    { text: "油旁未標註的誤判率是確認背景的 3.8 倍", options: { bullet: { indent: 18 }, breakLine: true } },
    { text: "假設：這一圈較難學，可能使模型在邊緣較保守而漏報", options: { bullet: { indent: 18 }, breakLine: true } },
    { text: " ", options: { fontSize: 8, breakLine: true } },
    { text: "控制條件", options: { bold: true, color: NAVY, fontSize: 18, breakLine: true } },
    { text: "其他設定與 A 完全相同（背景名冊、超參數、seed 42）", options: { bullet: { indent: 18 }, breakLine: true } },
    { text: "val／test 評分方式不變，可直接和 A 比較", options: { bullet: { indent: 18 }, breakLine: true } },
    { text: "開跑前檢查：fold 1 同 40 張小圖，油像素不變（652,486），多 216,189 個不計分像素", options: { bullet: { indent: 18 } } },
  ], { x: x0 + 0.35, y: 2.2, w: w - 0.7, h: 4.4, fontSize: 15, paraSpaceAfter: 6, objectName: "ring-text" });
}

// ---------------------------------------------------------------- 10 ring8 results
{
  const s = slide("CONTENT", "實驗　R ring8　結果", "R ring8：recall 上升，precision 與 Boundary IoU 下降",
    "這是 R 跟 A 的比較。漏報有減少，recall 從 0.566 升到 0.605，CI 不含 0。但誤報增加更多：precision 從 0.653 掉到 0.564，油汙 8 像素內的誤報每折從大約 60 萬增加到 106 萬像素，占全部誤報的比例從 55.6% 變 62.4%。結果主指標 Oil IoU 少 0.020，CI 剛好含 0，含背景版本少 0.024，CI 上界剛好碰到 0；最明顯的是 Boundary IoU 從 0.29 掉到 0.16。另外 R 的 fold 3 只跑 32 個 epoch 就早停，A 跑滿 100，所以 fold 3 的落後可能部分來自訓練長度，單一 seed 分不出來。依照事先訂的判讀規則，標註以 NOAA 專家標註為準，多預測的部分就是誤報，所以這個設定不採用；判讀表和實例放在備用頁。",
    "實驗：R ring8");
  const sig = (t, good) => td(t, { align: "center", bold: true, color: good ? BLUE : TERRA });
  const ns = t => td(t, { align: "center", color: SLATE });
  const num = (t, o = {}) => td(t, Object.assign({ align: "right" }, o));
  s.addTable([
    [th("指標", { align: "left" }), th("A 基準"), th("R ring8"), th("R − A [95% CI]")],
    [td("Oil IoU（原主指標）"), num("0.4332"), num("0.4129"), ns("−0.020 [−0.051, +0.003]")],
    [td("含背景 Oil IoU"), num("0.4145"), num("0.3905"), ns("−0.024 [−0.055, −0.000]")],
    [td("Boundary IoU"), num("0.2929"), num("0.1585", { bold: true, color: TERRA }), sig("−0.134 [−0.157, −0.115]", false)],
    [td("precision"), num("0.653"), num("0.564", { bold: true, color: TERRA }), sig("−0.089 [−0.121, −0.058]", false)],
    [td("recall"), num("0.566"), num("0.605", { bold: true, color: BLUE }), sig("+0.039 [+0.011, +0.070]", true)],
    [td("油汙 8 px 內誤報（每折像素）"), num("約 60 萬"), num("約 106 萬", { bold: true, color: TERRA }), ns("占全部誤報 55.6% → 62.4%")],
    [td("背景小圖誤報（每百萬像素）"), num("2,785"), num("3,813"), ns("+1,028 [−114, +2,194]")],
    [td("實際 epochs（fold 1／2／3）"), num("37／66／100"), num("92／82／32"), ns("fold 3 早停，A 跑滿 100")],
  ], Object.assign({ x: M, y: 1.85, w: W - 2 * M, colW: [3.6, 2.0, 2.0, W - 2 * M - 7.6], rowH: 0.48, objectName: "res3-table" }, TBL));
  txt(s, [{ text: "藍", options: { bold: true, color: BLUE } }, { text: "＝顯著變好　", options: {} },
          { text: "棕", options: { bold: true, color: TERRA } }, { text: "＝顯著變差　灰＝CI 含 0 或無 CI", options: {} }],
    { x: M, y: 6.5, w: 6.2, h: 0.35, fontSize: 12, color: SLATE, objectName: "res3-legend" });
  txt(s, "依事先訂的判讀規則（見備用頁），此設定不採用",
    { x: 6.6, y: 6.5, w: W - M - 6.6, h: 0.35, fontSize: 13, bold: true, color: TERRA, align: "right", objectName: "res3-verdict" });
}

// ---------------------------------------------------------------- 11 look-alike: threshold sweep
{
  const s = slide("CONTENT", "look-alike 分析（1）門檻掃描", "相同 recall 下，C 的背景警報率較低",
    "接下來是跟 look-alike 有關的分析。HNM 讓背景誤報變少，但 recall 也下降，這有可能只是模型整體比較少預測油，效果類似把門檻調高。所以這裡在 recall 相同的條件下比較：每組各自調整門檻，讓油汙周邊的 recall 等於 A 在門檻 0.5 時的 recall，也就是圖上的虛線，再比較 test 背景小圖中有任一像素被判成油的比例，這裡叫背景警報率。每條線是一組模型把門檻由高往低調，圓點是門檻 0.5。C 的曲線在 A、B 下方，在相同 recall 下，C 的背景警報率三折平均是 2.5%，A 是 4.8%，差 −0.023，CI 不含 0；B 跟 A 沒有顯著差異。所以 HNM 的誤報下降不是單純因為變保守。另外也看到，模型的輸出接近二值，門檻降到接近 0 也只多 0.03 到 0.13 的 recall，D 在 fold 2、3 不管門檻怎麼調都達不到 A 的 recall，表示有一部分油汙被判成背景的機率很高，沒辦法用門檻救回來。這裡的 CI 是條件式的，大約只有一半的 bootstrap 抽樣中 C 能達到目標 recall。",
    "look-alike 分析");
  txt(s, "各組調整門檻，使油汙周邊 recall 等於 A 在門檻 0.5 的 recall（虛線），再比較 test 背景小圖中有任一像素判為油的比例",
    { x: M, y: 1.6, w: W - 2 * M, h: 0.35, fontSize: 14, color: SLATE, objectName: "sweep-sub" });
  const iw = W - 2 * M, ih = iw * 780 / 2520;
  s.addImage({ path: IMG + "sweep.png", x: M, y: 2.0, w: iw, h: ih, altText: "三折的 recall 對背景小圖警報率曲線：C HNM-A 的線在 A、B 下方", objectName: "sweep-img" });
  const facts = [
    ["2.5% vs 4.8%", "C − A：−0.023 [−0.033, −0.011]；C − B：−0.027 [−0.040, −0.017]", BLUE],
    ["B ≈ A", "B − A：+0.004 [−0.005, +0.015]", SLATE],
    ["部分油汙無法以門檻救回", "輸出接近二值；D 在 fold 2、3 無法達到 A 的 recall", TERRA],
  ];
  const gap = 0.3, cw = (W - 2 * M - 2 * gap) / 3, y = 2.0 + ih + 0.12;
  facts.forEach(([big, d, col], i) => {
    const x = M + i * (cw + gap);
    card(s, x, y, cw, 6.9 - y, WHITE, `sweep-card-${i}`);
    txt(s, big, { x: x + 0.25, y: y + 0.08, w: cw - 0.5, h: 0.42, fontSize: 20, bold: true, color: col, objectName: `sweep-big-${i}` });
    txt(s, d, { x: x + 0.25, y: y + 0.52, w: cw - 0.5, h: 0.5, fontSize: 12, color: INK, objectName: `sweep-desc-${i}` });
  });
}

// ---------------------------------------------------------------- 12 look-alike: which oil is lost
{
  const s = slide("CONTENT", "look-alike 分析（2）漏報的油汙", "C、D 多漏掉的是弱對比、較細的油汙",
    "接著看 HNM 漏掉的是哪些油汙。我把 1,140 個 GT 油汙區塊依屬性分層，算各組的 recall，再跟隨機背景 B 比較，這樣可以扣掉換背景池本身的影響。左圖是 recall 的差。C 的下降主要在弱對比的油汙，也就是近紅外波段跟周圍海水亮度差不多的油汙，差 −0.054；另外是厚度 6 到 15 像素的細長油汙，差 −0.066；亮油、暗油和面積大的油汙跟 B 沒有顯著差異。D 除了這些之外，亮油和大片油汙也有下降。右邊：有 73 塊油汙是 A 偵測到、C 幾乎完全漏掉的，中位面積 545 像素、厚度 6.7 像素、對比接近 0；C 給這些油汙的最大機率中位數只有 0.03，所以調門檻也偵測不到。漏報集中在少數場景，有 20 個場景是 A 幾乎都偵測到、C 接近 0；C 跟 D 漏掉的像素有 59% 重疊，B 只有 4%，所以這跟 HNM 有關，不太像隨機變動。補充一下，整塊漏掉的部分大約只占 C recall 下降的四成，其餘是油汙範圍變小。",
    "look-alike 分析");
  const cats = ["亮油", "暗油", "弱對比", "細 <6 px", "6–15 px", "厚 ≥15 px"];
  s.addChart(pres.charts.BAR, [
    { name: "C HNM-A − B", labels: cats, values: [0.011, 0.007, -0.054, -0.048, -0.066, 0.004] },
    { name: "D HNM-A+SI − B", labels: cats, values: [-0.072, -0.000, -0.073, -0.017, -0.052, -0.040] },
  ], Object.assign({ x: M, y: 1.75, w: 7.1, h: 4.55, barDir: "col", barGrouping: "clustered", barGapWidthPct: 55,
    chartColors: [HX.accent1, HX.accent3], valAxisMinVal: -0.1, valAxisMaxVal: 0.04, valAxisMajorUnit: 0.02,
    valAxisLabelFormatCode: "+0.00;-0.00;0", showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.00;-0.00;0",
    dataLabelFontSize: 10, showLegend: true, legendPos: "t", showTitle: true, title: "油汙 recall 差（對 B，三折平均）",
    titleFontSize: 14, titleColor: HX.dk1, objectName: "lost-chart" }, chartText));
  txt(s, "CI 不含 0：C − B 弱對比、6–15 px；D − B 亮油、弱對比、6–15 px、≥15 px（對比用 B8A，同 10/22 分層）",
    { x: M, y: 6.4, w: 7.1, h: 0.45, fontSize: 12, color: SLATE, objectName: "lost-chart-note" });
  const x0 = 8.05, w = W - M - x0;
  const facts = [
    ["73 塊", "A 偵測到、C 幾乎完全漏掉的油汙：中位面積 545 px、厚 6.7 px、對比 ≈ 0（兩者都偵測到者：1,544 px、9.0 px、+0.07）", TERRA],
    ["0.03", "上述油汙在 C 中的最大 P(oil) 中位數；降低門檻也偵測不到", TERRA],
    ["20 景", "漏報集中在少數場景；C 與 D 漏掉的像素 59% 重疊（B 為 4%）", NAVY],
  ];
  facts.forEach(([big, d, col], i) => {
    const y = 1.85 + i * 1.62;
    card(s, x0, y, w, 1.47, WHITE, `lost-card-${i}`);
    txt(s, big, { x: x0 + 0.3, y: y + 0.12, w: w - 0.6, h: 0.55, fontSize: 26, bold: true, color: col, objectName: `lost-big-${i}` });
    txt(s, d, { x: x0 + 0.3, y: y + 0.7, w: w - 0.6, h: 0.7, fontSize: 13, color: INK, objectName: `lost-desc-${i}` });
  });
}

// ---------------------------------------------------------------- 13 look-alike: examples
{
  const s = slide("CONTENT", "look-alike 分析（3）實例", "C 漏掉的油汙與 HNM 選入的背景小圖外觀相近",
    "左邊是三個被 C 漏掉的油汙，每一列是同一個位置，依序是真彩影像加 GT 輪廓、A 的機率圖、C 的機率圖，黃色是高機率、紫色是低機率。第一列是希臘 Agia Zoni 事件 9 月 18 日的油汙，A 整塊都有偵測到，C 是 0；第二列是墨西哥灣一條細長的油汙；第三列是大西洋暗色海面上一條很細的油汙。這幾塊在真彩影像上都不太明顯。右邊是 HNM 選進訓練的背景小圖，NOAA 標為背景，但基準模型判成油。上面兩張是同一個地點 5 天前、9 月 13 日的影像，可以看到細的暗線和淡的色帶；下面兩張是墨西哥灣同一個圖塊其他日期的條紋。外觀跟左邊漏掉的油汙蠻接近的。我也檢查過漏報量跟「同地點的硬負樣本數量」有沒有關係，結果沒有相關，所以應該不是模型記住地點，比較可能是學到這類外觀屬於背景。下一頁用模型特徵來看。",
    "look-alike 分析");
  const sz = 1.42, gap = 0.08, y0 = 2.05;
  ["影像＋GT", "P(oil)　A", "P(oil)　C"].forEach((h, j) =>
    txt(s, h, { x: M + j * (sz + gap), y: 1.7, w: sz, h: 0.3, fontSize: 13, bold: true, color: NAVY, align: "center", objectName: `lex-colh-${j}` }));
  const rows = [["ex34", "希臘 Agia Zoni · 34SGH", "2017/9/18 · recall A 1.00 → C 0.00"],
                ["ex16", "墨西哥灣 · 16RCT", "2020/3/23 · recall A 1.00 → C 0.00"],
                ["ex19", "大西洋 · 19TDE", "2019/11/26 · recall A 0.61 → C 0.20"]];
  rows.forEach(([ex, where, cap], i) => {
    const y = y0 + i * (sz + 0.13);
    ["rgb", "A", "C"].forEach((k, j) =>
      s.addImage({ path: `${IMG}${ex}_${k}.png`, x: M + j * (sz + gap), y, w: sz, h: sz,
        altText: `${where} ${k === "rgb" ? "真彩影像與 GT 輪廓" : k + " 的油汙機率"}`, objectName: `lex-${ex}-${k}` }));
    const tx = M + 3 * (sz + gap) + 0.1;
    txt(s, [{ text: where, options: { bold: true, color: NAVY, breakLine: true } }, { text: cap, options: {} }],
      { x: tx, y: y + 0.35, w: 7.0 - tx, h: 0.8, fontSize: 12, objectName: `lex-cap-${i}` });
  });
  txt(s, "P(oil)：黃＝高、紫＝低；白線＝GT 油汙；深灰＝未推論", { x: M, y: 6.6, w: 6.3, h: 0.3, fontSize: 11, color: SLATE, objectName: "lex-legend" });
  const x0 = 7.25, tw = 2.12, tg = 0.15, tx0 = x0 + (W - M - x0 - 2 * tw - tg) / 2;
  txt(s, "C 的硬負樣本（NOAA 標為背景，基準模型判為油）", { x: x0, y: 1.7, w: W - M - x0, h: 0.3, fontSize: 13, bold: true, color: NAVY, objectName: "hn-head" });
  const tiles = [["hn1", "34SGH 9/13（5 天前）"], ["hn2", "34SGH 9/13（5 天前）"], ["hn3", "16RCT 2019/3/24"], ["hn4", "16RCT 2023/8/10"]];
  tiles.forEach(([k, cap], i) => {
    const x = tx0 + (i % 2) * (tw + tg), y = 2.08 + Math.floor(i / 2) * (tw + 0.33);
    s.addImage({ path: `${IMG}${k}.png`, x, y, w: tw, h: tw, altText: `硬負樣本 ${cap}`, objectName: `hn-img-${i}` });
    txt(s, cap, { x, y: y + tw + 0.02, w: tw, h: 0.25, fontSize: 11, color: SLATE, objectName: `hn-cap-${i}` });
  });
}

// ---------------------------------------------------------------- 14 look-alike: feature space
{
  const s = slide("CONTENT", "look-alike 分析（4）特徵空間", "在基準模型的特徵中，漏報油汙接近硬負樣本",
    "這頁用模型特徵來看。特徵取自基準 A，也就是挑選硬負樣本時用的模型，取 Prithvi encoder 的輸出。每個 test 油汙區塊、每張訓練小圖中的油汙、每張硬負樣本和隨機背景，各取一個平均特徵向量。圖是三折的 PCA 投影：藍色是訓練油汙，橘色是 C 的硬負樣本，灰色是隨機背景，綠圈是 A、C 都有偵測到的 test 油汙，紅叉是 C 漏掉的。紅叉大多落在橘色的範圍。量化的方式是：對每個 test 油汙區塊，在訓練油汙加硬負樣本裡找最近的 10 個鄰居，看其中硬負樣本占多少。漏掉的油汙，鄰居中硬負樣本占 30% 到 50%；沒漏掉的大約是 0。用這個比例預測油汙會不會被漏掉，AUC 是 C 0.78、D 0.83，比用弱對比的 0.59、面積小的 0.66 高。另外，漏掉的油汙離訓練油汙比較遠、離一般背景比較近。目前的解讀是：HNM 選到的 look-alike 跟這些弱對比油汙在特徵上有重疊，把這些 look-alike 當背景訓練，重疊範圍內的油汙也會被判成背景。這是相關性分析，特徵取自 A，而且只有一個 seed。",
    "look-alike 分析");
  txt(s, "基準 A 的 Prithvi encoder 特徵（即挑選硬負樣本所用的模型）；每個區域取平均特徵，以 PCA 投影",
    { x: M, y: 1.6, w: W - 2 * M, h: 0.35, fontSize: 14, color: SLATE, objectName: "fs-sub" });
  const iw = W - 2 * M, ih = iw * 780 / 2520;
  s.addImage({ path: IMG + "pca_enc.png", x: M, y: 2.0, w: iw, h: ih, altText: "三折特徵主成分投影：C 漏掉的油（紅叉）多落在硬負樣本（橘）區域", objectName: "fs-img" });
  const facts = [
    ["AUC 0.78／0.83", "以最近 10 個鄰居中硬負樣本的比例預測 C／D 是否漏報", BLUE],
    ["30–50% vs 0", "漏報油汙的鄰居中硬負樣本比例（未漏報者約為 0）", TERRA],
    ["0.59、0.66", "對照：以弱對比、面積小預測的 AUC", SLATE],
  ];
  const gap = 0.3, cw = (W - 2 * M - 2 * gap) / 3, y = 2.0 + ih + 0.12;
  facts.forEach(([big, d, col], i) => {
    const x = M + i * (cw + gap);
    card(s, x, y, cw, 6.9 - y, WHITE, `fs-card-${i}`);
    txt(s, big, { x: x + 0.25, y: y + 0.08, w: cw - 0.5, h: 0.42, fontSize: 20, bold: true, color: col, objectName: `fs-big-${i}` });
    txt(s, d, { x: x + 0.25, y: y + 0.52, w: cw - 0.5, h: 0.5, fontSize: 12, color: INK, objectName: `fs-desc-${i}` });
  });
}

// ---------------------------------------------------------------- 15 look-alike: kNN-filtered hard negatives (preliminary)
{
  const s = slide("CONTENT", "look-alike 分析（5）初步結果", "依特徵替換硬負樣本：救回部分漏報油汙，背景警報率不變",
    "根據上一頁的結果，我先試了一個只用訓練資料的做法，昨晚跑完一輪，這裡是初步結果。G 是從 C 的硬負樣本出發，在 A 的 encoder 特徵上，如果一張硬負樣本最近的 10 個鄰居裡有三成以上是訓練油汙，就把它換成同一個場景裡的其他候選，三折各換掉 22% 到 33%，總張數不變。G0 是對照組，在同樣的場景換掉同樣張數，但換哪幾張是隨機的。表上看三件事。第一，相同 recall 下的背景警報率，G 是 2.5%，跟 C 一樣，仍然比 A、B 低，所以分辨 look-alike 的能力沒有因為換掉這些小圖而變差。第二，C 整塊漏掉的 73 塊油汙，G 抓回約一半，recall 0.50，G0 只有 0.08，G 減 G0 的 CI 不含 0。這一欄是用 C 自己的漏報挑出來的，跟 C 比一定偏向 G，所以主要看 G 跟 G0 的差。第三，主指標 Oil IoU 沒有變好，G 跟 C 都是 0.42，仍低於 B；弱對比油汙的 recall 有上升，但跟 C 的差 CI 含 0。另外，對照組 G0 本身比 C 差很多，fold 3 只訓練 34 個 epoch 就早停，所以 G 跟 G0 的差有一部分是 G0 自己變差。目前只有一個 seed，C、G、G0 已經開始補第二個 seed，下週再完整報告。",
    "look-alike 分析");
  txt(s, "G：C 的硬負樣本中，在 A 的 encoder 特徵上最近 10 個鄰居有 ≥ 30% 是訓練油汙者（各折 22–33%），換成同場景的其他候選；"
    + "G0（對照）：同場景、同張數隨機替換。張數與其他設定皆與 C 相同",
    { x: M, y: 1.6, w: W - 2 * M, h: 0.6, fontSize: 14, color: SLATE, objectName: "knn-sub" });
  const n = (t, o = {}) => td(t, Object.assign({ align: "right" }, o));
  const R = (a, v, hl) => [td(a, hl ? { bold: true, fill: { color: hl } } : {})]
    .concat(v.map(t => n(t, Object.assign(t === "到不了" ? { color: SLATE } : {}, hl ? { bold: true, fill: { color: hl } } : {}))));
  s.addTable([
    [th("組", { align: "left" }), th("相同 recall 下的背景警報率"), th("C 整塊漏掉的油汙 recall"), th("弱對比油汙 recall"), th("Oil IoU")],
    R("A 基準", ["4.8%", "0.89", "0.53", "0.433"]),
    R("B 隨機（同池）", ["5.2%", "0.62", "0.53", "0.447"]),
    R("C HNM-A", ["2.5%", "0.03", "0.47", "0.421"]),
    R("G HNM-A+kNN", ["2.5%", "0.50", "0.50", "0.422"], "F4E6DA"),
    R("G0 隨機替換", ["到不了", "0.08", "0.44", "0.374"]),
  ], Object.assign({ x: M, y: 2.3, w: W - 2 * M, colW: [2.6, 2.75, 2.95, 2.0, W - 2 * M - 10.3], rowH: 0.42, objectName: "knn-table" }, TBL));
  const facts = [
    ["G − C +0.000", "相同 recall 下的背景警報率 [−0.008, +0.011]；G − A −0.022、G − B −0.026（CI 不含 0）", BLUE],
    ["G − G0 +0.35", "C 整塊漏掉的油汙 recall [+0.11, +0.56]；與 C 比較會偏向 G，以 G0 為對照", TERRA],
    ["Oil IoU 未改善", "G − C +0.001、G − B −0.025；G0 本身較差（fold 3 早停於 34 epochs），C、G、G0 補 seed 中", SLATE],
  ];
  const gap = 0.3, cw = (W - 2 * M - 2 * gap) / 3, y = 5.2;
  facts.forEach(([big, d, col], i) => {
    const x = M + i * (cw + gap);
    card(s, x, y, cw, 6.9 - y, WHITE, `knn-card-${i}`);
    txt(s, big, { x: x + 0.25, y: y + 0.12, w: cw - 0.5, h: 0.45, fontSize: 20, bold: true, color: col, objectName: `knn-big-${i}` });
    txt(s, d, { x: x + 0.25, y: y + 0.62, w: cw - 0.5, h: 1.1, fontSize: 13, color: INK, objectName: `knn-desc-${i}` });
  });
}

// ---------------------------------------------------------------- 16 next steps
{
  pres.addSection({ title: "下一步" }); section = "下一步";
  const s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: section });
  s.addText("後續規劃與討論", { placeholder: "kicker" });
  s.addText("後續規劃：降低 HNM 對弱對比油汙的影響", { placeholder: "title" });
  s.addNotes("接下來的規劃。第一，上一頁的依特徵替換硬負樣本，目前只有一個 seed，而且對照組 G0 的變動很大，所以 C、G、G0 已經開始補第二個 seed，只換訓練的 seed、背景名冊不變，確認 G 的效果會不會重現，下週報告。第二，如果效果不穩定，再試混合背景，硬負樣本跟隨機背景各一半，或是提高弱對比油汙在訓練中的權重。第三想請大家討論主表要放哪些指標，我的想法是 Oil IoU、含背景 Oil IoU、Boundary IoU 並列，再加上相同 recall 下的背景警報率。第四是邊界的部分，之後可以試油汙外圈加權的 loss。限制方面：目前都是單一 seed，相同 recall 比較的 CI 是條件式的，特徵分析也只是相關性。");
  const cards = [
    ["依特徵替換硬負樣本：補 seed", "C、G、G0 補第二個 seed（只換訓練 seed，背景名冊不變），確認 G 的效果是否重現"],
    ["混合背景或提高弱對比油汙權重", "硬負樣本與隨機背景各半；或提高弱對比油汙的訓練權重"],
    ["討論：主表指標", "Oil IoU、含背景 Oil IoU、Boundary IoU 並列，並加入相同 recall 下的背景警報率"],
    ["邊界（次要）", "嘗試油汙外圈加權的 loss；方塊假影從推論端處理"],
  ];
  const cw = (W - 2 * M - 0.4) / 2, ch = 2.0;
  cards.forEach(([h, d], i) => {
    const x = M + (i % 2) * (cw + 0.4), y = 1.95 + Math.floor(i / 2) * (ch + 0.35);
    card(s, x, y, cw, ch, "1F2E52", `next-card-${i}`);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.32, w: 0.5, h: 0.5, fill: { color: i < 2 ? AMBER : TERRA }, line: { type: "none" }, objectName: `next-num-bg-${i}` });
    txt(s, String(i + 1), { x: x + 0.3, y: y + 0.32, w: 0.5, h: 0.5, fontSize: 18, bold: true, color: NAVY, align: "center", valign: "middle", objectName: `next-num-${i}` });
    txt(s, h, { x: x + 1.0, y: y + 0.3, w: cw - 1.3, h: 0.5, fontSize: 20, bold: true, color: PAPER, objectName: `next-h-${i}` });
    txt(s, d, { x: x + 1.0, y: y + 0.9, w: cw - 1.3, h: 1.0, fontSize: 15, color: "C9D2E0", objectName: `next-d-${i}` });
  });
  txt(s, "限制：單一 seed；CI 僅反映 test 場景抽樣；相同 recall 比較的 CI 為條件式；特徵分析為相關性，特徵取自 A",
    { x: M, y: 6.75, w: W - 2 * M - 1.0, h: 0.4, fontSize: 12, color: "9AA8BF", objectName: "next-limits" });
}

// ---------------------------------------------------------------- 17 backup
{
  pres.addSection({ title: "備用" }); section = "備用";
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText("備用", { placeholder: "kicker" });
  s.addText("七組逐折數字", { placeholder: "title" });
  s.addNotes("備用頁：七組逐折的 Oil IoU，以及三折平均的含背景 Oil IoU、Boundary IoU、precision、recall、背景小圖誤報和實際 epoch 數。");
  const R = (a, f1, f2, f3, m, wb, bi, p, r, bg, ep, hl) => [a, f1, f2, f3, m, wb, bi, p, r, bg, ep].map((t, j) =>
    td(t, Object.assign({ align: j === 0 ? "left" : "right" }, hl ? { fill: { color: hl } } : {})));
  s.addTable([
    ["組", "Oil IoU f1", "f2", "f3", "平均", "含背景", "Boundary", "precision", "recall", "背景誤報/M", "epochs"].map((t, j) => th(t, { align: j === 0 ? "left" : "center", fontSize: 12 })),
    R("A 基準", "0.4637", "0.4634", "0.3726", "0.4332", "0.4145", "0.2929", "0.653", "0.566", "2,785", "37/66/100"),
    R("B 隨機（同池）", "0.5189", "0.4521", "0.3698", "0.4469", "0.4255", "0.3349", "0.712", "0.545", "2,697", "105/55/78", "E3ECF6"),
    R("C HNM-A", "0.4789", "0.4253", "0.3572", "0.4205", "0.4069", "0.2861", "0.672", "0.534", "1,747", "42/50/100"),
    R("D HNM-A+SI", "0.4866", "0.4098", "0.3014", "0.3993", "0.3893", "0.2885", "0.668", "0.501", "1,181", "42/57/81"),
    R("E HNM-EffNet", "0.4253", "0.4379", "0.3412", "0.4015", "0.3865", "0.2452", "0.620", "0.538", "2,338", "34/56/39"),
    R("F HNM-EffNet+SI", "0.4412", "0.4077", "0.3316", "0.3935", "0.3822", "0.2715", "0.658", "0.493", "1,906", "50/47/40"),
    R("R ring8", "0.4715", "0.4219", "0.3452", "0.4129", "0.3905", "0.1585", "0.564", "0.605", "3,813", "92/82/32", "F4E6DA"),
  ], Object.assign({ x: M, y: 1.85, w: W - 2 * M, colW: [1.75, 1.05, 0.95, 0.95, 0.95, 1.0, 1.1, 1.05, 0.95, 1.25, 1.133], rowH: 0.5, objectName: "backup-table" }, TBL));
  txt(s, "precision、recall 為三折平均；背景誤報為 test 背景小圖每百萬像素的誤報像素", { x: M, y: 6.3, w: 10, h: 0.35, fontSize: 12, color: SLATE, objectName: "backup-note" });
}

// ---------------------------------------------------------------- 18 backup ring8 examples
{
  const s = slide("CONTENT", "備用　R ring8 實例", "R 在油汙外圍多預測一圈（實例）",
    "看實際的例子。每列同一個位置，左邊是影像加 GT 輪廓，中間是 A 的預測，右邊是 R 的預測；綠色是抓對的油，紅色是誤報，橘色是漏報。上面這列是 fold 3 的一景：A 漏掉左下一大塊油，R 全部抓到了，這是 recall 上升的部分；但 R 在油汙外圍多畫了一圈紅色，誤報從 6 千多增加到 1 萬 4 千像素。下面這列是 fold 2 的細長油帶：抓對的油幾乎一樣，但 R 外圍的紅色那圈明顯變寬，誤報加倍。這就是 Boundary IoU 大降的原因：預測的邊界整圈往外移。",
    "備用");
  const sz = 2.05, gap = 0.12, x0 = M, y0 = 1.95;
  const rows = [["ex2", "fold 3 · 2021-06-17 · S2 15RVM", "A：漏報 19,970、誤報 6,341 → R：漏報 0、誤報 13,964\n（亮景，底圖依此視窗 2–98% 拉伸）"],
                ["ex4", "fold 2 · 2018-06-27 · S2 16RCT", "抓對的油幾乎相同（2,222 → 2,240）；誤報 2,802 → 5,866"]];
  ["影像＋GT 輪廓", "A 基準", "R ring8"].forEach((h, j) =>
    txt(s, h, { x: x0 + j * (sz + gap), y: y0 - 0.05, w: sz, h: 0.3, fontSize: 14, bold: true, color: NAVY, align: "center", objectName: `ex-colh-${j}` }));
  rows.forEach(([ex, scene, cap], i) => {
    const y = y0 + 0.3 + i * (sz + 0.25);
    ["rgb", "A", "R"].forEach((k, j) =>
      s.addImage({ path: `${IMG}${ex}_${k}.png`, x: x0 + j * (sz + gap), y, w: sz, h: sz,
        altText: `${scene} ${k === "rgb" ? "真彩影像與 GT 輪廓" : k + " 的預測"}`, objectName: `ex-${ex}-${k}` }));
    const tx = x0 + 3 * (sz + gap) + 0.25;
    txt(s, [{ text: scene, options: { bold: true, color: NAVY, breakLine: true } }, { text: cap, options: {} }],
      { x: tx, y: y + 0.45, w: W - M - tx, h: 1.2, fontSize: 15, objectName: `ex-cap-${i}` });
  });
  const lx = x0 + 3 * (sz + gap) + 0.25;
  const leg = [["抓對的油", "3CC83C"], ["誤報", "E62828"], ["漏報", "FFA500"], ["GT 輪廓", "00FFFF"]];
  leg.forEach(([t, col], i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: lx + i * 1.3, y: y0 + 0.02, w: 0.22, h: 0.22, fill: { color: col }, line: { color: "5B6472", width: 0.5 }, objectName: `ex-leg-sw-${i}` });
    txt(s, t, { x: lx + 0.3 + i * 1.3, y: y0 - 0.02, w: 1.0, h: 0.3, fontSize: 12, objectName: `ex-leg-${i}` });
  });
}

// ---------------------------------------------------------------- 19 backup ring8 interpretation
{
  const s = slide("CONTENT", "備用　R ring8 判讀", "ring8 判讀（依事先訂定的規則）",
    "這個判讀表是看到結果之前就先寫好的，避免看了數字才找解釋。結果的現象符合第二列：Oil IoU 持平、油旁誤報增加、precision 下降。但第二列的下一步是「討論標註範圍」，而我們的標註是 NOAA 專家標註，以標註為準，所以多畫的就是畫錯，結論落在第四列：油外圈的未標註區應該視為背景，拿掉之後模型缺少邊界的監督，預測就往外擴。上週看到油旁未標註誤判率是 3.8 倍，代表這圈是困難背景，應該加強，而不是忽略。所以不採用 ring8。",
    "備用");
  const mark = (t, on) => td(t, { align: "center", bold: on, color: on ? WHITE : SLATE, fill: on ? { color: TERRA } : undefined });
  const hi = (t, on) => td(t, on ? { bold: true, fill: { color: "F4E6DA" } } : {});
  s.addTable([
    [th("事先訂的結果", { align: "left" }), th("解讀", { align: "left" }), th("本次")],
    [td("Oil IoU 或 Boundary IoU 顯著變好、recall 上升"), td("油外圈有沒框到的油"), mark("✗", false)],
    [td("Oil IoU 持平，油旁誤報增加、precision 下降"), td("模型學會在油旁多畫，多畫的在標註下算誤報"), mark("現象符合", false)],
    [td("各指標都無差異"), td("邊界誤報不是訓練標籤造成"), mark("✗", false)],
    [hi("R 顯著變差", true), hi("外圈未標註區應視為背景，忽略後模型缺少邊界監督", true), mark("✓ 結論", true)],
  ], Object.assign({ x: M, y: 1.85, w: W - 2 * M, colW: [4.6, 5.6, W - 2 * M - 10.2], rowH: 0.55, objectName: "interp-table" }, TBL));
  card(s, M, 5.0, W - 2 * M, 1.75, NAVY, "interp-callout");
  txt(s, [
    { text: "標註為 NOAA 專家標註，多預測的部分視為誤報", options: { bold: true, color: AMBER, fontSize: 18, breakLine: true } },
    { text: "油旁未標註區提供油汙邊界的監督；其誤判率為確認背景的 3.8 倍，較適合加權而非忽略", options: { color: PAPER, breakLine: true } },
    { text: "注意：單一 seed；R 的 fold 3 只跑 32 epochs 就早停（A 跑滿 100）", options: { color: "C9D2E0", fontSize: 14 } },
  ], { x: M + 0.4, y: 5.2, w: W - 2 * M - 0.8, h: 1.45, fontSize: 16, paraSpaceAfter: 4, objectName: "interp-callout-text" });
}

// ---------------------------------------------------------------- 20 backup sweep table
{
  const s = slide("CONTENT", "備用　門檻掃描", "同 recall 下的背景警報率（逐折）", "備用頁：每組在 A 的 argmax recall 下的背景小圖警報率，逐折與三折平均；「到不了」表示該組只有把所有像素判成油才到得了目標 recall。最後一欄是各組在預設門檻 0.5 的警報率。", "備用");
  const n = (t, o = {}) => td(t, Object.assign({ align: "right" }, o));
  const R = (a, v, hl) => [td(a)].concat(v.map(t => n(t, Object.assign(t === "到不了" ? { color: SLATE } : {}, hl ? { fill: { color: hl } } : {}))));
  s.addTable([
    [th("組", { align: "left" }), th("fold 1"), th("fold 2"), th("fold 3"), th("三折平均"), th("門檻 0.5（平均）")],
    R("A 基準", ["4.19%", "3.59%", "6.58%", "4.79%", "4.79%"]),
    R("B 隨機（同池）", ["5.21%", "4.36%", "5.98%", "5.19%", "4.67%"]),
    R("C HNM-A", ["2.39%", "3.18%", "1.99%", "2.52%", "1.77%"], "E3ECF6"),
    R("D HNM-A+SI", ["2.65%", "到不了", "到不了", "到不了", "1.45%"]),
    R("E HNM-EffNet", ["8.80%", "4.27%", "4.51%", "5.86%", "4.50%"]),
    R("F HNM-EffNet+SI", ["到不了", "7.12%", "6.43%", "到不了", "3.22%"]),
    R("R ring8", ["3.25%", "3.93%", "4.70%", "3.96%", "4.62%"]),
  ], Object.assign({ x: M, y: 1.85, w: W - 2 * M, colW: [2.6, 1.8, 1.8, 1.8, 1.9, W - 2 * M - 9.9], rowH: 0.5, objectName: "bk-sweep-table" }, TBL));
  txt(s, "目標 recall（A 在 0.5）：fold 1 0.650、fold 2 0.580、fold 3 0.469。C − A 三折平均 −0.023 [−0.033, −0.011]、C − B −0.027 [−0.040, −0.017]；CI 只用到得了目標的抽樣（約一半）",
    { x: M, y: 6.0, w: W - 2 * M, h: 0.6, fontSize: 12, color: SLATE, objectName: "bk-sweep-note" });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
