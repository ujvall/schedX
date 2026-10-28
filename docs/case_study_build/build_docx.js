// Build docs/final_case_study.docx from docs/final_case_study.md.
//
// Usage (from the repository root, with the `docx` npm package available):
//   node docs/case_study_build/build_docx.js
//
// Supports the Markdown subset used by the case study: headings (#, ##, ###),
// paragraphs, **bold**, *italic*, `code`, bullet and numbered lists, pipe
// tables, block quotes, fenced code blocks, and images with captions.
// Content comes only from the Markdown file; nothing is added here.

const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, ExternalHyperlink, Footer, ImageRun, LevelFormat,
  Packer, PageNumber, Paragraph, ShadingType, Table, TableCell, TableRow, TextRun, WidthType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..", "..");
const SRC = path.join(ROOT, "docs", "final_case_study.md");
const OUT = path.join(ROOT, "docs", "final_case_study.docx");

const FONT = "Calibri";
const BODY = 21;            // half-points: 10.5 pt
const SMALL = 17;           // 8.5 pt for tables and captions
const TEXT_WIDTH = 9638;    // A4 (11906) minus 2 x 1134 DXA margins
const INK = "1F2933";
const MUTED = "52514E";

// --- inline formatting -------------------------------------------------------------------
function runs(text, opts = {}) {
  const out = [];
  // URLs stop before trailing sentence punctuation so links stay valid.
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|https?:\/\/[^\s)]*[^\s).,;:])/g;
  let last = 0;
  for (const m of text.matchAll(re)) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...opts }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...opts }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: "Consolas", ...opts, size: (opts.size || BODY) - 1 }));
    else if (t.startsWith("http")) {
      out.push(new ExternalHyperlink({ link: t, children: [new TextRun({ text: t, style: "Hyperlink", ...opts })] }));
    } else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...opts }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...opts }));
  return out;
}

// --- images ------------------------------------------------------------------------------
function pngSize(buf) {
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}

function image(file, caption) {
  const buf = fs.readFileSync(file);
  const { w, h } = pngSize(buf);
  const widthPx = 640; // about 16.9 cm at 96 dpi, the full text width
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 60, after: 20 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: widthPx, height: Math.round(widthPx * h / w) },
                                altText: { title: caption.split(".")[0], description: caption, name: path.basename(file) } })],
    }),
    new Paragraph({ spacing: { after: 120 }, children: runs(caption, { size: SMALL, color: MUTED, italics: false }) }),
  ];
}

// --- tables ------------------------------------------------------------------------------
function cells(line) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
}

function table(lines) {
  const header = cells(lines[0]);
  const body = lines.slice(2).map(cells);
  const n = header.length;
  // Width proportional to the longest text in each column, with a floor.
  const lens = header.map((h, i) => Math.max(h.length, ...body.map((r) => (r[i] || "").length), 6));
  const total = lens.reduce((a, b) => a + b, 0);
  const widths = lens.map((l) => Math.floor((TEXT_WIDTH * l) / total));
  widths[n - 1] += TEXT_WIDTH - widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
  const mk = (text, i, head) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    borders: { top: border, bottom: border, left: border, right: border },
    shading: head ? { type: ShadingType.CLEAR, color: "auto", fill: "E8EEF4" } : undefined,
    // keepNext on every row but the last keeps a short table on one page.
    children: [new Paragraph({ keepNext: keep || undefined, children: runs(text, { size: SMALL, bold: head || undefined }) })],
  });
  let keep = true;
  const rows = [new TableRow({ tableHeader: true, children: header.map((h, i) => mk(h, i, true)) })];
  body.forEach((r, j) => {
    keep = j < body.length - 1;
    rows.push(new TableRow({ cantSplit: true, children: r.map((c, i) => mk(c, i, false)) }));
  });
  return new Table({ width: { size: TEXT_WIDTH, type: WidthType.DXA }, columnWidths: widths, rows });
}

// --- block parser ------------------------------------------------------------------------
function build(md) {
  const lines = md.replace(/\r\n/g, "\n").split("\n");
  const out = [];
  let i = 0;
  let listCount = 0;
  let inTitleBlock = true; // lines before the first "## " heading are centered
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    if (line.startsWith("```")) {
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) {
        out.push(new Paragraph({ spacing: { after: 0 }, shading: { type: ShadingType.CLEAR, color: "auto", fill: "F3F4F6" },
                                 children: [new TextRun({ text: lines[i], font: "Consolas", size: SMALL })] }));
        i++;
      }
      i++;
      out.push(new Paragraph({ spacing: { after: 80 }, children: [] }));
      continue;
    }
    if (line.startsWith("# ")) {
      out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
                               children: [new TextRun({ text: line.slice(2), bold: true, size: 32, color: INK })] }));
      i++; continue;
    }
    if (line.startsWith("## ")) {
      inTitleBlock = false;
      out.push(new Paragraph({ heading: "Heading1", keepNext: true, children: [new TextRun(line.slice(3))] }));
      i++; continue;
    }
    if (line.startsWith("### ")) {
      out.push(new Paragraph({ heading: "Heading2", keepNext: true, children: [new TextRun(line.slice(4))] }));
      i++; continue;
    }
    if (line.startsWith("> ")) {
      out.push(new Paragraph({
        spacing: { before: 60, after: 120 }, shading: { type: ShadingType.CLEAR, color: "auto", fill: "FFF4E5" },
        border: { left: { style: BorderStyle.SINGLE, size: 18, color: "E07B00", space: 6 } },
        children: runs(line.slice(2), { size: SMALL }),
      }));
      i++; continue;
    }
    const img = line.match(/^!\[(.*)\]\((.*)\)$/);
    if (img) {
      out.push(...image(path.resolve(path.dirname(SRC), img[2]), img[1]));
      i++; continue;
    }
    if (line.startsWith("|")) {
      const block = [];
      while (i < lines.length && lines[i].startsWith("|")) block.push(lines[i++]);
      out.push(table(block));
      out.push(new Paragraph({ spacing: { after: 60 }, children: [] }));
      continue;
    }
    if (/^\* /.test(line) || /^\d+\. /.test(line)) {
      const numbered = /^\d+\. /.test(line);
      listCount += 1; // each numbered list restarts at 1
      while (i < lines.length && (numbered ? /^\d+\. /.test(lines[i]) : /^\* /.test(lines[i]))) {
        const text = lines[i].replace(/^(\* |\d+\. )/, "");
        const lastItem = !(i + 1 < lines.length && (numbered ? /^\d+\. /.test(lines[i + 1]) : /^\* /.test(lines[i + 1])));
        out.push(new Paragraph({
          numbering: numbered ? { reference: "numbers", level: 0, instance: listCount } : { reference: "bullets", level: 0 },
          spacing: { after: lastItem ? 100 : 30 }, children: runs(text) }));
        i++;
      }
      continue;
    }
    // Paragraph: join consecutive non-empty plain lines.
    const para = [];
    while (i < lines.length && lines[i].trim() && !/^(#|>|\||!\[|```|\* |\d+\. )/.test(lines[i])) para.push(lines[i++].trim());
    const text = para.join(" ");
    const isCaption = /^\*\*(Table|Figure) \d+\.\*\*/.test(text);
    const isRef = /^\[\d+\] /.test(text);
    out.push(new Paragraph({
      alignment: inTitleBlock ? AlignmentType.CENTER : isRef ? AlignmentType.LEFT : AlignmentType.JUSTIFIED,
      keepNext: isCaption || undefined,
      indent: isRef ? { left: 400, hanging: 400 } : undefined,
      spacing: { after: isRef ? 40 : 100 },
      children: runs(text, isRef || isCaption ? { size: isRef ? SMALL + 1 : SMALL } : {}),
    }));
  }
  return out;
}

const doc = new Document({
  creator: "Ujval Sai",
  title: "Machine-Learning-Assisted Adaptive CPU Scheduling",
  description: "SchedX case study",
  styles: {
    default: { document: { run: { font: FONT, size: BODY, color: INK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, color: "1F4E79" }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, color: "1F4E79" }, paragraph: { spacing: { before: 120, after: 60 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
                                          style: { paragraph: { indent: { left: 360, hanging: 240 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
                                          style: { paragraph: { indent: { left: 360, hanging: 300 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
      new TextRun({ text: "SchedX case study — page ", size: SMALL, color: MUTED }),
      new TextRun({ children: [PageNumber.CURRENT], size: SMALL, color: MUTED }),
      new TextRun({ text: " of ", size: SMALL, color: MUTED }),
      new TextRun({ children: [PageNumber.TOTAL_PAGES], size: SMALL, color: MUTED }),
    ] })] }) },
    children: build(fs.readFileSync(SRC, "utf-8")),
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log(`wrote ${OUT} (${buf.length} bytes)`);
});
