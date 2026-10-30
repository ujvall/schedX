// Build docs/final_case_study.docx from docs/final_case_study.md in IEEE conference layout:
// US Letter, Times New Roman, single-column title block, then two 3.5-inch columns;
// Roman-numeral small-caps headings, "Abstract—" and "Index Terms—" run-ins,
// three-rule tables with captions above, "Fig. n." captions below.
//
// Usage (from the repository root, with the `docx` npm package available):
//   node docs/case_study_build/build_docx.js
//
// Content comes only from the Markdown file; nothing is added here.

const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, ExternalHyperlink, ImageRun, LevelFormat, Packer, Paragraph,
  SectionType, ShadingType, Table, TableCell, TableRow, TextRun, WidthType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..", "..");
const SRC = path.join(ROOT, "docs", "final_case_study.md");
const OUT = path.join(ROOT, "docs", "final_case_study.docx");

const FONT = "Times New Roman";
const BODY = 20;            // half-points: 10 pt
const ABSTRACT = 18;        // 9 pt
const SMALL = 16;           // 8 pt: tables, captions, references
const COLUMN = 5040;        // 3.5 in
const GUTTER = 360;         // 0.25 in
const IMAGE_WIDTH_PX = 336; // 3.5 in at 96 dpi

// --- inline formatting -------------------------------------------------------------------
function runs(text, opts = {}) {
  const out = [];
  // URLs stop before trailing sentence punctuation so links stay valid.
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|https?:\/\/[^\s)]*[^\s).,;:])/g;
  let last = 0;
  for (const m of text.matchAll(re)) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...opts }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), ...opts, bold: true }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), ...opts, font: "Courier New", size: (opts.size || BODY) - 2 }));
    else if (t.startsWith("http")) {
      out.push(new ExternalHyperlink({ link: t, children: [new TextRun({ text: t, ...opts, color: "0563C1", underline: {} })] }));
    } else out.push(new TextRun({ text: t.slice(1, -1), ...opts, italics: true }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...opts }));
  return out;
}

// --- figures and tables --------------------------------------------------------------------
function image(file, caption) {
  const buf = fs.readFileSync(file);
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const label = caption.match(/^Fig\. \d+\./)[0];
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 80, after: 40 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: IMAGE_WIDTH_PX, height: Math.round(IMAGE_WIDTH_PX * h / w) },
                                altText: { title: label, description: caption, name: path.basename(file) } })],
    }),
    new Paragraph({ alignment: AlignmentType.LEFT, keepLines: true, spacing: { after: 160 }, children: [
      new TextRun({ text: label, size: SMALL }), ...runs(caption.slice(label.length), { size: SMALL })] }),
  ];
}

function cells(line) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
}

function table(lines) {
  const header = cells(lines[0]);
  const body = lines.slice(2).map(cells);
  const n = header.length;
  const hasCI = body.some((r) => r.some((c) => c.includes("[")));
  // Column width follows the longest word and text length, so short columns are not squeezed.
  const weight = header.map((h, i) => {
    const texts = [h, ...body.map((r) => r[i] || "")];
    const longest = Math.max(...texts.map((t) => t.length));
    const word = Math.max(...texts.flatMap((t) => t.split(/\s+/).map((x) => x.length)));
    if (body.some((r) => (r[i] || "").includes("["))) return longest + 1; // keep CIs on one line
    if (hasCI) return word + 1; // leave the CI column its room; other columns wrap at spaces
    return Math.max(word + 1, Math.min(longest, word * 2.2));
  });
  const total = weight.reduce((a, b) => a + b, 0);
  const widths = weight.map((l) => Math.floor((COLUMN * l) / total));
  widths[n - 1] += COLUMN - widths.reduce((a, b) => a + b, 0);
  const none = { style: BorderStyle.NONE, size: 0, color: "auto" };
  const rule = { style: BorderStyle.SINGLE, size: 6, color: "000000" };
  const mk = (text, i, head, lastRow) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    margins: { top: 20, bottom: 20, left: 40, right: 40 },
    borders: { left: none, right: none, top: head ? rule : none, bottom: head || lastRow ? rule : none },
    children: [new Paragraph({ keepNext: !lastRow || undefined, alignment: AlignmentType.LEFT,
                               // A non-breaking space keeps each "[low, high]" interval whole on its line.
                               children: runs(text.replace(/\[([^\]]+), ([^\]]+)\]/, "[$1, $2]"), { size: SMALL, bold: head || undefined, noProof: true }) })],
  });
  const rows = [new TableRow({ tableHeader: true, children: header.map((h, i) => mk(h, i, true, false)) })];
  body.forEach((r, j) => rows.push(new TableRow({ cantSplit: true, children: r.map((c, i) => mk(c, i, false, j === body.length - 1)) })));
  return new Table({
    width: { size: COLUMN, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER, rows,
    borders: { top: none, bottom: none, left: none, right: none, insideHorizontal: none, insideVertical: none },
  });
}

function tableCaption(text) {
  const m = text.match(/^\*\*(TABLE [IVX]+)\.\*\* (.*)$/);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 0 },
                    children: [new TextRun({ text: m[1], size: SMALL })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { after: 60 },
                    children: [new TextRun({ text: m[2], size: SMALL, smallCaps: true })] }),
  ];
}

// --- block parser ------------------------------------------------------------------------
function parse(md) {
  const lines = md.replace(/\r\n/g, "\n").split("\n");
  const title = [], columns = [];
  let out = title;
  let i = 0, listCount = 0, inAbstract = false, inRefs = false, titleLine = 0;
  const plainStart = /^(#|>|\||!\[|```|\* |\d+\. )/;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    if (line.startsWith("```")) {
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) {
        out.push(new Paragraph({ spacing: { after: 0 }, keepLines: true, indent: { left: 120 },
                                 children: [new TextRun({ text: lines[i], font: "Courier New", size: SMALL - 1 })] }));
        i++;
      }
      i++;
      out.push(new Paragraph({ spacing: { after: 60 }, children: [] }));
      continue;
    }
    if (line.startsWith("# ")) {
      title.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
                                 children: [new TextRun({ text: line.slice(2), size: 48, noProof: true })] }));
      i++; continue;
    }
    if (line.startsWith("## ")) {
      const h = line.slice(3);
      out = columns; // the two-column body starts at the first section heading
      inAbstract = h === "Abstract";
      inRefs = h === "References";
      if (!inAbstract) {
        out.push(new Paragraph({ heading: "Heading1", keepNext: true, children: [new TextRun(inRefs ? "References" : h)] }));
      }
      i++; continue;
    }
    if (line.startsWith("### ")) {
      out.push(new Paragraph({ heading: "Heading2", keepNext: true, children: [new TextRun(line.slice(4))] }));
      i++; continue;
    }
    if (line.startsWith("> ")) {
      out.push(new Paragraph({
        spacing: { before: 60, after: 160 }, shading: { type: ShadingType.CLEAR, color: "auto", fill: "FFF4E5" },
        border: { left: { style: BorderStyle.SINGLE, size: 18, color: "E07B00", space: 6 } },
        children: runs(line.slice(2), { size: SMALL }),
      }));
      i++; continue;
    }
    const img = line.match(/^!\[(.*)\]\((.*)\)$/);
    if (img) { out.push(...image(path.resolve(path.dirname(SRC), img[2]), img[1])); i++; continue; }
    if (line.startsWith("|")) {
      const block = [];
      while (i < lines.length && lines[i].startsWith("|")) block.push(lines[i++]);
      out.push(table(block));
      out.push(new Paragraph({ spacing: { after: 100 }, children: [] }));
      continue;
    }
    if (/^\* /.test(line) || /^\d+\. /.test(line)) {
      const numbered = /^\d+\. /.test(line);
      listCount += 1; // each numbered list restarts at 1
      const same = (l) => (numbered ? /^\d+\. /.test(l) : /^\* /.test(l));
      while (i < lines.length && same(lines[i])) {
        const lastItem = !(i + 1 < lines.length && same(lines[i + 1]));
        out.push(new Paragraph({
          alignment: AlignmentType.JUSTIFIED,
          numbering: numbered ? { reference: "numbers", level: 0, instance: listCount } : { reference: "bullets", level: 0 },
          spacing: { after: lastItem ? 120 : 20 }, children: runs(lines[i].replace(/^(\* |\d+\. )/, "")) }));
        i++;
      }
      continue;
    }
    const para = [];
    while (i < lines.length && lines[i].trim() && !plainStart.test(lines[i])) para.push(lines[i++].trim());
    const text = para.join(" ");
    if (/^\*\*TABLE /.test(text)) { out.push(...tableCaption(text)); continue; }
    if (out === title) {
      titleLine += 1; // subtitle, author, course line, repository line
      const size = titleLine === 2 ? 22 : titleLine === 1 ? BODY : 18;
      out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: runs(text, { size }) }));
      continue;
    }
    if (inAbstract && text.startsWith("**Index Terms**")) {
      out.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 }, indent: { firstLine: 200 }, children: [
        new TextRun({ text: "Index Terms", size: ABSTRACT, bold: true, italics: true }),
        new TextRun({ text: text.replace("**Index Terms**", ""), size: ABSTRACT, bold: true })] }));
      continue;
    }
    if (inAbstract) {
      out.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 100 }, children: [
        new TextRun({ text: "Abstract", size: ABSTRACT, bold: true, italics: true }),
        new TextRun({ text: `—${text}`, size: ABSTRACT, bold: true })] }));
      continue;
    }
    if (inRefs) {
      out.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: { left: 340, hanging: 340 }, spacing: { after: 30 },
                               children: runs(text.replace(/^(\[\d+\]) /, "$1\t"), { size: SMALL }) }));
      continue;
    }
    out.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 288 }, spacing: { after: 100 }, children: runs(text) }));
  }
  return { title, columns };
}

const { title, columns } = parse(fs.readFileSync(SRC, "utf-8"));
const PAGE = { size: { width: 12240, height: 15840 }, margin: { top: 1080, bottom: 1440, left: 900, right: 900, header: 0, footer: 0 } };

const doc = new Document({
  creator: "Ujval Sai",
  title: "Machine-Learning-Assisted Adaptive CPU Scheduling",
  description: "SchedX case study (IEEE two-column format)",
  hyphenation: { autoHyphenation: true, doNotHyphenateCaps: true },
  styles: {
    default: { document: { run: { font: FONT, size: BODY }, paragraph: { spacing: { line: 240 } } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: BODY, smallCaps: true, color: "000000" },
        paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 160, after: 80 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: BODY, italics: true, color: "000000" },
        paragraph: { spacing: { before: 100, after: 60 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
                                          style: { paragraph: { indent: { left: 288, hanging: 180 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1)", alignment: AlignmentType.LEFT,
                                          style: { paragraph: { indent: { left: 288, hanging: 240 } } } }] },
    ],
  },
  sections: [
    { properties: { page: PAGE }, children: title },
    { properties: { type: SectionType.CONTINUOUS, page: PAGE, column: { count: 2, space: GUTTER, equalWidth: true } }, children: columns },
    { properties: { type: SectionType.CONTINUOUS, page: PAGE }, children: [new Paragraph({ spacing: { after: 0, line: 20, lineRule: "exact" }, children: [] })] },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log(`wrote ${OUT} (${buf.length} bytes)`);
});
