const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const globalRoot = execSync('npm root -g', { encoding: 'utf8' }).trim();
const docx = require(path.join(globalRoot, 'docx'));
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  Table,
  TableRow,
  TableCell,
  ImageRun,
  AlignmentType,
  HeadingLevel,
  WidthType,
  BorderStyle,
  LevelFormat,
  PageNumber,
  Footer,
} = docx;

const ROOT = path.resolve(__dirname, '..');
const args = process.argv.slice(2);
const inputArg = args[0] || path.join('proposal', 'proposal-final.md');
const outputArg = args[1] || path.join('proposal', 'proposal-final.docx');
const INPUT = path.isAbsolute(inputArg) ? inputArg : path.join(ROOT, inputArg);
const OUTPUT = path.isAbsolute(outputArg) ? outputArg : path.join(ROOT, outputArg);
const FIG_ROOT = path.join(ROOT, 'proposal');

const md = fs.readFileSync(INPUT, 'utf8').replace(/\r\n/g, '\n');
const lines = md.split('\n');

const tableBorder = { style: BorderStyle.SINGLE, size: 1, color: 'BFC7D5' };
const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };

const children = [];
let i = 0;
let numberedRef = 0;
let bulletRef = 0;

function cleanInline(text) {
  return text
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .trim();
}

function pushParagraph(text, opts = {}) {
  if (!text || !text.trim()) return;
  children.push(new Paragraph({
    ...opts,
    children: [new TextRun({ text: text.trim() })],
  }));
}

function parseImage(line) {
  const m = line.match(/^!\[[^\]]*\]\(([^)]+)\)/);
  if (!m) return false;
  const rel = m[1];
  const abs = path.join(FIG_ROOT, rel.replace(/^figures\//, 'figures/'));
  if (fs.existsSync(abs)) {
    const ext = path.extname(abs).slice(1).toLowerCase();
    const allowed = ['png', 'jpg', 'jpeg'];
    if (allowed.includes(ext)) {
      children.push(new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new ImageRun({ type: ext === 'jpg' ? 'jpeg' : ext, data: fs.readFileSync(abs), transformation: { width: 520, height: 300 } })],
      }));
      return true;
    }
  }
  children.push(new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: `[Gambar: ${rel}]`, italics: true })] }));
  return true;
}

function parseTable(start) {
  const rows = [];
  let idx = start;
  while (idx < lines.length && lines[idx].trim().startsWith('|')) {
    rows.push(lines[idx].trim());
    idx++;
  }
  const parsed = [];
  rows.forEach((row, rIndex) => {
    const cells = row.slice(1, -1).split('|').map(c => cleanInline(c));
    if (rIndex === 1 && cells.every(c => /^:?-+:?$/.test(c))) return;
    parsed.push(cells);
  });
  if (parsed.length) {
    const widths = parsed[0].map(() => Math.floor(9000 / parsed[0].length));
    children.push(new Table({
      columnWidths: widths,
      rows: parsed.map((row, ridx) => new TableRow({
        tableHeader: ridx === 0,
        children: row.map((cell, cidx) => new TableCell({
          borders: cellBorders,
          width: { size: widths[cidx], type: WidthType.DXA },
          children: [new Paragraph({ children: [new TextRun({ text: cell, bold: ridx === 0 })] })],
        })),
      })),
    }));
  }
  return idx;
}

function parseList(start, type) {
  let idx = start;
  const reference = type === 'bullet' ? `bullet-${++bulletRef}` : `number-${++numberedRef}`;
  while (idx < lines.length) {
    const trimmed = lines[idx].trim();
    const bullet = trimmed.match(/^[-*]\s+(.*)$/);
    const number = trimmed.match(/^\d+\.\s+(.*)$/);
    const match = type === 'bullet' ? bullet : number;
    if (!match) break;
    children.push(new Paragraph({
      numbering: { reference, level: 0 },
      children: [new TextRun({ text: cleanInline(match[1]) })],
    }));
    idx++;
  }
  return { idx, reference };
}

while (i < lines.length) {
  const line = lines[i];
  const trimmed = line.trim();

  if (!trimmed) {
    i++;
    continue;
  }
  if (trimmed === '---') {
    children.push(new Paragraph({ children: [] }));
    i++;
    continue;
  }
  if (parseImage(trimmed)) {
    i++;
    continue;
  }
  if (trimmed.startsWith('|')) {
    i = parseTable(i);
    continue;
  }
  if (/^[-*]\s+/.test(trimmed)) {
    i = parseList(i, 'bullet').idx;
    continue;
  }
  if (/^\d+\.\s+/.test(trimmed)) {
    i = parseList(i, 'number').idx;
    continue;
  }
  if (trimmed.startsWith('# ')) {
    children.push(new Paragraph({ heading: HeadingLevel.TITLE, alignment: AlignmentType.CENTER, children: [new TextRun({ text: cleanInline(trimmed.slice(2)), bold: true })] }));
    i++;
    continue;
  }
  if (trimmed.startsWith('## ')) {
    children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: cleanInline(trimmed.slice(3)), bold: true })] }));
    i++;
    continue;
  }
  if (trimmed.startsWith('### ')) {
    children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: cleanInline(trimmed.slice(4)), bold: true })] }));
    i++;
    continue;
  }
  if (trimmed.startsWith('#### ')) {
    children.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: cleanInline(trimmed.slice(5)), bold: true })] }));
    i++;
    continue;
  }

  const paragraphLines = [trimmed];
  i++;
  while (i < lines.length) {
    const next = lines[i].trim();
    if (!next || next === '---' || next.startsWith('#') || next.startsWith('|') || /^[-*]\s+/.test(next) || /^\d+\.\s+/.test(next) || /^!\[[^\]]*\]\(([^)]+)\)/.test(next)) {
      break;
    }
    paragraphLines.push(next);
    i++;
  }
  pushParagraph(cleanInline(paragraphLines.join(' ')));
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: 'Times New Roman', size: 24 } } },
    paragraphStyles: [
      { id: 'Title', name: 'Title', basedOn: 'Normal', run: { size: 34, bold: true, font: 'Times New Roman' }, paragraph: { spacing: { before: 160, after: 140 }, alignment: AlignmentType.CENTER } },
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', quickFormat: true, run: { size: 28, bold: true, font: 'Times New Roman' }, paragraph: { spacing: { before: 220, after: 120 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', quickFormat: true, run: { size: 24, bold: true, font: 'Times New Roman' }, paragraph: { spacing: { before: 160, after: 100 }, outlineLevel: 1 } },
      { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', quickFormat: true, run: { size: 22, bold: true, font: 'Times New Roman' }, paragraph: { spacing: { before: 120, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: 'bullet-1', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      ...Array.from({ length: 50 }, (_, n) => ({ reference: `bullet-${n+2}`, levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] })),
      ...Array.from({ length: 80 }, (_, n) => ({ reference: `number-${n+1}`, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] })),
    ],
  },
  sections: [{
    properties: { page: { margin: { top: 1701, right: 1701, bottom: 1701, left: 2268 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun('Page '), new TextRun({ children: [PageNumber.CURRENT] })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(OUTPUT, buffer);
  console.log(`Generated ${OUTPUT}`);
});
