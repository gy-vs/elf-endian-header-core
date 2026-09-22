/**
 * Minimal ELF binary reader.
 *
 * Layout (32/64-bit) and byte order are chosen from `e_ident`. Every offset
 * and size is carried as a `bigint` so values above `Number.MAX_SAFE_INTEGER`
 * can be validated instead of silently corrupting results. All parse results
 * retain their absolute byte range inside the source buffer.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ElfBits = 32 | 64;

/** Absolute half-open byte range [offset, offset + size) inside the file. */
export interface ByteRange {
  readonly offset: bigint;
  readonly size: bigint;
  /** First byte not contained in the range. */
  readonly end: bigint;
}

/** Size of a single integer read from the buffer. */
export type IntKind =
  | 'u8'
  | 'u16'
  | 'u32'
  | 'u64'
  | 'xword'; // Elf64_Xword: u32 on ELF32, u64 on ELF64

export interface StructFieldSpec {
  readonly name: string;
  /** Offset of the field inside the structure. */
  readonly offset: number;
  readonly kind: IntKind;
  /** Read the value as a bigint (used for every offset/size/address). */
  readonly big?: boolean;
}

export interface ElfIdentification {
  readonly range: ByteRange;
  readonly bits: ElfBits;
  readonly littleEndian: boolean;
  readonly version: number;
  readonly osAbi: number;
  readonly abiVersion: number;
}

export interface RawElfHeader {
  readonly eiClass: number;
  readonly eiData: number;
  readonly eiVersion: number;
  readonly eiOsAbi: number;
  readonly eiAbiVersion: number;
  readonly type: number;
  readonly machine: number;
  readonly version: number;
  readonly entry: bigint;
  readonly phoff: bigint;
  readonly shoff: bigint;
  readonly flags: number;
  readonly ehsize: bigint;
  readonly phentsize: bigint;
  readonly phnum: number;
  readonly shentsize: bigint;
  readonly shnum: number;
  readonly shstrndx: number;
  [field: string]: number | bigint;
}

export interface ElfFileHeader {
  readonly range: ByteRange;
  readonly identification: ElfIdentification;
  readonly fields: RawElfHeader;
}

export interface RawProgramHeader {
  readonly type: number;
  readonly offset: bigint;
  readonly vaddr: bigint;
  readonly paddr: bigint;
  readonly filesz: bigint;
  readonly memsz: bigint;
  readonly flags: number;
  readonly align: bigint;
  [field: string]: number | bigint;
}

export interface ProgramHeader {
  readonly index: number;
  readonly range: ByteRange;
  readonly fields: RawProgramHeader;
}

export interface RawSectionHeader {
  readonly name: number;
  readonly type: number;
  readonly flags: bigint;
  readonly addr: bigint;
  readonly offset: bigint;
  readonly size: bigint;
  readonly link: number;
  readonly info: number;
  readonly addralign: bigint;
  readonly entsize: bigint;
  [field: string]: number | bigint;
}

export interface SectionHeader {
  readonly index: number;
  readonly range: ByteRange;
  readonly fields: RawSectionHeader;
}

export interface ParsedProgramHeaders {
  readonly entrySize: bigint;
  readonly count: number;
  readonly range: ByteRange;
  readonly headers: readonly ProgramHeader[];
}

export interface ParsedSectionHeaders {
  /** Raw `e_shnum`/`e_shstrndx` values from the file header. */
  readonly rawCount: number;
  readonly rawStringIndex: number;
  /** Count after extended numbering (section 0 `sh_size`). */
  readonly count: number;
  readonly entrySize: bigint;
  readonly range: ByteRange;
  readonly headers: readonly SectionHeader[];
  /** Resolved string table index, or `null` when there is no section table. */
  readonly stringIndex: number | null;
}

export interface ParsedElf {
  readonly dataLength: bigint;
  readonly header: ElfFileHeader;
  readonly programHeaders: ParsedProgramHeaders;
  readonly sectionHeaders: ParsedSectionHeaders;
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

export class ElfParseError extends Error {
  /** Name of the offending field (e.g. `e_shoff`, `sh_size`). */
  readonly field: string;
  /** Absolute byte offset of the offending value inside the file. */
  readonly offset: bigint;

  constructor(message: string, field: string, offset: bigint) {
    super(`${message} (field '${field}' at offset 0x${offset.toString(16)} / ${offset})`);
    this.name = 'ElfParseError';
    this.field = field;
    this.offset = offset;
  }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const EI_MAG0 = 0;
export const EI_MAG1 = 1;
export const EI_MAG2 = 2;
export const EI_MAG3 = 3;
export const EI_CLASS = 4;
export const EI_DATA = 5;
export const EI_VERSION = 6;
export const EI_OSABI = 7;
export const EI_ABIVERSION = 8;

export const ELFMAG = '\x7fELF';
export const ELFCLASS32 = 1;
export const ELFCLASS64 = 2;
export const ELFDATA2LSB = 1;
export const ELFDATA2MSB = 2;
export const EV_CURRENT = 1;

export const PT_LOAD = 1;

export const SHT_NULL = 0;
export const SHT_NOBITS = 8;

/** Reserved section index: real value lives in section 0. */
export const SHN_XINDEX = 0xffff;
export const SHN_LORESERVE = 0xff00;

export const ELF_EHDR_SIZE: Readonly<Record<ElfBits, number>> = { 32: 52, 64: 64 };
export const ELF_PHDR_SIZE: Readonly<Record<ElfBits, number>> = { 32: 32, 64: 56 };
export const ELF_SHDR_SIZE: Readonly<Record<ElfBits, number>> = { 32: 40, 64: 64 };

export const IDENT_SIZE = 16;
export const EI_NIDENT = 16;

// ---------------------------------------------------------------------------
// Field layouts
// ---------------------------------------------------------------------------

/** Fields of the file header following the 16-byte identification block. */
export function ehdrFields(bits: ElfBits): readonly StructFieldSpec[] {
  const xw: IntKind = bits === 32 ? 'u32' : 'u64';
  return [
    { name: 'type', offset: 16, kind: 'u16' },
    { name: 'machine', offset: 18, kind: 'u16' },
    { name: 'version', offset: 20, kind: 'u32' },
    { name: 'entry', offset: 24, kind: xw, big: true },
    { name: 'phoff', offset: bits === 32 ? 28 : 32, kind: xw, big: true },
    { name: 'shoff', offset: bits === 32 ? 32 : 40, kind: xw, big: true },
    { name: 'flags', offset: bits === 32 ? 36 : 48, kind: 'u32' },
    { name: 'ehsize', offset: bits === 32 ? 40 : 52, kind: 'u16', big: true },
    { name: 'phentsize', offset: bits === 32 ? 42 : 54, kind: 'u16', big: true },
    { name: 'phnum', offset: bits === 32 ? 44 : 56, kind: 'u16' },
    { name: 'shentsize', offset: bits === 32 ? 46 : 58, kind: 'u16', big: true },
    { name: 'shnum', offset: bits === 32 ? 48 : 60, kind: 'u16' },
    { name: 'shstrndx', offset: bits === 32 ? 50 : 62, kind: 'u16' },
  ];
}

export function phdrFields(bits: ElfBits): readonly StructFieldSpec[] {
  if (bits === 32) {
    return [
      { name: 'type', offset: 0, kind: 'u16' },
      { name: 'offset', offset: 4, kind: 'u32', big: true },
      { name: 'vaddr', offset: 8, kind: 'u32', big: true },
      { name: 'paddr', offset: 12, kind: 'u32', big: true },
      { name: 'filesz', offset: 16, kind: 'u32', big: true },
      { name: 'memsz', offset: 20, kind: 'u32', big: true },
      { name: 'flags', offset: 24, kind: 'u32' },
      { name: 'align', offset: 28, kind: 'u32', big: true },
    ];
  }
  return [
    { name: 'type', offset: 0, kind: 'u32' },
    { name: 'flags', offset: 4, kind: 'u32' },
    { name: 'offset', offset: 8, kind: 'u64', big: true },
    { name: 'vaddr', offset: 16, kind: 'u64', big: true },
    { name: 'paddr', offset: 24, kind: 'u64', big: true },
    { name: 'filesz', offset: 32, kind: 'u64', big: true },
    { name: 'memsz', offset: 40, kind: 'u64', big: true },
    { name: 'align', offset: 48, kind: 'u64', big: true },
  ];
}

export function shdrFields(bits: ElfBits): readonly StructFieldSpec[] {
  const xw: IntKind = bits === 32 ? 'u32' : 'u64';
  if (bits === 32) {
    return [
      { name: 'name', offset: 0, kind: 'u32' },
      { name: 'type', offset: 4, kind: 'u32' },
      { name: 'flags', offset: 8, kind: xw, big: true },
      { name: 'addr', offset: 12, kind: xw, big: true },
      { name: 'offset', offset: 16, kind: xw, big: true },
      { name: 'size', offset: 20, kind: xw, big: true },
      { name: 'link', offset: 24, kind: 'u32' },
      { name: 'info', offset: 28, kind: 'u32' },
      { name: 'addralign', offset: 32, kind: xw, big: true },
      { name: 'entsize', offset: 36, kind: xw, big: true },
    ];
  }
  return [
    { name: 'name', offset: 0, kind: 'u32' },
    { name: 'type', offset: 4, kind: 'u32' },
    { name: 'flags', offset: 8, kind: xw, big: true },
    { name: 'addr', offset: 16, kind: xw, big: true },
    { name: 'offset', offset: 24, kind: xw, big: true },
    { name: 'size', offset: 32, kind: xw, big: true },
    { name: 'link', offset: 40, kind: 'u32' },
    { name: 'info', offset: 44, kind: 'u32' },
    { name: 'addralign', offset: 48, kind: xw, big: true },
    { name: 'entsize', offset: 56, kind: xw, big: true },
  ];
}

// ---------------------------------------------------------------------------
// Low level reading
// ---------------------------------------------------------------------------

class Reader {
  readonly view: DataView;
  constructor(readonly data: Uint8Array, readonly littleEndian: boolean) {
    this.view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  }

  /** Absolute unsigned read; throws a field-tagged error when out of bounds. */
  read(kind: IntKind, bits: ElfBits, absOffset: bigint, field: string): bigint {
    const width = kind === 'xword' ? (bits === 32 ? 4 : 8) : kindWidth(kind);
    const off = this.checkRange(absOffset, BigInt(width), field);
    const le = this.littleEndian;
    switch (width) {
      case 1:
        return BigInt(this.view.getUint8(off));
      case 2:
        return BigInt(this.view.getUint16(off, le));
      case 4:
        return BigInt(this.view.getUint32(off, le));
      default:
        return this.view.getBigUint64(off, le);
    }
  }

  /** Like {@link read} but returns a safe `number`. */
  readNum(kind: IntKind, bits: ElfBits, absOffset: bigint, field: string): number {
    const v = this.read(kind, bits, absOffset, field);
    return Number(v);
  }

  checkRange(offset: bigint, size: bigint, field: string): number {
    if (offset < 0n || size < 0n) {
      throw new ElfParseError('negative range', field, offset);
    }
    const length = BigInt(this.data.byteLength);
    if (offset > length || size > length - offset) {
      throw new ElfParseError(
        `range [0x${offset.toString(16)}, +0x${size.toString(16)}) exceeds file size 0x${length.toString(16)}`,
        field,
        offset,
      );
    }
    return Number(offset);
  }
}
function kindWidth(kind: IntKind): 1 | 2 | 4 | 8 {
  switch (kind) {
    case 'u8':
      return 1;
    case 'u16':
      return 2;
    case 'u32':
      return 4;
    case 'u64':
      return 8;
    case 'xword':
      throw new Error('xword width depends on ELF class');
  }
}

export function makeRange(offset: bigint, size: bigint): ByteRange {
  return { offset, size, end: offset + size };
}

function parseStruct(
  reader: Reader,
  bits: ElfBits,
  base: bigint,
  fields: readonly StructFieldSpec[],
): Record<string, number | bigint> {
  const out: Record<string, number | bigint> = {};
  for (const f of fields) {
    const abs = base + BigInt(f.offset);
    out[f.name] = f.big
      ? reader.read(f.kind, bits, abs, f.name)
      : reader.readNum(f.kind, bits, abs, f.name);
  }
  return out;
}

/** Absolute file offset of a structure field. */
export function fieldOffset(structBase: bigint, field: StructFieldSpec): bigint {
  return structBase + BigInt(field.offset);
}

// ---------------------------------------------------------------------------
// Identification
// ---------------------------------------------------------------------------

export function parseIdentification(data: Uint8Array): ElfIdentification {
  if (data.byteLength < IDENT_SIZE) {
    throw new ElfParseError(
      `file is shorter than e_ident (${data.byteLength} < ${IDENT_SIZE} bytes)`,
      'e_ident',
      BigInt(data.byteLength),
    );
  }
  for (let i = 0; i < 4; i++) {
    if (data[i] !== ELFMAG.charCodeAt(i)) {
      throw new ElfParseError('bad ELF magic number', 'e_ident[EI_MAG]', BigInt(i));
    }
  }

  const classByte = data[EI_CLASS];
  if (classByte !== ELFCLASS32 && classByte !== ELFCLASS64) {
    throw new ElfParseError(
      `unsupported EI_CLASS ${classByte} (expected 1 (ELF32) or 2 (ELF64))`,
      'e_ident[EI_CLASS]',
      BigInt(EI_CLASS),
    );
  }
  const dataByte = data[EI_DATA];
  if (dataByte !== ELFDATA2LSB && dataByte !== ELFDATA2MSB) {
    throw new ElfParseError(
      `unsupported EI_DATA ${dataByte} (expected 1 (LSB) or 2 (MSB))`,
      'e_ident[EI_DATA]',
      BigInt(EI_DATA),
    );
  }
  if (data[EI_VERSION] !== EV_CURRENT) {
    throw new ElfParseError(
      `unsupported EI_VERSION ${data[EI_VERSION]} (expected ${EV_CURRENT})`,
      'e_ident[EI_VERSION]',
      BigInt(EI_VERSION),
    );
  }

  return {
    range: makeRange(0n, BigInt(IDENT_SIZE)),
    bits: classByte === ELFCLASS32 ? 32 : 64,
    littleEndian: dataByte === ELFDATA2LSB,
    version: data[EI_VERSION],
    osAbi: data[EI_OSABI],
    abiVersion: data[EI_ABIVERSION],
  };
}

// ---------------------------------------------------------------------------
// File header
// ---------------------------------------------------------------------------

export function parseFileHeader(data: Uint8Array): ElfFileHeader {
  const ident = parseIdentification(data);
  const bits = ident.bits;
  const reader = new Reader(data, ident.littleEndian);
  const ehSize = ELF_EHDR_SIZE[bits];

  if (BigInt(data.byteLength) < BigInt(ehSize)) {
    throw new ElfParseError(
      `file is shorter than the ${bits}-bit ELF header (${data.byteLength} < ${ehSize} bytes)`,
      'e_ehsize',
      BigInt(data.byteLength),
    );
  }

  const fields = parseStruct(reader, bits, 0n, ehdrFields(bits)) as unknown as RawElfHeader;

  if (fields.version !== EV_CURRENT) {
    throw new ElfParseError(
      `unsupported e_version ${fields.version} (expected ${EV_CURRENT})`,
      'e_version',
      fieldOffset(0n, ehdrFields(bits).find((f) => f.name === 'version')!),
    );
  }
  if (fields.ehsize !== BigInt(ehSize)) {
    throw new ElfParseError(
      `invalid e_ehsize ${fields.ehsize} (expected ${ehSize} for ELF${bits})`,
      'e_ehsize',
      fieldOffset(0n, ehdrFields(bits).find((f) => f.name === 'ehsize')!),
    );
  }

  return {
    range: makeRange(0n, BigInt(ehSize)),
    identification: ident,
    fields: {
      ...fields,
      eiClass: data[EI_CLASS],
      eiData: data[EI_DATA],
      eiVersion: data[EI_VERSION],
      eiOsAbi: data[EI_OSABI],
      eiAbiVersion: data[EI_ABIVERSION],
    },
  };
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/**
 * Ensure `offset + size` lies fully inside the buffer (bigint arithmetic).
 *
 * @param fieldLocation absolute file offset of the offending field, reported
 *   on the error (the bad range itself is quoted in the message).
 */
export function checkRange(
  dataLength: bigint,
  offset: bigint,
  size: bigint,
  field: string,
  fieldLocation: bigint = offset,
): ByteRange {
  if (offset < 0n || size < 0n) {
    throw new ElfParseError('negative offset or size', field, fieldLocation);
  }
  if (offset > dataLength || size > dataLength - offset) {
    throw new ElfParseError(
      `range [0x${offset.toString(16)}, +0x${size.toString(16)}) exceeds file size 0x${dataLength.toString(16)}`,
      field,
      fieldLocation,
    );
  }
  return makeRange(offset, size);
}

/** Assert that the three structural tables do not share any bytes. */
function assertDisjoint(a: ByteRange, b: ByteRange, field: string, fieldLocation: bigint): void {
  if (a.size === 0n || b.size === 0n) return;
  if (a.offset < b.end && b.offset < a.end) {
    throw new ElfParseError(
      `range [0x${a.offset.toString(16)}, 0x${a.end.toString(16)}) overlaps [0x${b.offset.toString(16)}, 0x${b.end.toString(16)})`,
      field,
      fieldLocation,
    );
  }
}

// ---------------------------------------------------------------------------
// Program headers
// ---------------------------------------------------------------------------

export function parseProgramHeaders(data: Uint8Array, header: ElfFileHeader): ParsedProgramHeaders {
  const bits = header.identification.bits;
  const reader = new Reader(data, header.identification.littleEndian);
  const dataLength = BigInt(data.byteLength);
  const h = header.fields;
  const layouts = ehdrFields(bits);
  const phoffField = layouts.find((f) => f.name === 'phoff')!;
  const phentsizeField = layouts.find((f) => f.name === 'phentsize')!;

  if (h.phnum === 0 && h.phoff !== 0n) {
    throw new ElfParseError('e_phoff is non-zero while e_phnum is zero', 'e_phoff', fieldOffset(0n, phoffField));
  }
  if (h.phnum !== 0 && h.phoff === 0n) {
    throw new ElfParseError('e_phnum is non-zero while e_phoff is zero', 'e_phoff', fieldOffset(0n, phoffField));
  }

  const expected = BigInt(ELF_PHDR_SIZE[bits]);
  if (h.phnum !== 0 && h.phentsize !== expected) {
    throw new ElfParseError(
      `invalid e_phentsize ${h.phentsize} (expected ${expected} for ELF${bits})`,
      'e_phentsize',
      fieldOffset(0n, phentsizeField),
    );
  }

  const count = h.phnum;
  const entrySize = h.phentsize;
  const tableSize = entrySize * BigInt(count);
  const tableRange =
    count === 0
      ? makeRange(0n, 0n)
      : checkRange(
          dataLength,
          h.phoff,
          tableSize,
          'e_phoff',
          fieldOffset(0n, phoffField),
        );

  const pLayout = phdrFields(bits);
  const pOffsetField = pLayout.find((f) => f.name === 'offset')!;
  const headers: ProgramHeader[] = [];
  for (let i = 0; i < count; i++) {
    const base = h.phoff + entrySize * BigInt(i);
    const fields = parseStruct(reader, bits, base, pLayout) as unknown as RawProgramHeader;
    const range = makeRange(base, entrySize);

    // Segment payload occupying file bytes must be fully in range.
    if (fields.filesz > 0n) {
      checkRange(
        dataLength,
        fields.offset,
        fields.filesz,
        'p_offset',
        fieldOffset(base, pOffsetField),
      );
    }
    headers.push({ index: i, range, fields });
  }

  return { entrySize, count, range: tableRange, headers };
}

// ---------------------------------------------------------------------------
// Section headers
// ---------------------------------------------------------------------------

export function parseSectionHeaders(data: Uint8Array, header: ElfFileHeader): ParsedSectionHeaders {
  const bits = header.identification.bits;
  const reader = new Reader(data, header.identification.littleEndian);
  const dataLength = BigInt(data.byteLength);
  const h = header.fields;
  const layouts = ehdrFields(bits);
  const shoffField = layouts.find((f) => f.name === 'shoff')!;
  const shnumField = layouts.find((f) => f.name === 'shnum')!;
  const shentsizeField = layouts.find((f) => f.name === 'shentsize')!;
  const shstrndxField = layouts.find((f) => f.name === 'shstrndx')!;
  const sLayout = shdrFields(bits);
  const shSizeField = sLayout.find((f) => f.name === 'size')!;
  const shLinkField = sLayout.find((f) => f.name === 'link')!;
  const shOffsetField = sLayout.find((f) => f.name === 'offset')!;

  // No section header table: shoff must be zero (and then shnum/shstrndx too).
  if (h.shoff === 0n) {
    if (h.shnum !== 0) {
      throw new ElfParseError('e_shnum is non-zero while e_shoff is zero', 'e_shnum', fieldOffset(0n, shnumField));
    }
    if (h.shstrndx !== 0) {
      throw new ElfParseError(
        'e_shstrndx is non-zero while e_shoff is zero',
        'e_shstrndx',
        fieldOffset(0n, shstrndxField),
      );
    }
    return {
      rawCount: 0,
      rawStringIndex: h.shstrndx,
      count: 0,
      entrySize: 0n,
      range: makeRange(0n, 0n),
      headers: [],
      stringIndex: null,
    };
  }

  const expected = BigInt(ELF_SHDR_SIZE[bits]);
  if (h.shentsize !== expected) {
    throw new ElfParseError(
      `invalid e_shentsize ${h.shentsize} (expected ${expected} for ELF${bits})`,
      'e_shentsize',
      fieldOffset(0n, shentsizeField),
    );
  }

  // Section 0 must always be readable; extended numbering sources the real
  // count from its sh_size, so it cannot be trusted to match e_shnum yet.
  checkRange(dataLength, h.shoff, h.shentsize, 'e_shoff', fieldOffset(0n, shoffField));
  const shdr0 = parseStruct(reader, bits, h.shoff, sLayout);

  let count: number;
  if (h.shnum === 0) {
    // Extended section numbering: real count is section 0's sh_size.
    const realCount = shdr0.size as bigint;
    if (realCount === 0n) {
      throw new ElfParseError(
        'e_shnum is 0 (extended numbering) but section 0 sh_size is also 0',
        'sh_size',
        fieldOffset(h.shoff, shSizeField),
      );
    }
    if (realCount > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new ElfParseError(
        `extended section count ${realCount} exceeds Number.MAX_SAFE_INTEGER`,
        'sh_size',
        fieldOffset(h.shoff, shSizeField),
      );
    }
    count = Number(realCount);
  } else {
    count = h.shnum;
  }

  const entrySize = h.shentsize;
  const tableSize = entrySize * BigInt(count);
  const tableRange = checkRange(
    dataLength,
    h.shoff,
    tableSize,
    'e_shoff',
    fieldOffset(0n, shoffField),
  );

  const headers: SectionHeader[] = [];
  for (let i = 0; i < count; i++) {
    const base = h.shoff + entrySize * BigInt(i);
    const fields =
      i === 0
        ? (shdr0 as unknown as RawSectionHeader)
        : (parseStruct(reader, bits, base, sLayout) as unknown as RawSectionHeader);
    const range = makeRange(base, entrySize);

    // Only sections occupying file bytes are range checked. SHT_NOBITS has
    // no file image; section 0 (SHT_NULL) carries no payload either.
    if (i !== 0 && fields.type !== SHT_NOBITS && fields.size > 0n) {
      checkRange(
        dataLength,
        fields.offset,
        fields.size,
        'sh_offset',
        fieldOffset(base, shOffsetField),
      );
    }
    headers.push({ index: i, range, fields });
  }

  // Resolve the section name string table index.
  let stringIndex: number;
  if (h.shstrndx === SHN_XINDEX) {
    if (h.shnum !== 0) {
      throw new ElfParseError(
        'e_shstrndx is SHN_XINDEX but e_shnum is not 0 (extended numbering requires both escapes)',
        'e_shstrndx',
        fieldOffset(0n, shstrndxField),
      );
    }
    const link = shdr0.link as number;    if (link >= count) {
      throw new ElfParseError(
        `extended shstrndx (section 0 sh_link=${link}) is out of range for ${count} sections`,
        'sh_link',
        fieldOffset(h.shoff, shLinkField),
      );
    }
    stringIndex = link;
  } else {
    if (h.shstrndx >= SHN_LORESERVE) {
      throw new ElfParseError(
        `unsupported reserved e_shstrndx value ${h.shstrndx}`,
        'e_shstrndx',
        fieldOffset(0n, shstrndxField),
      );
    }
    if (h.shstrndx >= count) {
      throw new ElfParseError(
        `e_shstrndx ${h.shstrndx} is out of range for ${count} sections`,
        'e_shstrndx',
        fieldOffset(0n, shstrndxField),
      );
    }
    stringIndex = h.shstrndx;
  }

  return {
    rawCount: h.shnum,
    rawStringIndex: h.shstrndx,
    count,
    entrySize,
    range: tableRange,
    headers,
    stringIndex,
  };
}

// ---------------------------------------------------------------------------
// Whole file
// ---------------------------------------------------------------------------

export function parseElf(data: Uint8Array): ParsedElf {
  const header = parseFileHeader(data);
  const programHeaders = parseProgramHeaders(data, header);
  const sectionHeaders = parseSectionHeaders(data, header);

  const layouts = ehdrFields(header.identification.bits);
  const phoffLoc = fieldOffset(0n, layouts.find((f) => f.name === 'phoff')!);
  const shoffLoc = fieldOffset(0n, layouts.find((f) => f.name === 'shoff')!);

  // Structural tables must not overlap each other or the file header.
  assertDisjoint(header.range, programHeaders.range, 'e_phoff', phoffLoc);
  assertDisjoint(header.range, sectionHeaders.range, 'e_shoff', shoffLoc);
  if (programHeaders.range.size > 0n && sectionHeaders.range.size > 0n) {
    assertDisjoint(programHeaders.range, sectionHeaders.range, 'e_phoff', phoffLoc);
  }

  return {
    dataLength: BigInt(data.byteLength),
    header,
    programHeaders,
    sectionHeaders,
  };
}
