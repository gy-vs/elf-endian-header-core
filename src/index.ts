// ELF structure parser: identification, file header, program/section headers.
// Supports ELF32/ELF64 and LSB/MSB data encoding. All file offsets and sizes
// are handled as bigint so values beyond Number.MAX_SAFE_INTEGER are exact.

export const EI_NIDENT = 16;
const EI_MAG = [0x7f, 0x45, 0x4c, 0x46] as const; // \x7fELF
const ELFCLASS32 = 1;
const ELFCLASS64 = 2;
const ELFDATA2LSB = 1;
const ELFDATA2MSB = 2;
const EV_CURRENT = 1;
const SHN_UNDEF = 0;
const SHN_XINDEX = 0xffff;
const PN_XNUM = 0xffff;

/** Absolute, half-open byte interval inside the file: [start, end). */
export interface ByteRange {
  start: bigint;
  end: bigint;
  size: bigint;
}

/** Error carrying the offending ELF field name and its absolute byte offset. */
export class ElfParseError extends Error {
  readonly field: string;
  readonly offset: bigint;

  constructor(field: string, offset: bigint | number, message: string) {
    const off = BigInt(offset);
    super(`${message} (field=${field}, offset=${off})`);
    this.name = 'ElfParseError';
    this.field = field;
    this.offset = off;
  }
}

export interface ElfIdentification {
  eiClass: 32 | 64;
  eiData: 1 | 2;
  littleEndian: boolean;
  eiVersion: number;
  osAbi: number;
  abiVersion: number;
  pad: Uint8Array;
  range: ByteRange;
}

export interface ElfFileHeader {
  bits: 32 | 64;
  littleEndian: boolean;
  type: number;
  machine: number;
  version: number;
  entry: bigint;
  programHeaderOffset: bigint;
  sectionHeaderOffset: bigint;
  flags: bigint;
  headerSize: number;
  programHeaderEntrySize: number;
  programHeaderCount: number;
  sectionHeaderEntrySize: number;
  sectionHeaderCount: number;
  sectionHeaderStringIndex: number;
  range: ByteRange;
}

export interface ProgramHeader {
  index: number;
  type: bigint;
  flags: bigint;
  offset: bigint;
  virtualAddress: bigint;
  physicalAddress: bigint;
  fileSize: bigint;
  memorySize: bigint;
  alignment: bigint;
  /** Absolute byte range of this entry in the file. */
  range: ByteRange;
}

export interface SectionHeader {
  index: number;
  name: bigint;
  type: bigint;
  flags: bigint;
  address: bigint;
  offset: bigint;
  size: bigint;
  link: bigint;
  info: bigint;
  addressAlignment: bigint;
  entrySize: bigint;
  /** Absolute byte range of this entry in the file. */
  range: ByteRange;
}

export interface ElfFile {
  identification: ElfIdentification;
  header: ElfFileHeader;
  programHeaders: ProgramHeader[];
  sectionHeaders: SectionHeader[];
  /** Effective number of program headers (extended numbering resolved). */
  programHeaderCount: bigint;
  /** Effective number of section headers (extended numbering resolved). */
  sectionHeaderCount: bigint;
  /** Effective index of the section name string table. */
  sectionNameStringIndex: bigint;
  extendedSectionNumbering: boolean;
  extendedProgramNumbering: boolean;
  ranges: {
    ehdr: ByteRange;
    phdr?: ByteRange;
    shdr?: ByteRange;
  };
}

/** @deprecated kept for backwards compatibility; use parseElf instead. */
export interface ElfHeader {
  bits: 32 | 64;
  littleEndian: boolean;
  type: number;
  machine: number;
}

type FieldWidth = 1 | 2 | 4 | 8;

interface FieldSpec {
  name: string;
  width: FieldWidth;
}

function range(start: bigint, size: bigint): ByteRange {
  return { start, end: start + size, size };
}

function overlaps(a: ByteRange, b: ByteRange): boolean {
  return a.start < b.end && b.start < a.end;
}

/**
 * Read an unsigned integer. u16 stays a number; u32/u64 are bigint.
 * Throws ElfParseError tagged with the field and absolute field offset.
 */
function readUnsigned(
  data: Uint8Array,
  offset: bigint,
  width: FieldWidth,
  littleEndian: boolean,
  field: string,
  fileSize: bigint,
): bigint {
  if (offset < 0n) {
    throw new ElfParseError(field, offset, 'negative offset');
  }
  if (offset + BigInt(width) > fileSize) {
    throw new ElfParseError(
      field,
      offset,
      `need ${width} byte(s) ending at ${offset + BigInt(width)} but file size is ${fileSize}`,
    );
  }
  const base = Number(offset); // safe: end is bounded by data.length
  let value = 0n;
  if (littleEndian) {
    for (let i = 0; i < width; i++) {
      value |= BigInt(data[base + i]) << BigInt(8 * i);
    }
  } else {
    for (let i = 0; i < width; i++) {
      value = (value << 8n) | BigInt(data[base + i]);
    }
  }
  return value;
}

// File header fields following e_ident (offset 16 onwards).
const ehdrFields: Record<32 | 64, FieldSpec[]> = {
  32: [
    { name: 'e_type', width: 2 },
    { name: 'e_machine', width: 2 },
    { name: 'e_version', width: 4 },
    { name: 'e_entry', width: 4 },
    { name: 'e_phoff', width: 4 },
    { name: 'e_shoff', width: 4 },
    { name: 'e_flags', width: 4 },
    { name: 'e_ehsize', width: 2 },
    { name: 'e_phentsize', width: 2 },
    { name: 'e_phnum', width: 2 },
    { name: 'e_shentsize', width: 2 },
    { name: 'e_shnum', width: 2 },
    { name: 'e_shstrndx', width: 2 },
  ],
  64: [
    { name: 'e_type', width: 2 },
    { name: 'e_machine', width: 2 },
    { name: 'e_version', width: 4 },
    { name: 'e_entry', width: 8 },
    { name: 'e_phoff', width: 8 },
    { name: 'e_shoff', width: 8 },
    { name: 'e_flags', width: 4 },
    { name: 'e_ehsize', width: 2 },
    { name: 'e_phentsize', width: 2 },
    { name: 'e_phnum', width: 2 },
    { name: 'e_shentsize', width: 2 },
    { name: 'e_shnum', width: 2 },
    { name: 'e_shstrndx', width: 2 },
  ],
};

const phdrFields: Record<32 | 64, FieldSpec[]> = {
  32: [
    { name: 'p_type', width: 4 },
    { name: 'p_offset', width: 4 },
    { name: 'p_vaddr', width: 4 },
    { name: 'p_paddr', width: 4 },
    { name: 'p_filesz', width: 4 },
    { name: 'p_memsz', width: 4 },
    { name: 'p_flags', width: 4 },
    { name: 'p_align', width: 4 },
  ],
  64: [
    { name: 'p_type', width: 4 },
    { name: 'p_flags', width: 4 },
    { name: 'p_offset', width: 8 },
    { name: 'p_vaddr', width: 8 },
    { name: 'p_paddr', width: 8 },
    { name: 'p_filesz', width: 8 },
    { name: 'p_memsz', width: 8 },
    { name: 'p_align', width: 8 },
  ],
};

const shdrFields: Record<32 | 64, FieldSpec[]> = {
  32: [
    { name: 'sh_name', width: 4 },
    { name: 'sh_type', width: 4 },
    { name: 'sh_flags', width: 4 },
    { name: 'sh_addr', width: 4 },
    { name: 'sh_offset', width: 4 },
    { name: 'sh_size', width: 4 },
    { name: 'sh_link', width: 4 },
    { name: 'sh_info', width: 4 },
    { name: 'sh_addralign', width: 4 },
    { name: 'sh_entsize', width: 4 },
  ],
  64: [
    { name: 'sh_name', width: 4 },
    { name: 'sh_type', width: 4 },
    { name: 'sh_flags', width: 8 },
    { name: 'sh_addr', width: 8 },
    { name: 'sh_offset', width: 8 },
    { name: 'sh_size', width: 8 },
    { name: 'sh_link', width: 4 },
    { name: 'sh_info', width: 4 },
    { name: 'sh_addralign', width: 8 },
    { name: 'sh_entsize', width: 8 },
  ],
};

/** Canonical struct sizes per class. */
export const elfSizes = {
  ehdr: { 32: 52, 64: 64 } as Record<32 | 64, number>,
  phent: { 32: 32, 64: 56 } as Record<32 | 64, number>,
  shent: { 32: 40, 64: 64 } as Record<32 | 64, number>,
};

function structSize(fields: FieldSpec[]): bigint {
  return BigInt(fields.reduce((acc, f) => acc + f.width, 0));
}

/** Absolute offset of a field within a struct beginning at base. */
function fieldOffset(fields: FieldSpec[], base: bigint, name: string): bigint {
  let rel = 0;
  for (const f of fields) {
    if (f.name === name) return base + BigInt(rel);
    rel += f.width;
  }
  throw new Error(`unknown field ${name}`);
}

type StructValues = Record<string, number | bigint>;

function decodeStruct(
  data: Uint8Array,
  start: bigint,
  fields: FieldSpec[],
  littleEndian: boolean,
  kind: string,
  fileSize: bigint,
): StructValues {
  const size = structSize(fields);
  if (start < 0n || start + size > fileSize) {
    throw new ElfParseError(
      kind,
      start < 0n ? 0n : start,
      `${kind} needs ${size} byte(s) ending at ${start + size} but file size is ${fileSize}`,
    );
  }
  const values: StructValues = {};
  let rel = 0;
  for (const f of fields) {
    const raw = readUnsigned(
      data,
      start + BigInt(rel),
      f.width,
      littleEndian,
      `${kind}.${f.name}`,
      fileSize,
    );
    values[f.name] = f.width <= 2 ? Number(raw) : raw;
    rel += f.width;
  }
  return values;
}

/** Parse only e_ident; layout and endianness for everything else come from here. */
export function parseIdentification(
  data: Uint8Array,
  options: { requireCurrentVersion?: boolean } = {},
): ElfIdentification {
  const requireCurrentVersion = options.requireCurrentVersion ?? true;
  const fileSize = BigInt(data.length);

  if (fileSize < 4n) {
    throw new ElfParseError(
      'e_ident[EI_MAG0]',
      fileSize,
      'file too short for 4-byte ELF magic',
    );
  }
  for (let i = 0; i < 4; i++) {
    if (data[i] !== EI_MAG[i]) {
      throw new ElfParseError(
        `e_ident[EI_MAG${i}]`,
        i,
        `bad magic byte 0x${data[i].toString(16).padStart(2, '0')}, expected 0x${EI_MAG[i].toString(16)}`,
      );
    }
  }

  if (fileSize < BigInt(EI_NIDENT)) {
    throw new ElfParseError(
      'e_ident[EI_CLASS]',
      4,
      `e_ident truncated: need ${EI_NIDENT} bytes but file size is ${fileSize}`,
    );
  }

  let bits: 32 | 64;
  switch (data[4]) {
    case ELFCLASS32:
      bits = 32;
      break;
    case ELFCLASS64:
      bits = 64;
      break;
    default:
      throw new ElfParseError(
        'e_ident[EI_CLASS]',
        4,
        `invalid ELF class ${data[4]} (expected 1=ELFCLASS32 or 2=ELFCLASS64)`,
      );
  }

  let littleEndian: boolean;
  switch (data[5]) {
    case ELFDATA2LSB:
      littleEndian = true;
      break;
    case ELFDATA2MSB:
      littleEndian = false;
      break;
    default:
      throw new ElfParseError(
        'e_ident[EI_DATA]',
        5,
        `invalid ELF data encoding ${data[5]} (expected 1=2's complement little-endian or 2=big-endian)`,
      );
  }

  if (requireCurrentVersion && data[6] !== EV_CURRENT) {
    throw new ElfParseError(
      'e_ident[EI_VERSION]',
      6,
      `invalid ELF version ${data[6]}, expected ${EV_CURRENT} (EV_CURRENT)`,
    );
  }

  return {
    eiClass: bits,
    eiData: littleEndian ? 1 : 2,
    littleEndian,
    eiVersion: data[6],
    osAbi: data[7],
    abiVersion: data[8],
    pad: data.slice(9, EI_NIDENT),
    range: range(0n, BigInt(EI_NIDENT)),
  };
}

/** Parse the whole ELF: identification, file header, program/section tables. */
export function parseElf(data: Uint8Array): ElfFile {
  const fileSize = BigInt(data.length);
  const ident = parseIdentification(data);
  const bits = ident.eiClass;
  const little = ident.littleEndian;
  const ehFields = ehdrFields[bits];
  const phFields = phdrFields[bits];
  const shFields = shdrFields[bits];
  const canonicalEhdrSize = BigInt(elfSizes.ehdr[bits]);
  const canonicalPhSize = BigInt(elfSizes.phent[bits]);
  const canonicalShSize = BigInt(elfSizes.shent[bits]);

  // The whole file header must be present before any of its fields are trusted.
  if (fileSize < canonicalEhdrSize) {
    throw new ElfParseError(
      'Ehdr',
      0,
      `ELF${bits} file header truncated: need ${canonicalEhdrSize} bytes but file size is ${fileSize}`,
    );
  }

  const h = decodeStruct(data, BigInt(EI_NIDENT), ehFields, little, 'Ehdr', fileSize);
  const ehdrRange = range(0n, canonicalEhdrSize);

  const headerSize = h.e_ehsize as number;
  if (BigInt(headerSize) !== canonicalEhdrSize) {
    throw new ElfParseError(
      'e_ehsize',
      fieldOffset(ehFields, BigInt(EI_NIDENT), 'e_ehsize'),
      `e_ehsize is ${headerSize} but ELF${bits} file header must be ${canonicalEhdrSize} bytes`,
    );
  }

  const header: ElfFileHeader = {
    bits,
    littleEndian: little,
    type: h.e_type as number,
    machine: h.e_machine as number,
    version: Number(h.e_version),
    entry: h.e_entry as bigint,
    programHeaderOffset: h.e_phoff as bigint,
    sectionHeaderOffset: h.e_shoff as bigint,
    flags: h.e_flags as bigint,
    headerSize,
    programHeaderEntrySize: h.e_phentsize as number,
    programHeaderCount: h.e_phnum as number,
    sectionHeaderEntrySize: h.e_shentsize as number,
    sectionHeaderCount: h.e_shnum as number,
    sectionHeaderStringIndex: h.e_shstrndx as number,
    range: ehdrRange,
  };

  // ---- Phase 1: resolve table counts and full ranges (bigint throughout).
  // In extended section numbering this reads section 0 only; ordinary 16-bit
  // Ehdr fields are never trusted for the counts.
  const shoff = header.sectionHeaderOffset;
  const rawShnum = header.sectionHeaderCount;
  const rawShentsize = BigInt(header.sectionHeaderEntrySize);
  const rawShstrndx = header.sectionHeaderStringIndex;

  let sectionCount: bigint;
  let sectionNameStringIndex: bigint;
  let shdrRange: ByteRange | undefined;
  let shZero: StructValues | undefined;
  let extendedSectionNumbering = false;

  if (shoff === 0n) {
    // No section header table: the count/index fields must not claim one, and
    // extended numbering cannot be in effect (it requires section 0).
    if (rawShnum !== SHN_UNDEF) {
      throw new ElfParseError(
        'e_shnum',
        fieldOffset(ehFields, BigInt(EI_NIDENT), 'e_shnum'),
        `e_shoff is 0 (no section table) but e_shnum is ${rawShnum}`,
      );
    }
    if (rawShstrndx !== SHN_UNDEF) {
      throw new ElfParseError(
        'e_shstrndx',
        fieldOffset(ehFields, BigInt(EI_NIDENT), 'e_shstrndx'),
        `e_shoff is 0 (no section table) but e_shstrndx is ${rawShstrndx}`,
      );
    }
    sectionCount = 0n;
    sectionNameStringIndex = 0n;
  } else {
    if (rawShentsize !== canonicalShSize) {
      throw new ElfParseError(
        'e_shentsize',
        fieldOffset(ehFields, BigInt(EI_NIDENT), 'e_shentsize'),
        `e_shentsize is ${rawShentsize} but ELF${bits} section header entry must be ${canonicalShSize} bytes`,
      );
    }

    // Before section 0 is read, its single-entry span must not alias Ehdr.
    if (overlaps(ehdrRange, range(shoff, canonicalShSize))) {
      throw new ElfParseError(
        'e_shoff',
        shoff,
        `section header table at [${shoff}, ${shoff + canonicalShSize}) overlaps file header ${fmtRange(ehdrRange)}`,
      );
    }

    if (rawShnum === SHN_UNDEF) {
      // Extended section numbering: real counts live in section 0.
      extendedSectionNumbering = true;
      shZero = decodeStruct(data, shoff, shFields, little, 'Shdr[0]', fileSize);
      sectionCount = shZero.sh_size as bigint;
      if (sectionCount === 0n) {
        throw new ElfParseError(
          'sh_size',
          fieldOffset(shFields, shoff, 'sh_size'),
          'e_shnum is SHN_UNDEF so section 0 sh_size must hold the real section count, but it is 0',
        );
      }
    } else {
      sectionCount = BigInt(rawShnum);
    }

    const tableEnd = shoff + canonicalShSize * sectionCount;
    if (tableEnd > fileSize) {
      throw new ElfParseError(
        'e_shoff',
        shoff,
        `section header table for ${sectionCount} entr${sectionCount === 1n ? 'y' : 'ies'} ends at ${tableEnd} beyond file size ${fileSize}`,
      );
    }
    shdrRange = range(shoff, canonicalShSize * sectionCount);

    if (rawShstrndx === SHN_UNDEF && !extendedSectionNumbering) {
      sectionNameStringIndex = 0n;
    } else if (rawShstrndx === SHN_XINDEX) {
      // Real name-string index lives in section 0's sh_link.
      sectionNameStringIndex = (shZero ?? decodeStruct(data, shoff, shFields, little, 'Shdr[0]', fileSize)).sh_link as bigint;
      if (sectionNameStringIndex >= sectionCount) {
        throw new ElfParseError(
          'sh_link',
          fieldOffset(shFields, shoff, 'sh_link'),
          `e_shstrndx is SHN_XINDEX; section 0 sh_link=${sectionNameStringIndex} is not a valid section index (< ${sectionCount})`,
        );
      }
    } else {
      sectionNameStringIndex = BigInt(rawShstrndx);
      if (sectionNameStringIndex >= sectionCount) {
        throw new ElfParseError(
          'e_shstrndx',
          fieldOffset(ehFields, BigInt(EI_NIDENT), 'e_shstrndx'),
          `e_shstrndx=${sectionNameStringIndex} is not a valid section index (< ${sectionCount})`,
        );
      }
    }
  }

  // Program header count; PN_XNUM means the real count is section 0 sh_info.
  const phoff = header.programHeaderOffset;
  const rawPhnum = header.programHeaderCount;
  const rawPhentsize = BigInt(header.programHeaderEntrySize);
  let programCount: bigint;
  let phdrRange: ByteRange | undefined;
  let extendedProgramNumbering = false;

  if (rawPhnum === PN_XNUM) {
    // Real count lives in sh_info of section header 0 (whether or not
    // extended section numbering is also in effect).
    if (shdrRange === undefined) {
      throw new ElfParseError(
        'e_phnum',
        fieldOffset(ehFields, BigInt(EI_NIDENT), 'e_phnum'),
        'e_phnum is PN_XNUM but section header 0 (sh_info) is unavailable',
      );
    }
    if (shZero === undefined) {
      shZero = decodeStruct(data, shoff, shFields, little, 'Shdr[0]', fileSize);
    }
    extendedProgramNumbering = true;
    programCount = shZero.sh_info as bigint;
    if (programCount === 0n) {
      throw new ElfParseError(
        'sh_info',
        fieldOffset(shFields, shoff, 'sh_info'),
        'e_phnum is PN_XNUM so section 0 sh_info must hold the real program header count, but it is 0',
      );
    }
  } else {
    programCount = BigInt(rawPhnum);
  }

  if (phoff === 0n) {
    if (programCount !== 0n) {
      throw new ElfParseError(
        'e_phnum',
        fieldOffset(ehFields, BigInt(EI_NIDENT), 'e_phnum'),
        `e_phoff is 0 (no program table) but effective e_phnum is ${programCount}`,
      );
    }
  } else {
    if (rawPhentsize !== canonicalPhSize) {
      throw new ElfParseError(
        'e_phentsize',
        fieldOffset(ehFields, BigInt(EI_NIDENT), 'e_phentsize'),
        `e_phentsize is ${rawPhentsize} but ELF${bits} program header entry must be ${canonicalPhSize} bytes`,
      );
    }
    const tableEnd = phoff + canonicalPhSize * programCount;
    if (tableEnd > fileSize) {
      throw new ElfParseError(
        'e_phoff',
        phoff,
        `program header table for ${programCount} entr${programCount === 1n ? 'y' : 'ies'} ends at ${tableEnd} beyond file size ${fileSize}`,
      );
    }
    phdrRange = range(phoff, canonicalPhSize * programCount);
  }

  // ---- Phase 2: structural ranges (Ehdr and the two tables) must not overlap.
  if (phdrRange && overlaps(ehdrRange, phdrRange)) {
    throw new ElfParseError(
      'e_phoff',
      phoff,
      `program header table ${fmtRange(phdrRange)} overlaps file header ${fmtRange(ehdrRange)}`,
    );
  }
  if (shdrRange && overlaps(ehdrRange, shdrRange)) {
    throw new ElfParseError(
      'e_shoff',
      shoff,
      `section header table ${fmtRange(shdrRange)} overlaps file header ${fmtRange(ehdrRange)}`,
    );
  }
  if (phdrRange && shdrRange && overlaps(phdrRange, shdrRange)) {
    throw new ElfParseError(
      'e_shoff',
      shoff,
      `section header table ${fmtRange(shdrRange)} overlaps program header table ${fmtRange(phdrRange)}`,
    );
  }

  // ---- Phase 3: ranges are proven disjoint and in-bounds; decode entries.
  const sectionHeaders: SectionHeader[] = [];
  if (shdrRange) {
    const count = Number(sectionCount); // safe: table is bounded by data.length
    for (let i = 0; i < count; i++) {
      const start = shoff + canonicalShSize * BigInt(i);
      const v = i === 0 && shZero ? shZero : decodeStruct(data, start, shFields, little, `Shdr[${i}]`, fileSize);
      sectionHeaders.push({
        index: i,
        name: v.sh_name as bigint,
        type: v.sh_type as bigint,
        flags: v.sh_flags as bigint,
        address: v.sh_addr as bigint,
        offset: v.sh_offset as bigint,
        size: v.sh_size as bigint,
        link: v.sh_link as bigint,
        info: v.sh_info as bigint,
        addressAlignment: v.sh_addralign as bigint,
        entrySize: v.sh_entsize as bigint,
        range: range(start, canonicalShSize),
      });
    }
  }

  const programHeaders: ProgramHeader[] = [];
  if (phdrRange) {
    const count = Number(programCount); // safe: table is bounded by data.length
    for (let i = 0; i < count; i++) {
      const start = phoff + canonicalPhSize * BigInt(i);
      const v = decodeStruct(data, start, phFields, little, `Phdr[${i}]`, fileSize);
      programHeaders.push({
        index: i,
        type: v.p_type as bigint,
        flags: v.p_flags as bigint,
        offset: v.p_offset as bigint,
        virtualAddress: v.p_vaddr as bigint,
        physicalAddress: v.p_paddr as bigint,
        fileSize: v.p_filesz as bigint,
        memorySize: v.p_memsz as bigint,
        alignment: v.p_align as bigint,
        range: range(start, canonicalPhSize),
      });
    }
  }

  return {
    identification: ident,
    header,
    programHeaders,
    sectionHeaders,
    programHeaderCount: programCount,
    sectionHeaderCount: sectionCount,
    sectionNameStringIndex,
    extendedSectionNumbering,
    extendedProgramNumbering,
    ranges: {
      ehdr: ehdrRange,
      phdr: phdrRange,
      shdr: shdrRange,
    },
  };
}

function fmtRange(r: ByteRange): string {
  return `[${r.start}, ${r.end})`;
}

/**
 * Lightweight legacy parse: class/endianness plus e_type and e_machine only.
 * Prefer parseElf for full validation.
 */
export function parseHeader(data: Uint8Array): ElfHeader {
  if (data.length < 20) {
    throw new ElfParseError(
      'Ehdr',
      BigInt(data.length),
      `need at least 20 bytes for e_type/e_machine but file size is ${data.length}`,
    );
  }
  const ident = parseIdentification(data, { requireCurrentVersion: false });
  const fileSize = BigInt(data.length);
  const type = Number(
    readUnsigned(data, 16n, 2, ident.littleEndian, 'e_type', fileSize),
  );
  const machine = Number(
    readUnsigned(data, 18n, 2, ident.littleEndian, 'e_machine', fileSize),
  );
  return { bits: ident.eiClass, littleEndian: ident.littleEndian, type, machine };
}
