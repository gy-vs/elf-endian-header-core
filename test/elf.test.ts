import { describe, expect, it } from 'vitest';
import {
  ElfParseError,
  elfSizes,
  parseElf,
  parseHeader,
  parseIdentification,
} from '../src/index.js';

type Bits = 32 | 64;

interface FieldSpec {
  name: string;
  width: 1 | 2 | 4 | 8;
}

const ehFields: Record<Bits, FieldSpec[]> = {
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

const phFields: Record<Bits, FieldSpec[]> = {
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

const shFields: Record<Bits, FieldSpec[]> = {
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

type Entry = Record<string, bigint | number>;

interface BuildOpts {
  bits?: Bits;
  littleEndian?: boolean;
  osAbi?: number;
  abiVersion?: number;
  ehdr?: Partial<{
    type: number;
    machine: number;
    version: number;
    entry: bigint;
    phoff: bigint;
    shoff: bigint;
    flags: bigint;
    ehsize: number;
    phentsize: number;
    phnum: number;
    shentsize: number;
    shnum: number;
    shstrndx: number;
  }>;
  phdrs?: Entry[];
  shdrs?: Entry[];
  extra?: number;
  truncate?: number;
  patches?: Array<[number | bigint, number[]]>;
}

function writeValue(
  buf: Uint8Array,
  offset: number,
  width: number,
  raw: bigint | number,
  little: boolean,
): void {
  let value = BigInt(raw);
  if (width < 8) value &= (1n << BigInt(width * 8)) - 1n;
  for (let i = 0; i < width; i++) {
    const byte = Number((value >> BigInt(8 * (little ? i : width - 1 - i))) & 0xffn);
    buf[offset + i] = byte;
  }
}

function writeStruct(
  buf: Uint8Array,
  start: bigint,
  fields: FieldSpec[],
  values: Entry,
  little: boolean,
): void {
  let rel = 0;
  for (const f of fields) {
    writeValue(buf, Number(start) + rel, f.width, values[f.name] ?? 0, little);
    rel += f.width;
  }
}

function fieldAbsOffset(fields: FieldSpec[], base: bigint, name: string): bigint {
  let rel = 0;
  for (const f of fields) {
    if (f.name === name) return base + BigInt(rel);
    rel += f.width;
  }
  throw new Error(`unknown field ${name}`);
}

function buildElf(opts: BuildOpts = {}): Uint8Array {
  const bits = opts.bits ?? 32;
  const little = opts.littleEndian ?? true;
  const ehSize = elfSizes.ehdr[bits];
  const phEnt = elfSizes.phent[bits];
  const shEnt = elfSizes.shent[bits];
  const phdrs = opts.phdrs ?? [];
  const shdrs = opts.shdrs ?? [];

  let phoff = opts.ehdr?.phoff;
  if (phoff === undefined) phoff = phdrs.length ? BigInt(ehSize) : 0n;
  let shoff = opts.ehdr?.shoff;
  if (shoff === undefined) {
    shoff = shdrs.length
      ? phdrs.length
        ? phoff + BigInt(phEnt * phdrs.length)
        : BigInt(ehSize)
      : 0n;
  }

  const phnum = opts.ehdr?.phnum ?? phdrs.length;
  const shnum = opts.ehdr?.shnum ?? shdrs.length;
  const ehdrValues: Entry = {
    e_type: opts.ehdr?.type ?? 2,
    e_machine: opts.ehdr?.machine ?? 62,
    e_version: opts.ehdr?.version ?? 1,
    e_entry: opts.ehdr?.entry ?? 0x1000n,
    e_phoff: phoff,
    e_shoff: shoff,
    e_flags: opts.ehdr?.flags ?? 0n,
    e_ehsize: opts.ehdr?.ehsize ?? ehSize,
    e_phentsize: opts.ehdr?.phentsize ?? phEnt,
    e_phnum: phnum,
    e_shentsize: opts.ehdr?.shentsize ?? shEnt,
    e_shnum: shnum,
    e_shstrndx: opts.ehdr?.shstrndx ?? 0,
  };

  // Only structs actually written extend the allocation; a table offset that
  // points past the buffer with zero entries is just bytes we never touch.
  const candidates = [BigInt(ehSize)];
  if (phdrs.length) candidates.push(phoff + BigInt(phEnt * phdrs.length));
  if (shdrs.length) candidates.push(shoff + BigInt(shEnt * shdrs.length));
  const end = candidates.reduce((a, b) => (a > b ? a : b)) + BigInt(opts.extra ?? 0);

  const buf = new Uint8Array(Number(end));
  // Write tables first and Ehdr last: in deliberate-overlap tests the table
  // aliases the header bytes, and the parser must reject that on the still
  // readable header (the corrupted table entries are never decoded).
  phdrs.forEach((p, i) => writeStruct(buf, phoff + BigInt(phEnt * i), phFields[bits], p, little));
  shdrs.forEach((s, i) => writeStruct(buf, shoff + BigInt(shEnt * i), shFields[bits], s, little));
  // e_ident + rest of Ehdr
  buf.set([0x7f, 0x45, 0x4c, 0x46, bits === 32 ? 1 : 2, little ? 1 : 2, 1, opts.osAbi ?? 0, opts.abiVersion ?? 0]);
  writeStruct(buf, 16n, ehFields[bits], ehdrValues, little);

  for (const [at, bytes] of opts.patches ?? []) {
    buf.set(bytes, Number(at));
  }
  if (opts.truncate !== undefined) return buf.slice(0, opts.truncate);
  return buf;
}

function expectParseError(
  fn: () => unknown,
  field: string,
  offset: bigint | number,
): ElfParseError {
  try {
    fn();
  } catch (e) {
    expect(e).toBeInstanceOf(ElfParseError);
    const err = e as ElfParseError;
    expect(err.field).toBe(field);
    expect(err.offset).toBe(BigInt(offset));
    return err;
  }
  throw new Error('expected ElfParseError');
}

const layouts: Array<{ bits: Bits; little: boolean; tag: string }> = [
  { bits: 32, little: true, tag: 'ELF32 LSB' },
  { bits: 32, little: false, tag: 'ELF32 MSB' },
  { bits: 64, little: true, tag: 'ELF64 LSB' },
  { bits: 64, little: false, tag: 'ELF64 MSB' },
];

describe('four layouts: 32/64 x LSB/MSB', () => {
  it.each(layouts)('parses identification, Ehdr, Phdrs, Shdrs — $tag', ({ bits, little }) => {
    const ehSize = elfSizes.ehdr[bits];
    const phEnt = elfSizes.phent[bits];
    const shEnt = elfSizes.shent[bits];
    const bigValue = bits === 64 ? 0x123456789abcdef0n : 0xdeadbeefn;

    const elf = buildElf({
      bits,
      littleEndian: little,
      osAbi: 3,
      abiVersion: 7,
      ehdr: {
        type: 2,
        machine: bits === 64 ? 62 : 3,
        version: 1,
        entry: bigValue,
        flags: 0x01020304n,
      },
      phdrs: [
        {
          p_type: 1n,
          p_flags: 5n,
          p_offset: 0x1000n,
          p_vaddr: bigValue,
          p_paddr: 0x2000n,
          p_filesz: 0x300n,
          p_memsz: 0x400n,
          p_align: 0x1000n,
        },
        {
          p_type: 2n,
          p_flags: 6n,
          p_offset: 0x5000n,
          p_vaddr: 0x6000n,
          p_paddr: 0x6000n,
          p_filesz: 0x100n,
          p_memsz: 0x200n,
          p_align: 0x1000n,
        },
      ],
      shdrs: [
        { sh_name: 0n, sh_type: 0n },
        { sh_name: 11n, sh_type: 1n, sh_flags: 2n, sh_addr: bigValue, sh_offset: 0x7000n, sh_size: 0x50n },
        { sh_name: 22n, sh_type: 3n, sh_flags: 0x01020304n, sh_offset: 0x8000n, sh_size: 0x60n, sh_link: 1n, sh_info: 4n },
      ],
    });

    const f = parseElf(elf);

    // Identification
    expect(f.identification.eiClass).toBe(bits);
    expect(f.identification.littleEndian).toBe(little);
    expect(f.identification.eiData).toBe(little ? 1 : 2);
    expect(f.identification.osAbi).toBe(3);
    expect(f.identification.abiVersion).toBe(7);
    expect(f.identification.range).toEqual({ start: 0n, end: 16n, size: 16n });

    // File header
    expect(f.header.type).toBe(2);
    expect(f.header.machine).toBe(bits === 64 ? 62 : 3);
    expect(f.header.version).toBe(1);
    expect(f.header.entry).toBe(bigValue);
    expect(f.header.flags).toBe(0x01020304n);
    expect(f.header.headerSize).toBe(ehSize);
    expect(f.header.range).toEqual({ start: 0n, end: BigInt(ehSize), size: BigInt(ehSize) });

    // Endianness is actually honoured: raw bytes of e_flags at its file offset
    const flagsOff = fieldAbsOffset(ehFields[bits], 16n, 'e_flags');
    const expectedFlagsBytes = little
      ? [0x04, 0x03, 0x02, 0x01]
      : [0x01, 0x02, 0x03, 0x04];
    expect([...elf.slice(Number(flagsOff), Number(flagsOff) + 4)]).toEqual(expectedFlagsBytes);

    // Program headers
    expect(f.programHeaderCount).toBe(2n);
    expect(f.programHeaders).toHaveLength(2);
    const phBase = BigInt(ehSize);
    expect(f.ranges.phdr).toEqual({
      start: phBase,
      end: phBase + BigInt(phEnt * 2),
      size: BigInt(phEnt * 2),
    });
    expect(f.programHeaders[0]).toMatchObject({
      index: 0,
      type: 1n,
      flags: 5n,
      offset: 0x1000n,
      virtualAddress: bigValue,
      physicalAddress: 0x2000n,
      fileSize: 0x300n,
      memorySize: 0x400n,
      alignment: 0x1000n,
    });
    expect(f.programHeaders[0].range).toEqual({
      start: phBase,
      end: phBase + BigInt(phEnt),
      size: BigInt(phEnt),
    });
    expect(f.programHeaders[1].range.start).toBe(phBase + BigInt(phEnt));

    // Section headers
    expect(f.sectionHeaderCount).toBe(3n);
    expect(f.sectionHeaders).toHaveLength(3);
    const shBase = phBase + BigInt(phEnt * 2);
    expect(f.ranges.shdr?.start).toBe(shBase);
    expect(f.sectionHeaders[1]).toMatchObject({
      index: 1,
      name: 11n,
      type: 1n,
      flags: 2n,
      address: bigValue,
      offset: 0x7000n,
      size: 0x50n,
    });
    expect(f.sectionHeaders[2].link).toBe(1n);
    expect(f.sectionHeaders[2].info).toBe(4n);
    expect(f.sectionHeaders[2].range.start).toBe(shBase + BigInt(shEnt * 2));
    expect(f.sectionNameStringIndex).toBe(0n);
    expect(f.extendedSectionNumbering).toBe(false);
    expect(f.extendedProgramNumbering).toBe(false);
  });

  it('high-bit u32 values decode unsigned in both endians', () => {
    for (const { bits, little } of layouts) {
      const elf = buildElf({
        bits,
        littleEndian: little,
        shdrs: [{ sh_name: 0xffffffffn }],
      });
      const f = parseElf(elf);
      expect(f.sectionHeaders[0].name).toBe(0xffffffffn);
    }
  });
});

describe('extended numbering', () => {
  it.each(layouts)(
    'reads real shnum/shstrndx/phnum from section 0 — $tag',
    ({ bits, little }) => {
      const elf = buildElf({
        bits,
        littleEndian: little,
        ehdr: { shnum: 0, shstrndx: 0xffff, phnum: 0xffff },
        phdrs: [{ p_type: 1n }],
        shdrs: [
          { sh_type: 0n, sh_size: 3n, sh_link: 2n, sh_info: 1n },
          { sh_type: 1n },
          { sh_type: 2n },
        ],
      });
      const f = parseElf(elf);
      expect(f.extendedSectionNumbering).toBe(true);
      expect(f.extendedProgramNumbering).toBe(true);
      expect(f.sectionHeaderCount).toBe(3n);
      expect(f.sectionNameStringIndex).toBe(2n);
      expect(f.programHeaderCount).toBe(1n);
      // The ordinary 16-bit fields were not trusted.
      expect(f.header.sectionHeaderCount).toBe(0);
      expect(f.header.sectionHeaderStringIndex).toBe(0xffff);
      expect(f.header.programHeaderCount).toBe(0xffff);
      expect(f.sectionHeaders).toHaveLength(3);
      expect(f.programHeaders).toHaveLength(1);
    },
  );

  it('extended shnum with section 0 sh_size=0', () => {
    const bits = 32;
    const elf = buildElf({
      bits,
      ehdr: { shnum: 0 },
      shdrs: [{ sh_size: 0n }],
    });
    const shoff = BigInt(elfSizes.ehdr[bits]);
    expectParseError(
      () => parseElf(elf),
      'sh_size',
      fieldAbsOffset(shFields[bits], shoff, 'sh_size'),
    );
  });

  it('extended shstrndx with section 0 sh_link out of range', () => {
    const bits = 64;
    const elf = buildElf({
      bits,
      ehdr: { shnum: 0, shstrndx: 0xffff },
      shdrs: [
        { sh_size: 2n, sh_link: 99n },
        { sh_type: 1n },
      ],
    });
    const shoff = BigInt(elfSizes.ehdr[bits]);
    const err = expectParseError(
      () => parseElf(elf),
      'sh_link',
      fieldAbsOffset(shFields[bits], shoff, 'sh_link'),
    );
    expect(err.message).toContain('99');
  });

  it('PN_XNUM without a section table', () => {
    const elf = buildElf({
      bits: 32,
      ehdr: { phnum: 0xffff },
    });
    expectParseError(
      () => parseElf(elf),
      'e_phnum',
      fieldAbsOffset(ehFields[32], 16n, 'e_phnum'),
    );
  });

  it('PN_XNUM with section 0 sh_info=0', () => {
    const bits = 32;
    const elf = buildElf({
      bits,
      ehdr: { phnum: 0xffff },
      shdrs: [{ sh_size: 1n, sh_info: 0n }],
    });
    const shoff = BigInt(elfSizes.ehdr[bits]);
    expectParseError(
      () => parseElf(elf),
      'sh_info',
      fieldAbsOffset(shFields[bits], shoff, 'sh_info'),
    );
  });

  it('e_shnum=0 forces extended mode even without SHN_XINDEX', () => {
    // e_shnum=0 means section 0 holds the count; a non-table offset here is fatal,
    // proving section 0 (not the ordinary field) is consulted first.
    const bits = 32;
    const elf = buildElf({
      bits,
      ehdr: { shnum: 0, shoff: 0x100000n },
    });
    expectParseError(() => parseElf(elf), 'Shdr[0]', 0x100000n);
  });

  it('huge extended count in sh_size keeps exact bigint in the error', () => {
    const bits = 64;
    const hugeCount = 1n << 60n;
    const elf = buildElf({
      bits,
      ehdr: { shnum: 0, shstrndx: 0 },
      shdrs: [{ sh_size: hugeCount }],
    });
    const shoff = BigInt(elfSizes.ehdr[bits]);
    const err = expectParseError(() => parseElf(elf), 'e_shoff', shoff);
    expect(err.message).toContain(hugeCount.toString());
    expect(err.message).toContain((shoff + 64n * hugeCount).toString());
  });

  it('huge ordinary program count in 64-bit ELF stays exact', () => {
    const bits = 64;
    const elf = buildElf({
      bits,
      ehdr: { phnum: 0x100, phoff: 1n << 60n },
    });
    expectParseError(() => parseElf(elf), 'e_phoff', 1n << 60n);
  });
});

describe('no section table', () => {
  it('accepts e_shoff=0 with zeroed counts and parses program headers', () => {
    const elf = buildElf({
      bits: 64,
      ehdr: { shnum: 0, shstrndx: 0 },
      phdrs: [{ p_type: 1n }, { p_type: 2n }],
    });
    const f = parseElf(elf);
    expect(f.ranges.shdr).toBeUndefined();
    expect(f.sectionHeaders).toEqual([]);
    expect(f.sectionHeaderCount).toBe(0n);
    expect(f.sectionNameStringIndex).toBe(0n);
    expect(f.programHeaderCount).toBe(2n);
  });

  it('rejects nonzero e_shnum when e_shoff=0', () => {
    const elf = buildElf({
      bits: 32,
      ehdr: { shoff: 0, shnum: 3, shstrndx: 0 },
    });
    expectParseError(
      () => parseElf(elf),
      'e_shnum',
      fieldAbsOffset(ehFields[32], 16n, 'e_shnum'),
    );
  });

  it('rejects nonzero e_shstrndx when e_shoff=0', () => {
    const elf = buildElf({
      bits: 32,
      ehdr: { shoff: 0, shnum: 0, shstrndx: 1 },
    });
    expectParseError(
      () => parseElf(elf),
      'e_shstrndx',
      fieldAbsOffset(ehFields[32], 16n, 'e_shstrndx'),
    );
  });

  it('rejects nonzero phnum when e_phoff=0', () => {
    const elf = buildElf({
      bits: 32,
      ehdr: { phoff: 0n, phnum: 1, shoff: 0, shnum: 0, shstrndx: 0 },
    });
    expectParseError(
      () => parseElf(elf),
      'e_phnum',
      fieldAbsOffset(ehFields[32], 16n, 'e_phnum'),
    );
  });
});

describe('truncated input', () => {
  it('file shorter than magic', () => {
    const err = expectParseError(() => parseElf(new Uint8Array(3)), 'e_ident[EI_MAG0]', 3n);
    expect(err.message).toContain('magic');
  });

  it('file shorter than e_ident', () => {
    expectParseError(
      () => parseElf(new Uint8Array([0x7f, 0x45, 0x4c, 0x46, 1, 1, 1, 0, 0, 0])),
      'e_ident[EI_CLASS]',
      4n,
    );
  });

  it('file header truncated', () => {
    const elf = buildElf({ bits: 64 });
    expectParseError(() => parseElf(elf.slice(0, 40)), 'Ehdr', 0n);
  });

  it('section table truncated', () => {
    const bits = 64;
    const elf = buildElf({
      bits,
      shdrs: [{ sh_type: 1n }, { sh_type: 2n }],
      truncate: elfSizes.ehdr[bits] + elfSizes.shent[bits] * 2 - 8,
    });
    const shoff = BigInt(elfSizes.ehdr[bits]);
    const err = expectParseError(() => parseElf(elf), 'e_shoff', shoff);
    expect(err.message).toContain('beyond file size');
  });

  it('program table truncated', () => {
    const bits = 32;
    const elf = buildElf({
      bits,
      phdrs: [{ p_type: 1n }, { p_type: 2n }],
      truncate: elfSizes.ehdr[bits] + elfSizes.phent[bits] * 2 - 4,
    });
    expectParseError(() => parseElf(elf), 'e_phoff', BigInt(elfSizes.ehdr[bits]));
  });

  it('section 0 itself truncated in extended mode', () => {
    const bits = 32;
    const shoff = 200n;
    const elf = buildElf({
      bits,
      ehdr: { shnum: 0, shoff },
      truncate: Number(shoff) + 20,
    });
    expectParseError(() => parseElf(elf), 'Shdr[0]', shoff);
  });
});

describe('wrong entry sizes and e_ehsize', () => {
  it('rejects wrong e_ehsize', () => {
    const bits = 32;
    const elf = buildElf({ bits, ehdr: { ehsize: 40 } });
    expectParseError(
      () => parseElf(elf),
      'e_ehsize',
      fieldAbsOffset(ehFields[bits], 16n, 'e_ehsize'),
    );
  });

  it('rejects wrong e_shentsize', () => {
    const bits = 32;
    const elf = buildElf({
      bits,
      ehdr: { shentsize: 41 },
      shdrs: [{ sh_type: 1n }],
    });
    expectParseError(
      () => parseElf(elf),
      'e_shentsize',
      fieldAbsOffset(ehFields[bits], 16n, 'e_shentsize'),
    );
  });

  it('rejects wrong e_phentsize (64-bit)', () => {
    const bits = 64;
    const elf = buildElf({
      bits,
      ehdr: { phentsize: 48 },
      phdrs: [{ p_type: 1n }],
    });
    expectParseError(
      () => parseElf(elf),
      'e_phentsize',
      fieldAbsOffset(ehFields[bits], 16n, 'e_phentsize'),
    );
  });
});

describe('overlapping ranges', () => {
  it('rejects section table overlapping program table', () => {
    const bits = 32;
    const phoff = BigInt(elfSizes.ehdr[bits]);
    const shoff = phoff + 16n; // ph table of 2 entries spans [52, 116)
    const elf = buildElf({
      bits,
      ehdr: { phoff, shoff },
      phdrs: [{ p_type: 1n }, { p_type: 2n }],
      shdrs: [{ sh_type: 1n }],
    });
    const err = expectParseError(() => parseElf(elf), 'e_shoff', shoff);
    expect(err.message).toContain('overlap');
  });

  it('rejects section table overlapping the file header', () => {
    const bits = 64;
    const shoff = 8n;
    const elf = buildElf({
      bits,
      ehdr: { shoff },
      shdrs: [{ sh_type: 1n }],
    });
    expectParseError(() => parseElf(elf), 'e_shoff', shoff);
  });

  it('rejects program table overlapping the file header', () => {
    const bits = 32;
    const phoff = 16n;
    const elf = buildElf({
      bits,
      ehdr: { phoff },
      phdrs: [{ p_type: 1n }, { p_type: 2n }],
    });
    expectParseError(() => parseElf(elf), 'e_phoff', phoff);
  });
});

describe('offsets beyond Number.MAX_SAFE_INTEGER', () => {
  it('reports the exact huge offset when out of range', () => {
    // 2^60 is well beyond 2^53-1.
    const off = 1n << 60n;
    const elf = buildElf({
      bits: 64,
      ehdr: { shoff: off, shnum: 1 },
    });
    const err = expectParseError(() => parseElf(elf), 'e_shoff', off);
    expect(err.offset).toBeGreaterThan(BigInt(Number.MAX_SAFE_INTEGER));
    expect(err.message).toContain(off.toString());
  });

  it('count * entsize overflow uses bigint and stays exact', () => {
    // shnum=0xffff, entsize=64 => product ~4.19M but combined with a huge
    // shoff the failing end must be an exact bigint.
    const off = 1n << 60n;
    const elf = buildElf({
      bits: 64,
      ehdr: { shoff: off, shnum: 0xffff },
    });
    const err = expectParseError(() => parseElf(elf), 'e_shoff', off);
    expect(err.message).toContain((off + 64n * 0xffffn).toString());
  });

  it('preserves huge u64 entry offsets exactly on success', () => {
    const hugeOffset = 2n ** 62n + 7n;
    const elf = buildElf({
      bits: 64,
      phdrs: [{ p_type: 1n, p_offset: hugeOffset }],
      shdrs: [{ sh_type: 1n, sh_offset: hugeOffset + 1n }],
    });
    const f = parseElf(elf);
    expect(f.programHeaders[0].offset).toBe(hugeOffset);
    expect(f.sectionHeaders[0].offset).toBe(hugeOffset + 1n);
    expect(Number.isSafeInteger(Number(f.sectionHeaders[0].offset))).toBe(false);
  });
});

describe('identification errors', () => {
  it.each([0, 1, 2, 3] as const)('bad magic byte at %d', (i) => {
    const data = buildElf({ bits: 32 });
    data[i] = 0;
    expectParseError(() => parseElf(data), `e_ident[EI_MAG${i}]`, BigInt(i));
  });

  it('bad EI_CLASS', () => {
    const data = buildElf({ bits: 32 });
    data[4] = 9;
    expectParseError(() => parseElf(data), 'e_ident[EI_CLASS]', 4n);
  });

  it('bad EI_DATA', () => {
    const data = buildElf({ bits: 32 });
    data[5] = 9;
    expectParseError(() => parseElf(data), 'e_ident[EI_DATA]', 5n);
  });

  it('bad EI_VERSION', () => {
    const data = buildElf({ bits: 32 });
    data[6] = 0;
    expectParseError(() => parseElf(data), 'e_ident[EI_VERSION]', 6n);
  });

  it('parseIdentification returns pad bytes', () => {
    const elf = buildElf({ bits: 32 });
    elf.set([1, 2, 3, 4, 5, 6, 7], 9);
    const id = parseIdentification(elf);
    expect([...id.pad]).toEqual([1, 2, 3, 4, 5, 6, 7]);
  });
});

describe('ordinary field validation', () => {
  it('rejects e_shstrndx beyond ordinary shnum', () => {
    const bits = 32;
    const elf = buildElf({
      bits,
      ehdr: { shstrndx: 5 },
      shdrs: [{ sh_type: 1n }],
    });
    expectParseError(
      () => parseElf(elf),
      'e_shstrndx',
      fieldAbsOffset(ehFields[bits], 16n, 'e_shstrndx'),
    );
  });

  it('zero-count program table with out-of-range phoff is rejected', () => {
    const off = 1n << 50n;
    const elf = buildElf({
      bits: 64,
      ehdr: { phoff: off, phnum: 0, shoff: 0, shnum: 0, shstrndx: 0 },
    });
    expectParseError(() => parseElf(elf), 'e_phoff', off);
  });
});

describe('legacy parseHeader', () => {
  it('still returns bits/endian/type/machine', () => {
    const x = new Uint8Array(20);
    x.set([127, 69, 76, 70, 1, 1]);
    expect(parseHeader(x).bits).toBe(32);
    expect(parseHeader(x).littleEndian).toBe(true);
  });

  it('reads type and machine honoring endianness', () => {
    const elf = buildElf({
      bits: 64,
      littleEndian: false,
      ehdr: { type: 3, machine: 0x1234 },
    });
    const h = parseHeader(elf);
    expect(h.bits).toBe(64);
    expect(h.littleEndian).toBe(false);
    expect(h.type).toBe(3);
    expect(h.machine).toBe(0x1234);
  });
});
