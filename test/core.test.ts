import { describe, expect, it } from 'vitest';
import {
  ELFCLASS32,
  ELFCLASS64,
  ELFDATA2LSB,
  ELFDATA2MSB,
  ELF_EHDR_SIZE,
  ELF_PHDR_SIZE,
  ELF_SHDR_SIZE,
  EV_CURRENT,
  ElfParseError,
  PT_LOAD,
  SHT_NOBITS,
  SHT_NULL,
  SHN_XINDEX,
  ehdrFields,
  parseElf,
  parseFileHeader,
  parseIdentification,
  parseProgramHeaders,
  parseSectionHeaders,
  phdrFields,
  shdrFields,
  type ElfBits,
  type IntKind,
  type StructFieldSpec,
} from '../src/index.js';

// ---------------------------------------------------------------------------
// Tiny ELF builder used to synthesize malformed/edge-case files.
// ---------------------------------------------------------------------------

type Value = number | bigint;

class ElfBuilder {
  buf: Uint8Array;
  view: DataView;
  readonly bits: ElfBits;
  readonly le: boolean;
  phoff = 0;
  shoff = 0;
  phnum = 0;
  shnum = 0;
  phentsize: number;
  shentsize: number;
  shstrndx = 0;

  constructor(bits: ElfBits, le: boolean, size: number) {
    this.bits = bits;
    this.le = le;
    this.buf = new Uint8Array(size);
    this.view = new DataView(this.buf.buffer);
    this.phentsize = ELF_PHDR_SIZE[bits];
    this.shentsize = ELF_SHDR_SIZE[bits];
    this.buf.set([0x7f, 0x45, 0x4c, 0x46]);
    this.buf[4] = bits === 32 ? ELFCLASS32 : ELFCLASS64;
    this.buf[5] = le ? ELFDATA2LSB : ELFDATA2MSB;
    this.buf[6] = EV_CURRENT;
  }

  private width(kind: IntKind): number {
    if (kind === 'xword') return this.bits === 32 ? 4 : 8;
    if (kind === 'u64') return 8;
    return { u8: 1, u16: 2, u32: 4 }[kind as 'u8' | 'u16' | 'u32'];
  }

  private write(width: number, offset: number, value: Value): void {
    const v = typeof value === 'bigint' ? value : BigInt(value);
    switch (width) {
      case 1:
        this.view.setUint8(offset, Number(v));
        break;
      case 2:
        this.view.setUint16(offset, Number(v), this.le);
        break;
      case 4:
        this.view.setUint32(offset, Number(v), this.le);
        break;
      default:
        this.view.setBigUint64(offset, v, this.le);
    }
  }

  field(fields: readonly StructFieldSpec[], base: number, name: string, value: Value): this {
    const f = fields.find((x) => x.name === name)!;
    this.write(this.width(f.kind), base + f.offset, value);
    return this;
  }

  ehdr(name: string, value: Value): this {
    return this.field(ehdrFields(this.bits), 0, name, value);
  }

  phdr(index: number, name: string, value: Value): this {
    return this.field(phdrFields(this.bits), this.phoff + index * this.phentsize, name, value);
  }

  shdr(index: number, name: string, value: Value): this {
    return this.field(shdrFields(this.bits), this.shoff + index * this.shentsize, name, value);
  }

  rawAbs(offset: number, width: number, value: Value): this {
    this.write(width, offset, value);
    return this;
  }

  finishHeader(): this {
    this.ehdr('version', EV_CURRENT);
    this.ehdr('ehsize', ELF_EHDR_SIZE[this.bits]);
    this.ehdr('phentsize', this.phentsize);
    this.ehdr('shentsize', this.shentsize);
    this.ehdr('phoff', this.phoff);
    this.ehdr('shoff', this.shoff);
    this.ehdr('phnum', this.phnum);
    this.ehdr('shnum', this.shnum);
    this.ehdr('shstrndx', this.shstrndx);
    return this;
  }
}

/** Minimal valid ELF: file header + one PT_LOAD segment + n sections. */
function makeElf(
  bits: ElfBits,
  le: boolean,
  shnum: number,
  opts: { extended?: boolean } = {},
): ElfBuilder {
  const eh = ELF_EHDR_SIZE[bits];
  const phs = ELF_PHDR_SIZE[bits];
  const shs = ELF_SHDR_SIZE[bits];
  const phoff = eh;
  const shoff = eh + phs;
  const total = shoff + shs * shnum;
  const b = new ElfBuilder(bits, le, total);
  b.phoff = phoff;
  b.shoff = shoff;
  b.phnum = 1;
  b.shnum = opts.extended ? 0 : shnum;
  b.finishHeader();
  b.phdr(0, 'type', PT_LOAD);
  b.phdr(0, 'offset', 0n);
  b.phdr(0, 'filesz', 0n);
  b.phdr(0, 'flags', 5);
  b.phdr(0, 'align', bits === 32 ? 0x1000 : 0x1000n);
  for (let i = 0; i < shnum; i++) {
    b.shdr(i, 'type', i === 0 ? SHT_NULL : 1);
    b.shdr(i, 'size', 0n);
  }
  if (opts.extended) {
    // Extended numbering: section 0 sh_size carries the real count.
    b.shdr(0, 'type', SHT_NULL);
    b.shdr(0, 'size', BigInt(shnum));
  }
  return b;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('identification', () => {
  it('accepts the four class/endianness layouts at ident level', () => {
    for (const [bits, classByte] of [
      [32, ELFCLASS32],
      [64, ELFCLASS64],
    ] as const) {
      for (const [le, dataByte] of [
        [true, ELFDATA2LSB],
        [false, ELFDATA2MSB],
      ] as const) {
        const b = new ElfBuilder(bits, le, ELF_EHDR_SIZE[bits]).finishHeader();
        const id = parseIdentification(b.buf);
        expect(id.bits).toBe(bits);
        expect(id.littleEndian).toBe(le);
        expect(id.range.offset).toBe(0n);
        expect(id.range.size).toBe(16n);
        expect(id.range.end).toBe(16n);
        expect(b.buf[4]).toBe(classByte);
        expect(b.buf[5]).toBe(dataByte);
      }
    }
  });

  it('rejects bad magic with field and absolute offset', () => {
    const b = new ElfBuilder(32, true, 64);
    b.buf[2] = 0;
    expect(() => parseIdentification(b.buf)).toThrow(ElfParseError);
    try {
      parseIdentification(b.buf);
      throw new Error('should have thrown');
    } catch (e) {
      expect(e).toBeInstanceOf(ElfParseError);
      const err = e as ElfParseError;
      expect(err.field).toBe('e_ident[EI_MAG]');
      expect(err.offset).toBe(2n);
    }
  });

  it('rejects unknown EI_CLASS, EI_DATA and EI_VERSION', () => {
    let b = new ElfBuilder(32, true, 64);
    b.buf[4] = 3;
    expect(() => parseIdentification(b.buf)).toThrow(/EI_CLASS/);
    expect(() => parseIdentification(b.buf)).toThrow('3');

    b = new ElfBuilder(32, true, 64);
    b.buf[5] = 3;
    expect(() => parseIdentification(b.buf)).toThrow(/EI_DATA/);

    b = new ElfBuilder(32, true, 64);
    b.buf[6] = 0;
    expect(() => parseIdentification(b.buf)).toThrow(/EI_VERSION/);
  });

  it('rejects a truncated identification block', () => {
    try {
      parseIdentification(new Uint8Array(10));
      throw new Error('should have thrown');
    } catch (e) {
      expect(e).toBeInstanceOf(ElfParseError);
      expect((e as ElfParseError).field).toBe('e_ident');
      expect((e as ElfParseError).offset).toBe(10n);
    }
  });
});

describe('file header — all four layouts', () => {
  const cases: ReadonlyArray<[ElfBits, boolean]> = [
    [32, true],
    [32, false],
    [64, true],
    [64, false],
  ];

  for (const [bits, le] of cases) {
    it(`parses ELF${bits} ${le ? 'little' : 'big'}-endian header with absolute range`, () => {
      const b = makeElf(bits, le, 1);
      b.ehdr('type', 2);
      b.ehdr('machine', bits === 32 ? 3 : 0x3e);
      b.ehdr('entry', bits === 32 ? 0x8048000 : 0x400000n);
      b.ehdr('flags', 0xdeadbeef);
      b.finishHeader();

      const h = parseFileHeader(b.buf);
      expect(h.identification.bits).toBe(bits);
      expect(h.identification.littleEndian).toBe(le);
      expect(h.fields.type).toBe(2);
      expect(h.fields.machine).toBe(bits === 32 ? 3 : 0x3e);
      expect(h.fields.entry).toBe(bits === 32 ? BigInt(0x8048000) : 0x400000n);
      expect(h.fields.flags).toBe(0xdeadbeef);
      expect(h.fields.phoff).toBe(BigInt(ELF_EHDR_SIZE[bits]));
      expect(h.fields.shoff).toBe(BigInt(ELF_EHDR_SIZE[bits] + ELF_PHDR_SIZE[bits]));
      expect(h.range.offset).toBe(0n);
      expect(h.range.end).toBe(BigInt(ELF_EHDR_SIZE[bits]));
      expect(typeof h.fields.phoff).toBe('bigint');
      expect(typeof h.fields.shoff).toBe('bigint');
      expect(typeof h.fields.ehsize).toBe('bigint');
    });
  }

  it('proves endianness matters: same bytes decode differently', () => {
    const le = makeElf(32, true, 1);
    le.ehdr('type', 0x0102);
    le.finishHeader();
    expect(parseFileHeader(le.buf).fields.type).toBe(0x0102);

    const be = makeElf(32, false, 1);
    // Same physical bytes as the LE encoding of 0x0102.
    be.buf.set(le.buf.subarray(16, 18), 16);
    expect(parseFileHeader(be.buf).fields.type).toBe(0x0201);
  });

  it('rejects a truncated file header', () => {
    const full = makeElf(64, true, 1).buf;
    try {
      parseFileHeader(full.subarray(0, 40));
      throw new Error('should have thrown');
    } catch (e) {
      expect(e).toBeInstanceOf(ElfParseError);
      expect((e as ElfParseError).field).toBe('e_ehsize');
      expect((e as ElfParseError).offset).toBe(40n);
    }
  });

  it('rejects wrong e_ehsize and bad e_version', () => {
    let b = makeElf(32, true, 1);
    b.ehdr('ehsize', 40);
    expect(() => parseFileHeader(b.buf)).toThrow(/e_ehsize/);

    b = makeElf(32, true, 1);
    b.ehdr('version', 0);
    expect(() => parseFileHeader(b.buf)).toThrow(/e_version/);
  });
});

describe('program headers — four layouts', () => {
  const cases: ReadonlyArray<[ElfBits, boolean]> = [
    [32, true],
    [32, false],
    [64, true],
    [64, false],
  ];

  for (const [bits, le] of cases) {
    it(`parses program header table for ELF${bits} ${le ? 'LE' : 'BE'}`, () => {
      const b = makeElf(bits, le, 1);
      const filesz = 16n;
      b.phdr(0, 'type', PT_LOAD);
      b.phdr(0, 'offset', 8n); // payload lives inside the file header..table area
      b.phdr(0, 'vaddr', bits === 32 ? 0x10000 : 0x20000n);
      b.phdr(0, 'paddr', bits === 32 ? 0x10000 : 0x20000n);
      b.phdr(0, 'filesz', filesz);
      b.phdr(0, 'memsz', bits === 32 ? 0x200 : 0x300n);
      b.phdr(0, 'flags', 7);
      b.phdr(0, 'align', bits === 32 ? 0x1000 : 0x1000n);

      const elf = parseElf(b.buf);
      const ph = elf.programHeaders;
      expect(ph.count).toBe(1);
      expect(ph.entrySize).toBe(BigInt(ELF_PHDR_SIZE[bits]));
      expect(ph.range.offset).toBe(BigInt(ELF_EHDR_SIZE[bits]));
      expect(ph.range.end).toBe(
        BigInt(ELF_EHDR_SIZE[bits] + ELF_PHDR_SIZE[bits]),
      );
      expect(ph.headers[0].range.offset).toBe(BigInt(ELF_EHDR_SIZE[bits]));
      expect(ph.headers[0].fields.type).toBe(PT_LOAD);
      expect(ph.headers[0].fields.offset).toBe(8n);
      expect(ph.headers[0].fields.filesz).toBe(16n);
      expect(ph.headers[0].fields.flags).toBe(7);
    });
  }

  it('rejects wrong e_phentsize, reporting the field at its absolute offset', () => {
    const b = makeElf(64, true, 1);
    b.phentsize = 32;
    b.ehdr('phentsize', 32);
    try {
      parseProgramHeaders(b.buf, parseFileHeader(b.buf));
      throw new Error('should have thrown');
    } catch (e) {
      expect(e).toBeInstanceOf(ElfParseError);
      const err = e as ElfParseError;
      expect(err.field).toBe('e_phentsize');
      expect(err.offset).toBe(54n);
    }
  });

  it('rejects a truncated program header table', () => {
    const b = makeElf(32, true, 1);
    b.phnum = 5;
    b.ehdr('phnum', 5);
    // Buffer only holds one entry; table range check must fail.
    try {
      parseElf(b.buf.subarray(0, ELF_EHDR_SIZE[32] + ELF_PHDR_SIZE[32]));
      throw new Error('should have thrown');
    } catch (e) {
      expect(e).toBeInstanceOf(ElfParseError);
      expect((e as ElfParseError).field).toBe('e_phoff');
    }
  });

  it('rejects segment payloads that run past EOF', () => {
    const b = makeElf(64, true, 1);
    b.phdr(0, 'offset', BigInt(b.buf.length) - 4n);
    b.phdr(0, 'filesz', 16n);
    expect(() => parseElf(b.buf)).toThrow(/p_offset/);
  });

  it('requires phoff/shnum consistency', () => {
    const b = makeElf(32, true, 1);
    b.phnum = 0;
    b.ehdr('phnum', 0);
    try {
      parseElf(b.buf);
      throw new Error('should have thrown');
    } catch (e) {
      expect((e as ElfParseError).field).toBe('e_phoff');
    }
  });
});

describe('section headers — four layouts', () => {
  const cases: ReadonlyArray<[ElfBits, boolean]> = [
    [32, true],
    [32, false],
    [64, true],
    [64, false],
  ];

  for (const [bits, le] of cases) {
    it(`parses section header table for ELF${bits} ${le ? 'LE' : 'BE'}`, () => {
      const shs = ELF_SHDR_SIZE[bits];
      const b = makeElf(bits, le, 3);
      b.shstrndx = 2;
      b.ehdr('shstrndx', 2);
      b.shdr(1, 'type', 1 /* PROGBITS */);
      b.shdr(1, 'offset', 8n); // payload bytes inside the header area
      b.shdr(1, 'size', 16n);
      b.shdr(2, 'type', 3 /* STRTAB */);

      const elf = parseElf(b.buf);
      const sh = elf.sectionHeaders;
      expect(sh.count).toBe(3);
      expect(sh.rawCount).toBe(3);
      expect(sh.stringIndex).toBe(2);
      expect(sh.entrySize).toBe(BigInt(shs));
      const expectedShoff = ELF_EHDR_SIZE[bits] + ELF_PHDR_SIZE[bits];
      expect(sh.range.offset).toBe(BigInt(expectedShoff));
      expect(sh.range.end).toBe(BigInt(expectedShoff + shs * 3));
      expect(sh.headers[2].range.offset).toBe(BigInt(expectedShoff + shs * 2));
      expect(sh.headers[1].fields.type).toBe(1);
    });
  }

  it('rejects wrong e_shentsize at its absolute offset', () => {
    const b = makeElf(32, true, 1);
    b.shentsize = 36;
    b.ehdr('shentsize', 36);
    try {
      parseElf(b.buf);
      throw new Error('should have thrown');
    } catch (e) {
      const err = e as ElfParseError;
      expect(err.field).toBe('e_shentsize');
      expect(err.offset).toBe(46n);
    }
  });

  it('rejects a truncated section header table', () => {
    const b = makeElf(64, true, 4);
    const cut = b.shoff + ELF_SHDR_SIZE[64] * 2; // only 2 of 4 entries present
    try {
      parseElf(b.buf.subarray(0, cut));
      throw new Error('should have thrown');
    } catch (e) {
      expect(e).toBeInstanceOf(ElfParseError);
      expect((e as ElfParseError).field).toBe('e_shoff');
    }
  });

  it('rejects a non-NOBITS section whose bytes exceed EOF', () => {
    const b = makeElf(32, true, 2);
    b.shdr(1, 'type', 1);
    b.shdr(1, 'offset', BigInt(b.buf.length - 2));
    b.shdr(1, 'size', 10n);
    expect(() => parseElf(b.buf)).toThrow(/sh_offset/);
  });

  it('does not range-check SHT_NOBITS payloads', () => {
    const b = makeElf(64, true, 2);
    b.shdr(1, 'type', SHT_NOBITS);
    b.shdr(1, 'offset', 0x100000n);
    b.shdr(1, 'size', 0x1000000n);
    const elf = parseElf(b.buf);
    expect(elf.sectionHeaders.headers[1].fields.type).toBe(SHT_NOBITS);
  });

  it('rejects e_shstrndx out of range, at its absolute offset', () => {
    const b = makeElf(32, true, 1);
    b.shstrndx = 9;
    b.ehdr('shstrndx', 9);
    try {
      parseElf(b.buf);
      throw new Error('should have thrown');
    } catch (e) {
      const err = e as ElfParseError;
      expect(err.field).toBe('e_shstrndx');
      expect(err.offset).toBe(50n);
    }
  });
});

describe('extended section numbering (section 0 is authoritative)', () => {
  it('takes the count from section 0 sh_size when e_shnum is 0', () => {
    const b = makeElf(64, true, 7, { extended: true });
    const elf = parseElf(b.buf);
    expect(elf.sectionHeaders.rawCount).toBe(0);
    expect(elf.sectionHeaders.count).toBe(7);
    expect(elf.sectionHeaders.headers).toHaveLength(7);
  });

  it('resolves SHN_XINDEX shstrndx via section 0 sh_link', () => {
    const b = makeElf(32, true, 4, { extended: true });
    b.shstrndx = SHN_XINDEX;
    b.ehdr('shstrndx', SHN_XINDEX);
    b.shdr(0, 'link', 3);
    const elf = parseElf(b.buf);
    expect(elf.sectionHeaders.rawStringIndex).toBe(SHN_XINDEX);
    expect(elf.sectionHeaders.stringIndex).toBe(3);
  });

  it('does not trust e_shnum: truncated table detected via sh_size count', () => {
    // Header claims e_shnum=0; real count 9 lives in section 0, but the file
    // only contains 2 entries. The reader must discover this from section 0
    // rather than trusting the ordinary (zero) field.
    const b = makeElf(64, true, 2, { extended: false });
    b.shnum = 0;
    b.ehdr('shnum', 0);
    b.shdr(0, 'size', 9n);
    try {
      parseElf(b.buf);
      throw new Error('should have thrown');
    } catch (e) {
      expect(e).toBeInstanceOf(ElfParseError);
      expect((e as ElfParseError).field).toBe('e_shoff');
    }
  });

  it('rejects SHN_XINDEX without the e_shnum escape', () => {
    const b = makeElf(64, true, 3);
    b.shstrndx = SHN_XINDEX;
    b.ehdr('shstrndx', SHN_XINDEX);
    expect(() => parseElf(b.buf)).toThrow(/SHN_XINDEX/);
  });

  it('rejects extended shstrndx that points past the last section', () => {
    const b = makeElf(64, true, 3, { extended: true });
    b.shstrndx = SHN_XINDEX;
    b.ehdr('shstrndx', SHN_XINDEX);
    b.shdr(0, 'link', 42);
    try {
      parseElf(b.buf);
      throw new Error('should have thrown');
    } catch (e) {
      expect((e as ElfParseError).field).toBe('sh_link');
    }
  });

  it('rejects extended numbering when section 0 sh_size is zero', () => {
    const bits: ElfBits = 32;
    const b = makeElf(bits, true, 2);
    b.ehdr('shnum', 0);
    // makeElf wrote a positive section count into section 0 sh_size; zero it
    // at the field's real offset (20 for ELF32).
    const shSizeOff =
      b.shoff + shdrFields(bits).find((f) => f.name === 'size')!.offset;
    b.view.setUint32(shSizeOff, 0, true);
    try {
      parseElf(b.buf);
      throw new Error('should have thrown');
    } catch (e) {
      expect(e).toBeInstanceOf(ElfParseError);
      expect((e as ElfParseError).field).toBe('sh_size');
      expect((e as ElfParseError).offset).toBe(BigInt(shSizeOff));
    }
  });

  it('rejects reserved non-XINDEX shstrndx values', () => {
    const b = makeElf(32, true, 2);
    b.shstrndx = 0xff10;
    b.ehdr('shstrndx', 0xff10);
    expect(() => parseElf(b.buf)).toThrow(/reserved/);
  });
});

describe('files without a section header table', () => {
  it('parses when shoff/shnum/shstrndx are all zero', () => {
    const bits: ElfBits = 64;
    const total = ELF_EHDR_SIZE[bits] + ELF_PHDR_SIZE[bits];
    const b = new ElfBuilder(bits, true, total);
    b.phoff = ELF_EHDR_SIZE[bits];
    b.phnum = 1;
    b.finishHeader();
    b.phdr(0, 'type', PT_LOAD);
    const elf = parseElf(b.buf);
    expect(elf.sectionHeaders.count).toBe(0);
    expect(elf.sectionHeaders.headers).toEqual([]);
    expect(elf.sectionHeaders.stringIndex).toBeNull();
    expect(elf.sectionHeaders.range.size).toBe(0n);
  });

  it('rejects non-zero shnum with zero shoff', () => {
    const bits: ElfBits = 32;
    const total = ELF_EHDR_SIZE[bits] + ELF_PHDR_SIZE[bits];
    const b = new ElfBuilder(bits, true, total);
    b.phoff = ELF_EHDR_SIZE[bits];
    b.phnum = 1;
    b.finishHeader();
    b.shnum = 1;
    b.ehdr('shnum', 1);
    b.phdr(0, 'type', PT_LOAD);
    try {
      parseElf(b.buf);
      throw new Error('should have thrown');
    } catch (e) {
      expect((e as ElfParseError).field).toBe('e_shnum');
    }
  });
});

describe('overlapping structural ranges', () => {
  it('rejects program and section header tables occupying the same bytes', () => {
    const bits = 32;
    const shs = ELF_SHDR_SIZE[bits];
    const b = new ElfBuilder(bits, true, ELF_EHDR_SIZE[bits] + shs);
    b.phoff = ELF_EHDR_SIZE[bits];
    b.shoff = ELF_EHDR_SIZE[bits]; // same start as ph table
    b.phnum = 1;
    b.shnum = 1;
    b.finishHeader();
    b.phdr(0, 'type', PT_LOAD);
    b.shdr(0, 'type', SHT_NULL);
    try {
      parseElf(b.buf);
      throw new Error('should have thrown');
    } catch (e) {
      expect(e).toBeInstanceOf(ElfParseError);
      expect((e as ElfParseError).field).toBe('e_phoff');
      expect((e as ElfParseError).message).toMatch(/overlaps/);
    }
  });

  it('rejects a table overlapping the file header', () => {
    const bits = 64;
    const phs = ELF_PHDR_SIZE[bits];
    const shs = ELF_SHDR_SIZE[bits];
    const b = new ElfBuilder(bits, true, ELF_EHDR_SIZE[bits] + phs + shs);
    b.phoff = ELF_EHDR_SIZE[bits];
    b.shoff = 32; // section table starts inside the file header
    b.phnum = 1;
    b.shnum = 1;
    b.finishHeader();
    b.phdr(0, 'type', PT_LOAD); // filesz/p_offset zero -> no payload check
    b.shdr(0, 'type', SHT_NULL);
    try {
      parseElf(b.buf);
      throw new Error('should have thrown');
    } catch (e) {
      expect(e).toBeInstanceOf(ElfParseError);
      expect((e as ElfParseError).field).toBe('e_shoff');
      expect((e as ElfParseError).offset).toBe(40n);
      expect((e as ElfParseError).message).toMatch(/overlaps/);
    }
  });
});

describe('offsets beyond Number.MAX_SAFE_INTEGER', () => {
  it('round-trips a 64-bit shoff above 2^53 and then rejects it as out of range', () => {
    const huge = (1n << 54n) + 7n; // > Number.MAX_SAFE_INTEGER
    const b = makeElf(64, true, 1);
    b.rawAbs(40, 8, huge); // e_shoff for ELF64
    const h = parseFileHeader(b.buf);
    expect(h.fields.shoff).toBe(huge); // preserved exactly as bigint
    try {
      parseSectionHeaders(b.buf, h);
      throw new Error('should have thrown');
    } catch (e) {
      const err = e as ElfParseError;
      expect(err.field).toBe('e_shoff');
      expect(err.offset).toBe(40n);
      expect(err.message).toContain(huge.toString(16));
    }
  });

  it('rejects a huge program header offset without precision loss', () => {
    const huge = (1n << 60n) + 123n;
    const b = makeElf(64, true, 1);
    b.rawAbs(32, 8, huge); // e_phoff for ELF64
    b.rawAbs(56, 2, 1); // e_phnum
    const h = parseFileHeader(b.buf);
    expect(h.fields.phoff).toBe(huge);
    try {
      parseProgramHeaders(b.buf, h);
      throw new Error('should have thrown');
    } catch (e) {
      const err = e as ElfParseError;
      expect(err.field).toBe('e_phoff');
      expect(err.message).toContain(huge.toString(16));
    }
  });

  it('uses bigint arithmetic so table size = entsize*count cannot overflow', () => {
    // Build a file whose e_shnum is enormous; 0xffff * 64 runs far past EOF.
    // Multiplication happens in bigint, so no 32/53-bit wraparound hides it.
    const bits: ElfBits = 64;
    const b = makeElf(bits, true, 1);
    b.view.setUint16(60, 0xffff, true); // patch e_shnum after the header is laid out
    expect(b.buf[20]).toBe(EV_CURRENT); // e_version untouched by the patch
    try {
      parseElf(b.buf);
      throw new Error('should have thrown');
    } catch (e) {
      expect(e).toBeInstanceOf(ElfParseError);
      expect((e as ElfParseError).field).toBe('e_shoff');
      expect((e as ElfParseError).offset).toBe(40n);
    }
  });
});

describe('parseElf integration', () => {
  it('returns absolute ranges for every parsed structure', () => {
    const bits: ElfBits = 64;
    const b = makeElf(bits, true, 3);
    const elf = parseElf(b.buf);
    expect(elf.dataLength).toBe(BigInt(b.buf.length));
    expect(elf.header.range).toEqual({
      offset: 0n,
      size: BigInt(ELF_EHDR_SIZE[bits]),
      end: BigInt(ELF_EHDR_SIZE[bits]),
    });
    for (const [i, ph] of elf.programHeaders.headers.entries()) {
      expect(ph.range.offset).toBe(
        BigInt(ELF_EHDR_SIZE[bits] + i * ELF_PHDR_SIZE[bits]),
      );
      expect(ph.range.end - ph.range.offset).toBe(BigInt(ELF_PHDR_SIZE[bits]));
    }
    const shBase = ELF_EHDR_SIZE[bits] + ELF_PHDR_SIZE[bits];
    for (const [i, sh] of elf.sectionHeaders.headers.entries()) {
      expect(sh.range.offset).toBe(BigInt(shBase + i * ELF_SHDR_SIZE[bits]));
      expect(sh.range.end - sh.range.offset).toBe(BigInt(ELF_SHDR_SIZE[bits]));
    }
  });
});
