# ELF codec core

TypeScript library for ELF binary structures.

Parses ELF identification (`e_ident`), the file header (Ehdr), program header
table (Phdr) and section header table (Shdr), supporting all four layouts:
ELF32/ELF64 × little-endian/big-endian. Layout and endianness are selected
solely from `EI_CLASS` and `EI_DATA` in `e_ident`.

Run `npm install`, then `npm test` and `npm run build`.

## API

```ts
import { parseElf, ElfParseError } from './src/index.js';

const elf = parseElf(bytes);

elf.identification.eiClass;        // 32 | 64
elf.identification.littleEndian;   // boolean
elf.header;                        // ElfFileHeader
elf.programHeaders;                // ProgramHeader[]
elf.sectionHeaders;                // SectionHeader[]
elf.programHeaderCount;            // bigint, extended numbering (PN_XNUM) resolved
elf.sectionHeaderCount;            // bigint, extended numbering (e_shnum=0) resolved
elf.sectionNameStringIndex;        // bigint, SHN_XINDEX resolved via Shdr[0].sh_link
elf.ranges;                        // absolute {start,end,size} ranges of Ehdr/Phdr/Shdr
```

- Every file offset and size is a `bigint` (offsets beyond
  `Number.MAX_SAFE_INTEGER` stay exact); only 16-bit fields stay `number`.
- Each parsed struct carries an absolute, half-open byte `range`.
- Table ranges are validated against file length, entry sizes against the
  canonical per-class sizes, counts are bounds-checked, and Ehdr/Phdr/Shdr
  ranges are checked for overlap before any entries are decoded.
- Extended numbering: when `e_shnum == 0`, the real count comes from
  `Shdr[0].sh_size`; `SHN_XINDEX` (`0xffff`) for `e_shstrndx` resolves via
  `Shdr[0].sh_link`; `PN_XNUM` (`0xffff`) for `e_phnum` resolves via
  `Shdr[0].sh_info`. Section 0 is consulted before trusting ordinary fields.
- All failures throw `ElfParseError` carrying `field` and absolute byte
  `offset`, e.g.
  `ElfParseError: e_ehsize is 40 but ... (field=e_ehsize, offset=40)`.

`parseHeader(bytes)` remains available as a lightweight legacy parse
(class/endianness plus `e_type`/`e_machine`).
