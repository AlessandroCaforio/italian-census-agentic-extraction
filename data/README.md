# Sample Data

This directory contains representative samples from the census extraction pipeline, showing the before-and-after of the PDF processing workflow.

## Original PDFs (`original_pdf/`)

These are parts of the original 1927 Industrial Census volume (*Censimento Industriale e Commerciale del 1927*), as digitized by ISTAT. Each part contains multiple province tables across many pages.

The full volume was split into 17 parts, totaling ~102 MB. Only 2 sample parts are included here.

## Cropped Pairs (`cropped_pairs/`)

Each `pair_XXXX.pdf` is a two-page spread extracted from the original volume, containing one province's data table:
- **Left page**: Municipality names with sequential numbering
- **Right page**: Numerical data columns

These are the files that each AI agent receives and processes autonomously.

| File | Province | Accuracy |
|------|----------|----------|
| `pair_0001.pdf` | AGRIGENTO | ~97% |
| `pair_0015.pdf` | AREZZO | **100%** |
| `pair_0019.pdf` | BARI DELLE PUGLIE | **100%** (44/44) |
| `pair_0028.pdf` | BOLOGNA | **100%** (56/56) |
| `pair_0060.pdf` | CREMONA | ~98% |

## Population 1921 (`popolazione_1921/`)

Sample pages from the 1921 Population Census (*Censimento della Popolazione*), organized by region. These were processed using Mistral's OCR API rather than Claude's vision, as the table structure is more regular.

Each PDF contains TAVOLA XX (literacy data) with columns:
`COMUNI | MF | M | F | MF_literate | M_literate | F_literate | %MF | %M | %F`

## Source

All census data is from Italy's national statistical institute (ISTAT). The original documents are in the public domain (published 1927/1921, Italian government works).
