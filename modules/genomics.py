"""
Genomics / Bioinformatics Widget for QuantumRes Scientific Suite.

Provides DNA/RNA/Protein sequence analysis, alignment, ORF finding,
restriction enzyme mapping, codon usage, phylogenetics, and visualization.
All algorithms implemented from scratch (no BioPython dependency).
"""

import os
import re
import math
import random
import traceback
from collections import Counter, defaultdict

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QFileDialog, QLabel, QComboBox, QLineEdit, QGroupBox,
    QSplitter, QTextEdit, QToolBar, QTabWidget, QPlainTextEdit,
    QSpinBox, QDoubleSpinBox, QCheckBox, QHeaderView, QMessageBox,
    QFormLayout, QDialogButtonBox, QDialog, QSizePolicy, QApplication
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from mpl_style import style_figure, style_axes
except ImportError:
    def style_figure(f): pass
    def style_axes(a): pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CODON_TABLE = {
    "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L",
    "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
    "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M",
    "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
    "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S",
    "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "UAU": "Y", "UAC": "Y", "UAA": "*", "UAG": "*",
    "CAU": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAU": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAU": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "UGU": "C", "UGC": "C", "UGA": "*", "UGG": "W",
    "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGU": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

DNA_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G",
                  "a": "t", "t": "a", "g": "c", "c": "g"}
RNA_COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G",
                  "a": "u", "u": "a", "g": "c", "c": "g"}

# Average molecular weights (Da) for nucleotides and amino acids
NUC_MW = {"A": 331.2, "T": 322.2, "U": 306.2, "G": 347.2, "C": 307.2}
AA_MW = {
    "G": 57.05, "A": 71.08, "V": 99.13, "L": 113.16, "I": 113.16,
    "P": 97.12, "F": 147.18, "W": 186.21, "M": 131.20, "S": 87.08,
    "T": 101.10, "C": 103.14, "Y": 163.18, "H": 137.14, "D": 115.09,
    "E": 129.12, "N": 114.10, "Q": 128.13, "K": 128.17, "R": 156.19,
    "*": 0.0,
}

RESTRICTION_ENZYMES = {
    "EcoRI":   ("GAATTC", 1),
    "BamHI":   ("GGATCC", 1),
    "HindIII": ("AAGCTT", 1),
    "NotI":    ("GCGGCCGC", 2),
    "XhoI":    ("CTCGAG", 1),
    "SalI":    ("GTCGAC", 1),
    "PstI":    ("CTGCAG", 5),
    "SmaI":    ("CCCGGG", 3),
    "KpnI":    ("GGTACC", 5),
    "SacI":    ("GAGCTC", 5),
    "EcoRV":   ("GATATC", 3),
    "NcoI":    ("CCATGG", 1),
    "NdeI":    ("CATATG", 2),
    "XbaI":    ("TCTAGA", 1),
    "BglII":   ("AGATCT", 1),
    "ClaI":    ("ATCGAT", 2),
    "NheI":    ("GCTAGC", 1),
    "SpeI":    ("ACTAGT", 1),
    "MluI":    ("ACGCGT", 1),
    "ApaI":    ("GGGCCC", 5),
}

# Simple BLOSUM62-like scoring for protein alignment (identity=4, mismatch=-1)
_MATCH_SCORE = 4
_MISMATCH_SCORE = -1

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _detect_seq_type(seq: str) -> str:
    """Detect whether sequence is DNA, RNA, or Protein."""
    upper = seq.upper().replace("\n", "").replace(" ", "")
    if not upper:
        return "DNA"
    bases = set(upper)
    if bases <= {"A", "T", "G", "C", "N"}:
        return "DNA"
    if bases <= {"A", "U", "G", "C", "N"}:
        return "RNA"
    return "Protein"


def _parse_fasta(text: str):
    """Parse FASTA formatted text. Returns list of (header, sequence)."""
    sequences = []
    header = None
    seq_parts = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                sequences.append((header, "".join(seq_parts)))
            header = line[1:].strip()
            seq_parts = []
        else:
            seq_parts.append(line)
    if header is not None:
        sequences.append((header, "".join(seq_parts)))
    elif seq_parts:
        sequences.append(("Unnamed", "".join(seq_parts)))
    return sequences


def _gc_content(seq: str) -> float:
    """Return GC content as a fraction."""
    seq = seq.upper()
    gc = seq.count("G") + seq.count("C")
    total = len(seq)
    return gc / total if total > 0 else 0.0


def _melting_temp(seq: str) -> float:
    """Estimate Tm using the Wallace rule for short oligos and
    the simple salt-adjusted formula for longer sequences."""
    seq = seq.upper()
    n = len(seq)
    if n == 0:
        return 0.0
    a = seq.count("A")
    t = seq.count("T") + seq.count("U")
    g = seq.count("G")
    c = seq.count("C")
    if n < 14:
        return 2.0 * (a + t) + 4.0 * (g + c)
    gc_frac = (g + c) / n
    return 81.5 + 16.6 * math.log10(0.05) + 41.0 * gc_frac - 600.0 / n


def _molecular_weight(seq: str, seq_type: str) -> float:
    """Estimate molecular weight in Daltons."""
    seq = seq.upper()
    if seq_type in ("DNA", "RNA"):
        return sum(NUC_MW.get(ch, 0.0) for ch in seq) - (len(seq) - 1) * 18.02
    else:
        return sum(AA_MW.get(ch, 0.0) for ch in seq) - (len(seq) - 1) * 18.02 + 18.02


def _transcribe(dna: str) -> str:
    return dna.replace("T", "U").replace("t", "u")


def _reverse_complement(seq: str, seq_type: str = "DNA") -> str:
    comp = DNA_COMPLEMENT if seq_type == "DNA" else RNA_COMPLEMENT
    return "".join(comp.get(ch, ch) for ch in reversed(seq))


def _translate(rna: str) -> str:
    """Translate RNA sequence to protein using standard codon table."""
    rna = rna.upper()
    protein = []
    for i in range(0, len(rna) - 2, 3):
        codon = rna[i:i + 3]
        aa = CODON_TABLE.get(codon, "X")
        protein.append(aa)
    return "".join(protein)


def _find_orfs(seq: str, min_length: int = 30) -> list:
    """Find all ORFs in all 3 forward reading frames.
    Returns list of (frame, start, end, protein)."""
    rna = seq.upper().replace("T", "U")
    orfs = []
    for frame in range(3):
        i = frame
        while i < len(rna) - 2:
            codon = rna[i:i + 3]
            if codon == "AUG":
                start = i
                prot = ["M"]
                j = i + 3
                while j < len(rna) - 2:
                    c = rna[j:j + 3]
                    aa = CODON_TABLE.get(c, "X")
                    if aa == "*":
                        break
                    prot.append(aa)
                    j += 3
                end = j + 3
                protein = "".join(prot)
                if len(protein) >= min_length // 3:
                    orfs.append((frame + 1, start, end, protein))
                i = j + 3
            else:
                i += 3
    return orfs


def _find_restriction_sites(seq: str) -> list:
    """Find restriction enzyme cut sites. Returns list of (enzyme, position, recognition_seq)."""
    seq_upper = seq.upper()
    results = []
    for enzyme, (site, cut_offset) in RESTRICTION_ENZYMES.items():
        site_upper = site.upper()
        idx = 0
        while True:
            pos = seq_upper.find(site_upper, idx)
            if pos == -1:
                break
            results.append((enzyme, pos, pos + cut_offset, site))
            idx = pos + 1
    results.sort(key=lambda x: x[1])
    return results


# ---------------------------------------------------------------------------
# Alignment algorithms
# ---------------------------------------------------------------------------

def _needleman_wunsch(seq1: str, seq2: str, match=2, mismatch=-1, gap=-2):
    """Global alignment using Needleman-Wunsch algorithm."""
    n, m = len(seq1), len(seq2)
    score = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] + gap
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] + gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if seq1[i - 1] == seq2[j - 1] else mismatch
            score[i][j] = max(
                score[i - 1][j - 1] + s,
                score[i - 1][j] + gap,
                score[i][j - 1] + gap,
            )
    # Traceback
    align1, align2 = [], []
    i, j = n, m
    while i > 0 and j > 0:
        s = match if seq1[i - 1] == seq2[j - 1] else mismatch
        if score[i][j] == score[i - 1][j - 1] + s:
            align1.append(seq1[i - 1])
            align2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif score[i][j] == score[i - 1][j] + gap:
            align1.append(seq1[i - 1])
            align2.append("-")
            i -= 1
        else:
            align1.append("-")
            align2.append(seq2[j - 1])
            j -= 1
    while i > 0:
        align1.append(seq1[i - 1])
        align2.append("-")
        i -= 1
    while j > 0:
        align1.append("-")
        align2.append(seq2[j - 1])
        j -= 1
    return "".join(reversed(align1)), "".join(reversed(align2)), int(score[n][m])


def _smith_waterman(seq1: str, seq2: str, match=2, mismatch=-1, gap=-2):
    """Local alignment using Smith-Waterman algorithm."""
    n, m = len(seq1), len(seq2)
    score = np.zeros((n + 1, m + 1), dtype=int)
    max_score = 0
    max_pos = (0, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if seq1[i - 1] == seq2[j - 1] else mismatch
            score[i][j] = max(
                0,
                score[i - 1][j - 1] + s,
                score[i - 1][j] + gap,
                score[i][j - 1] + gap,
            )
            if score[i][j] > max_score:
                max_score = score[i][j]
                max_pos = (i, j)
    # Traceback
    align1, align2 = [], []
    i, j = max_pos
    while i > 0 and j > 0 and score[i][j] > 0:
        s = match if seq1[i - 1] == seq2[j - 1] else mismatch
        if score[i][j] == score[i - 1][j - 1] + s:
            align1.append(seq1[i - 1])
            align2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif score[i][j] == score[i - 1][j] + gap:
            align1.append(seq1[i - 1])
            align2.append("-")
            i -= 1
        else:
            align1.append("-")
            align2.append(seq2[j - 1])
            j -= 1
    return "".join(reversed(align1)), "".join(reversed(align2)), int(max_score)


def _dot_matrix(seq1: str, seq2: str, window: int = 1) -> np.ndarray:
    """Compute a dot-plot matrix for two sequences."""
    s1 = seq1.upper()
    s2 = seq2.upper()
    n, m = len(s1), len(s2)
    mat = np.zeros((n, m), dtype=np.uint8)
    for i in range(n - window + 1):
        for j in range(m - window + 1):
            if s1[i:i + window] == s2[j:j + window]:
                mat[i][j] = 1
    return mat


def _codon_usage(seq: str) -> dict:
    """Compute codon usage frequencies from a nucleotide sequence."""
    rna = seq.upper().replace("T", "U")
    counts = Counter()
    for i in range(0, len(rna) - 2, 3):
        codon = rna[i:i + 3]
        if codon in CODON_TABLE:
            counts[codon] += 1
    total = sum(counts.values()) or 1
    return {codon: counts.get(codon, 0) / total for codon in sorted(CODON_TABLE.keys())}


def _blast_local_search(query: str, database: list, word_size: int = 11, threshold: int = 20):
    """Simple BLAST-like local search: seed-and-extend.
    database is list of (name, sequence).
    Returns list of (name, position, score, aligned_query, aligned_subject)."""
    query = query.upper()
    results = []
    for name, subject in database:
        subject = subject.upper()
        # Seed phase: find exact word matches
        seeds = []
        for i in range(len(query) - word_size + 1):
            word = query[i:i + word_size]
            idx = 0
            while True:
                pos = subject.find(word, idx)
                if pos == -1:
                    break
                seeds.append((i, pos))
                idx = pos + 1
        # Extend seeds
        used = set()
        for qi, si in seeds:
            key = (qi // 10, si // 10)
            if key in used:
                continue
            used.add(key)
            # Extend left
            left = 0
            while qi - left - 1 >= 0 and si - left - 1 >= 0:
                if query[qi - left - 1] == subject[si - left - 1]:
                    left += 1
                else:
                    break
            # Extend right
            right = word_size
            while qi + right < len(query) and si + right < len(subject):
                if query[qi + right] == subject[si + right]:
                    right += 1
                else:
                    break
            total_len = left + right
            score = sum(
                2 if query[qi - left + k] == subject[si - left + k] else -1
                for k in range(total_len)
            )
            if score >= threshold:
                q_seg = query[qi - left:qi - left + total_len]
                s_seg = subject[si - left:si - left + total_len]
                results.append((name, si - left, score, q_seg, s_seg))
    results.sort(key=lambda x: -x[2])
    return results[:50]


# ---------------------------------------------------------------------------
# Phylogenetic tree (UPGMA)
# ---------------------------------------------------------------------------

def _pairwise_distance(seq1: str, seq2: str) -> float:
    """Simple p-distance between two aligned sequences."""
    s1, s2 = seq1.upper(), seq2.upper()
    length = min(len(s1), len(s2))
    if length == 0:
        return 1.0
    mismatches = sum(1 for i in range(length) if s1[i] != s2[i])
    return mismatches / length


def _upgma(names: list, dist_matrix: np.ndarray) -> str:
    """UPGMA clustering. Returns Newick-format tree string."""
    n = len(names)
    clusters = {i: names[i] for i in range(n)}
    sizes = {i: 1 for i in range(n)}
    heights = {i: 0.0 for i in range(n)}
    dm = dist_matrix.copy().astype(float)
    np.fill_diagonal(dm, np.inf)
    active = list(range(n))
    next_id = n

    while len(active) > 1:
        # Find minimum distance
        min_d = np.inf
        mi, mj = 0, 1
        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                d = dm[active[ii]][active[jj]]
                if d < min_d:
                    min_d = d
                    mi, mj = ii, jj
        ci, cj = active[mi], active[mj]
        new_h = min_d / 2.0
        bl_i = new_h - heights[ci]
        bl_j = new_h - heights[cj]
        new_label = f"({clusters[ci]}:{bl_i:.4f},{clusters[cj]}:{bl_j:.4f})"

        # Expand distance matrix
        old_size = dm.shape[0]
        new_dm = np.full((old_size + 1, old_size + 1), np.inf)
        new_dm[:old_size, :old_size] = dm
        for k in active:
            if k == ci or k == cj:
                continue
            d_new = (dm[ci][k] * sizes[ci] + dm[cj][k] * sizes[cj]) / (sizes[ci] + sizes[cj])
            new_dm[next_id][k] = d_new
            new_dm[k][next_id] = d_new
        dm = new_dm

        clusters[next_id] = new_label
        sizes[next_id] = sizes[ci] + sizes[cj]
        heights[next_id] = new_h
        active.remove(ci)
        active.remove(cj)
        active.append(next_id)
        next_id += 1

    root = active[0]
    return clusters[root] + ";"


def _neighbor_joining(names: list, dist_matrix: np.ndarray) -> str:
    """Neighbor-Joining tree construction. Returns Newick string."""
    n = len(names)
    if n <= 1:
        return names[0] + ";" if names else ";"
    dm = dist_matrix.copy().astype(float)
    active = list(range(n))
    labels = {i: names[i] for i in range(n)}
    next_id = n

    while len(active) > 2:
        r = len(active)
        # Compute row sums
        row_sums = {}
        for i in active:
            row_sums[i] = sum(dm[i][j] for j in active if j != i)
        # Find pair with minimum Q
        min_q = np.inf
        mi, mj = active[0], active[1]
        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                i, j = active[ii], active[jj]
                q = (r - 2) * dm[i][j] - row_sums[i] - row_sums[j]
                if q < min_q:
                    min_q = q
                    mi, mj = i, j
        # Branch lengths
        bl_i = dm[mi][mj] / 2.0 + (row_sums[mi] - row_sums[mj]) / (2.0 * (r - 2))
        bl_j = dm[mi][mj] - bl_i
        bl_i = max(bl_i, 0.0)
        bl_j = max(bl_j, 0.0)
        new_label = f"({labels[mi]}:{bl_i:.4f},{labels[mj]}:{bl_j:.4f})"

        # Expand matrix
        old_size = dm.shape[0]
        new_dm = np.full((old_size + 1, old_size + 1), 0.0)
        new_dm[:old_size, :old_size] = dm
        for k in active:
            if k == mi or k == mj:
                continue
            d_new = (dm[mi][k] + dm[mj][k] - dm[mi][mj]) / 2.0
            d_new = max(d_new, 0.0)
            new_dm[next_id][k] = d_new
            new_dm[k][next_id] = d_new
        dm = new_dm
        labels[next_id] = new_label
        active.remove(mi)
        active.remove(mj)
        active.append(next_id)
        next_id += 1

    # Last two
    i, j = active
    d = dm[i][j]
    return f"({labels[i]}:{d / 2:.4f},{labels[j]}:{d / 2:.4f});"


# ---------------------------------------------------------------------------
# Random Sequence Generator
# ---------------------------------------------------------------------------

def _generate_random_sequence(seq_type="DNA", length=100, composition=None):
    """Generate a random DNA/RNA/protein sequence with optional composition.
    composition: dict of {char: weight}, e.g. {'A': 0.3, 'T': 0.2, 'G': 0.25, 'C': 0.25}
    """
    if seq_type == "DNA":
        alphabet = list(composition.keys()) if composition else ["A", "T", "G", "C"]
        weights = list(composition.values()) if composition else [0.25, 0.25, 0.25, 0.25]
    elif seq_type == "RNA":
        alphabet = list(composition.keys()) if composition else ["A", "U", "G", "C"]
        weights = list(composition.values()) if composition else [0.25, 0.25, 0.25, 0.25]
    else:  # Protein
        aa_list = list("ACDEFGHIKLMNPQRSTVWY")
        alphabet = list(composition.keys()) if composition else aa_list
        weights = list(composition.values()) if composition else [1.0/len(aa_list)] * len(aa_list)
    total = sum(weights)
    weights = [w / total for w in weights]
    return "".join(random.choices(alphabet, weights=weights, k=length))


# ---------------------------------------------------------------------------
# Primer Design
# ---------------------------------------------------------------------------

def _design_primers(seq, target_start, target_end, primer_len_range=(18, 25),
                    tm_range=(55.0, 65.0), gc_range=(0.40, 0.60)):
    """Design forward and reverse primers for a target region.
    Returns list of (direction, sequence, start, tm, gc) tuples."""
    seq = seq.upper()
    results = []
    for plen in range(primer_len_range[0], primer_len_range[1] + 1):
        # Forward primer: upstream of target_start
        start = max(0, target_start - plen)
        fwd = seq[start:start + plen]
        if len(fwd) == plen:
            gc = _gc_content(fwd)
            tm = _melting_temp(fwd)
            if tm_range[0] <= tm <= tm_range[1] and gc_range[0] <= gc <= gc_range[1]:
                results.append(("Forward", fwd, start, tm, gc))
        # Reverse primer: downstream of target_end
        end = min(len(seq), target_end + plen)
        rev_region = seq[end - plen:end]
        if len(rev_region) == plen:
            rev = _reverse_complement(rev_region, "DNA")
            gc = _gc_content(rev)
            tm = _melting_temp(rev)
            if tm_range[0] <= tm <= tm_range[1] and gc_range[0] <= gc <= gc_range[1]:
                results.append(("Reverse", rev, end - plen, tm, gc))
    return results


# ---------------------------------------------------------------------------
# Progressive Multiple Sequence Alignment
# ---------------------------------------------------------------------------

def _progressive_msa(sequences, names=None, match=2, mismatch=-1, gap=-2):
    """Align 3+ sequences using progressive alignment (guide-tree approach).
    Returns list of (name, aligned_sequence) tuples."""
    if names is None:
        names = [f"Seq{i+1}" for i in range(len(sequences))]
    if len(sequences) < 2:
        return list(zip(names, sequences))

    # Build guide tree via pairwise distances
    n = len(sequences)
    aligned = list(sequences)
    aligned_names = list(names)

    # Align first two
    a1, a2, _ = _needleman_wunsch(aligned[0], aligned[1], match, mismatch, gap)
    result = [(aligned_names[0], a1), (aligned_names[1], a2)]

    # Progressively add remaining sequences
    for i in range(2, n):
        # Align new sequence against the consensus of current alignment
        consensus = _build_consensus([s for _, s in result])
        a_cons, a_new, _ = _needleman_wunsch(consensus, sequences[i], match, mismatch, gap)
        # Propagate gaps from consensus alignment to all existing aligned seqs
        new_result = []
        for name, old_aligned in result:
            new_aligned = _propagate_gaps(old_aligned, consensus, a_cons)
            new_result.append((name, new_aligned))
        new_result.append((aligned_names[i], a_new))
        result = new_result

    return result


def _build_consensus(aligned_seqs):
    """Build a simple consensus sequence from aligned sequences."""
    if not aligned_seqs:
        return ""
    length = max(len(s) for s in aligned_seqs)
    consensus = []
    for i in range(length):
        counts = Counter()
        for s in aligned_seqs:
            if i < len(s) and s[i] != "-":
                counts[s[i]] += 1
        if counts:
            consensus.append(counts.most_common(1)[0][0])
        else:
            consensus.append("-")
    return "".join(consensus)


def _propagate_gaps(old_aligned, old_consensus, new_consensus):
    """Insert gaps into old_aligned wherever new_consensus has gaps that old_consensus didn't."""
    result = []
    oi = 0  # index into old_aligned / old_consensus
    for nc in new_consensus:
        if nc == "-" and (oi >= len(old_consensus) or old_consensus[oi] != "-"):
            result.append("-")
        else:
            if oi < len(old_aligned):
                result.append(old_aligned[oi])
            else:
                result.append("-")
            oi += 1
    return "".join(result)


# ---------------------------------------------------------------------------
# Export alignment formats
# ---------------------------------------------------------------------------

def _export_fasta(aligned_pairs):
    """Export list of (name, sequence) as FASTA format string."""
    lines = []
    for name, seq in aligned_pairs:
        lines.append(f">{name}")
        for i in range(0, len(seq), 60):
            lines.append(seq[i:i+60])
    return "\n".join(lines)


def _export_clustal(aligned_pairs):
    """Export list of (name, sequence) as Clustal format string."""
    lines = ["CLUSTAL W (QuantumRes) multiple sequence alignment", ""]
    if not aligned_pairs:
        return "\n".join(lines)
    max_name_len = max(len(n) for n, _ in aligned_pairs)
    block = 60
    aln_len = max(len(s) for _, s in aligned_pairs)
    for start in range(0, aln_len, block):
        for name, seq in aligned_pairs:
            chunk = seq[start:start+block]
            lines.append(f"{name:<{max_name_len+4}s}{chunk}")
        # Conservation line
        cons = []
        for i in range(start, min(start+block, aln_len)):
            chars = set()
            for _, seq in aligned_pairs:
                if i < len(seq) and seq[i] != "-":
                    chars.add(seq[i])
            if len(chars) == 1:
                cons.append("*")
            elif len(chars) == 2:
                cons.append(":")
            else:
                cons.append(" ")
        lines.append(" " * (max_name_len + 4) + "".join(cons))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sequence Logo
# ---------------------------------------------------------------------------

def _compute_logo_data(aligned_seqs, seq_type="DNA"):
    """Compute information content per position for a sequence logo.
    Returns (positions, list of dicts {char: height})."""
    if not aligned_seqs:
        return [], []
    aln_len = max(len(s) for s in aligned_seqs)
    n_seqs = len(aligned_seqs)
    if seq_type in ("DNA", "RNA"):
        alphabet_size = 4
    else:
        alphabet_size = 20
    max_bits = math.log2(alphabet_size)

    logo_data = []
    for i in range(aln_len):
        counts = Counter()
        total = 0
        for s in aligned_seqs:
            if i < len(s) and s[i] != "-":
                counts[s[i].upper()] += 1
                total += 1
        if total == 0:
            logo_data.append({})
            continue
        # Compute entropy
        entropy = 0
        freqs = {}
        for ch, cnt in counts.items():
            f = cnt / total
            freqs[ch] = f
            if f > 0:
                entropy -= f * math.log2(f)
        info_content = max_bits - entropy
        # Height of each letter = freq * info_content
        heights = {}
        for ch, f in freqs.items():
            heights[ch] = f * info_content
        logo_data.append(heights)
    return list(range(aln_len)), logo_data


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class GenomicsWidget(QWidget):
    """Genomics and bioinformatics analysis widget."""

    log_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = None
        self._sequences = []  # list of (name, seq)
        self._init_ui()

    # ---- public API -------------------------------------------------------

    def set_logger(self, fn):
        """Set external logging callback ``fn(message: str)``."""
        self._logger = fn

    def load_file(self, path: str):
        """Load a FASTA file and populate the sequence list."""
        try:
            with open(path, "r") as fh:
                text = fh.read()
            seqs = _parse_fasta(text)
            if not seqs:
                # Try as raw sequence
                raw = text.strip().replace("\n", "").replace(" ", "")
                if raw:
                    seqs = [(os.path.basename(path), raw)]
            for name, seq in seqs:
                self._sequences.append((name, seq))
            self._refresh_sequence_list()
            self._log(f"Loaded {len(seqs)} sequence(s) from {path}")
        except Exception as exc:
            self._log(f"Error loading file: {exc}")

    # ---- logging ----------------------------------------------------------

    def _log(self, msg: str):
        self._output.appendPlainText(msg)
        if self._logger:
            try:
                self._logger(msg)
            except Exception:
                pass

    # ---- UI setup ---------------------------------------------------------

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)

        # Top: tabs
        self._tabs = QTabWidget()
        splitter.addWidget(self._tabs)

        self._build_sequence_tab()
        self._build_analysis_tab()
        self._build_alignment_tab()
        self._build_orf_tab()
        self._build_restriction_tab()
        self._build_codon_tab()
        self._build_blast_tab()
        self._build_phylo_tab()
        self._build_generator_tab()
        self._build_primer_tab()
        self._build_plasmid_tab()
        self._build_msa_tab()
        self._build_logo_tab()

        # Bottom: output log
        self._output = QPlainTextEdit()
        self._output.setReadOnly(True)
        self._output.setMaximumHeight(150)
        self._output.setFont(QFont("Consolas", 9))
        splitter.addWidget(self._output)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

    # ---- Sequence Tab -----------------------------------------------------

    def _build_sequence_tab(self):
        tab = QWidget()
        vl = QVBoxLayout(tab)

        # Toolbar
        tb = QHBoxLayout()
        btn_load = QPushButton("Load FASTA")
        btn_load.clicked.connect(self._on_load_fasta)
        tb.addWidget(btn_load)
        btn_paste = QPushButton("Add from Input")
        btn_paste.clicked.connect(self._on_add_from_input)
        tb.addWidget(btn_paste)
        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self._on_clear_sequences)
        tb.addWidget(btn_clear)
        self._seq_type_combo = QComboBox()
        self._seq_type_combo.addItems(["Auto", "DNA", "RNA", "Protein"])
        tb.addWidget(QLabel("Type:"))
        tb.addWidget(self._seq_type_combo)
        tb.addStretch()
        vl.addLayout(tb)

        # Input area
        self._seq_input = QPlainTextEdit()
        self._seq_input.setPlaceholderText("Paste sequence or FASTA here...")
        self._seq_input.setFont(QFont("Consolas", 10))
        self._seq_input.setMaximumHeight(120)
        vl.addWidget(self._seq_input)

        # Sequence list
        self._seq_table = QTableWidget(0, 4)
        self._seq_table.setHorizontalHeaderLabels(["Name", "Length", "Type", "Sequence (preview)"])
        self._seq_table.horizontalHeader().setStretchLastSection(True)
        self._seq_table.setSelectionBehavior(QTableWidget.SelectRows)
        vl.addWidget(self._seq_table)

        self._tabs.addTab(tab, "Sequences")

    def _on_load_fasta(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load FASTA File", "",
            "FASTA Files (*.fasta *.fa *.fna *.faa *.ffn);;All Files (*)"
        )
        if path:
            self.load_file(path)

    def _on_add_from_input(self):
        text = self._seq_input.toPlainText().strip()
        if not text:
            return
        seqs = _parse_fasta(text)
        if not seqs:
            raw = text.replace("\n", "").replace(" ", "")
            seqs = [("Input", raw)]
        for name, seq in seqs:
            self._sequences.append((name, seq))
        self._refresh_sequence_list()
        self._seq_input.clear()
        self._log(f"Added {len(seqs)} sequence(s) from input")

    def _on_clear_sequences(self):
        self._sequences.clear()
        self._refresh_sequence_list()
        self._log("All sequences cleared")

    def _refresh_sequence_list(self):
        self._seq_table.setRowCount(len(self._sequences))
        for row, (name, seq) in enumerate(self._sequences):
            stype = _detect_seq_type(seq)
            self._seq_table.setItem(row, 0, QTableWidgetItem(name))
            self._seq_table.setItem(row, 1, QTableWidgetItem(str(len(seq))))
            self._seq_table.setItem(row, 2, QTableWidgetItem(stype))
            preview = seq[:80] + ("..." if len(seq) > 80 else "")
            self._seq_table.setItem(row, 3, QTableWidgetItem(preview))
        # Update combo boxes in other tabs
        self._update_seq_combos()

    def _get_selected_seq(self) -> tuple:
        """Return (name, seq) of the first selected or first available sequence."""
        rows = self._seq_table.selectionModel().selectedRows()
        if rows:
            idx = rows[0].row()
        elif self._sequences:
            idx = 0
        else:
            return None, None
        return self._sequences[idx]

    def _update_seq_combos(self):
        """Refresh all sequence-selection combo boxes."""
        names = [n for n, _ in self._sequences]
        for combo in (self._align_seq1_combo, self._align_seq2_combo,
                      self._blast_query_combo, self._phylo_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            combo.blockSignals(False)

    # ---- Analysis Tab -----------------------------------------------------

    def _build_analysis_tab(self):
        tab = QWidget()
        vl = QVBoxLayout(tab)

        tb = QHBoxLayout()
        btn_analyse = QPushButton("Analyse Selected")
        btn_analyse.clicked.connect(self._on_analyse)
        tb.addWidget(btn_analyse)
        btn_rc = QPushButton("Reverse Complement")
        btn_rc.clicked.connect(self._on_reverse_complement)
        tb.addWidget(btn_rc)
        btn_transcribe = QPushButton("Transcribe")
        btn_transcribe.clicked.connect(self._on_transcribe)
        tb.addWidget(btn_transcribe)
        btn_translate = QPushButton("Translate")
        btn_translate.clicked.connect(self._on_translate)
        tb.addWidget(btn_translate)
        tb.addStretch()
        vl.addLayout(tb)

        # Results text
        self._analysis_output = QPlainTextEdit()
        self._analysis_output.setReadOnly(True)
        self._analysis_output.setFont(QFont("Consolas", 10))
        vl.addWidget(self._analysis_output, stretch=1)

        # Nucleotide frequency chart
        self._analysis_fig = Figure(figsize=(5, 2.5), dpi=100)
        style_figure(self._analysis_fig)
        self._analysis_canvas = FigureCanvas(self._analysis_fig)
        vl.addWidget(self._analysis_canvas, stretch=1)

        self._tabs.addTab(tab, "Analysis")

    def _on_analyse(self):
        name, seq = self._get_selected_seq()
        if seq is None:
            self._log("No sequence selected")
            return
        stype = self._effective_type(seq)
        gc = _gc_content(seq) if stype != "Protein" else None
        tm = _melting_temp(seq) if stype != "Protein" else None
        mw = _molecular_weight(seq, stype)

        lines = [
            f"Sequence: {name}",
            f"Type:     {stype}",
            f"Length:   {len(seq)} {'bp' if stype != 'Protein' else 'aa'}",
            f"MW:       {mw:,.1f} Da",
        ]
        if gc is not None:
            lines.append(f"GC%:      {gc * 100:.2f}%")
        if tm is not None:
            lines.append(f"Tm:       {tm:.1f} C")

        # Composition
        counts = Counter(seq.upper())
        comp = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        lines.append(f"Composition: {comp}")
        self._analysis_output.setPlainText("\n".join(lines))
        self._log(f"Analysis complete for {name}")

        # Plot nucleotide / amino-acid frequency
        self._analysis_fig.clear()
        ax = self._analysis_fig.add_subplot(111)
        labels = sorted(counts.keys())
        values = [counts[k] for k in labels]
        colors = []
        color_map = {"A": "#2ecc71", "T": "#e74c3c", "U": "#e74c3c",
                     "G": "#3498db", "C": "#f1c40f"}
        for lb in labels:
            colors.append(color_map.get(lb, "#95a5a6"))
        ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(f"{'Nucleotide' if stype != 'Protein' else 'Amino Acid'} Frequency")
        ax.set_ylabel("Count")
        self._analysis_fig.tight_layout()
        self._analysis_canvas.draw()

    def _on_reverse_complement(self):
        name, seq = self._get_selected_seq()
        if seq is None:
            self._log("No sequence selected")
            return
        stype = self._effective_type(seq)
        if stype == "Protein":
            self._log("Reverse complement not applicable to protein")
            return
        rc = _reverse_complement(seq, stype)
        self._sequences.append((f"{name}_RC", rc))
        self._refresh_sequence_list()
        self._analysis_output.setPlainText(f"Reverse complement of {name}:\n{rc}")
        self._log(f"Reverse complement added as {name}_RC")

    def _on_transcribe(self):
        name, seq = self._get_selected_seq()
        if seq is None:
            return
        stype = self._effective_type(seq)
        if stype != "DNA":
            self._log("Transcription requires DNA sequence")
            return
        rna = _transcribe(seq)
        self._sequences.append((f"{name}_RNA", rna))
        self._refresh_sequence_list()
        self._analysis_output.setPlainText(f"Transcription of {name}:\n{rna}")
        self._log(f"Transcribed {name} -> {name}_RNA")

    def _on_translate(self):
        name, seq = self._get_selected_seq()
        if seq is None:
            return
        stype = self._effective_type(seq)
        if stype == "Protein":
            self._log("Sequence is already protein")
            return
        rna = seq.upper().replace("T", "U")
        prot = _translate(rna)
        self._sequences.append((f"{name}_Protein", prot))
        self._refresh_sequence_list()
        self._analysis_output.setPlainText(f"Translation of {name}:\n{prot}")
        self._log(f"Translated {name} -> {name}_Protein")

    def _effective_type(self, seq: str) -> str:
        forced = self._seq_type_combo.currentText()
        if forced != "Auto":
            return forced
        return _detect_seq_type(seq)

    # ---- Alignment Tab ----------------------------------------------------

    def _build_alignment_tab(self):
        tab = QWidget()
        vl = QVBoxLayout(tab)

        form = QHBoxLayout()
        form.addWidget(QLabel("Seq 1:"))
        self._align_seq1_combo = QComboBox()
        self._align_seq1_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        form.addWidget(self._align_seq1_combo)
        form.addWidget(QLabel("Seq 2:"))
        self._align_seq2_combo = QComboBox()
        self._align_seq2_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        form.addWidget(self._align_seq2_combo)
        vl.addLayout(form)

        params = QHBoxLayout()
        params.addWidget(QLabel("Match:"))
        self._align_match = QSpinBox()
        self._align_match.setRange(-10, 10)
        self._align_match.setValue(2)
        params.addWidget(self._align_match)
        params.addWidget(QLabel("Mismatch:"))
        self._align_mismatch = QSpinBox()
        self._align_mismatch.setRange(-10, 10)
        self._align_mismatch.setValue(-1)
        params.addWidget(self._align_mismatch)
        params.addWidget(QLabel("Gap:"))
        self._align_gap = QSpinBox()
        self._align_gap.setRange(-10, 10)
        self._align_gap.setValue(-2)
        params.addWidget(self._align_gap)
        self._align_method = QComboBox()
        self._align_method.addItems(["Needleman-Wunsch (Global)", "Smith-Waterman (Local)"])
        params.addWidget(self._align_method)
        btn_align = QPushButton("Align")
        btn_align.clicked.connect(self._on_align)
        params.addWidget(btn_align)
        btn_dot = QPushButton("Dot Plot")
        btn_dot.clicked.connect(self._on_dot_plot)
        params.addWidget(btn_dot)
        params.addStretch()
        vl.addLayout(params)

        self._align_output = QPlainTextEdit()
        self._align_output.setReadOnly(True)
        self._align_output.setFont(QFont("Consolas", 10))
        vl.addWidget(self._align_output, stretch=1)

        self._align_fig = Figure(figsize=(5, 4), dpi=100)
        style_figure(self._align_fig)
        self._align_canvas = FigureCanvas(self._align_fig)
        vl.addWidget(self._align_canvas, stretch=1)

        self._tabs.addTab(tab, "Alignment")

    def _on_align(self):
        i1 = self._align_seq1_combo.currentIndex()
        i2 = self._align_seq2_combo.currentIndex()
        if i1 < 0 or i2 < 0 or i1 >= len(self._sequences) or i2 >= len(self._sequences):
            self._log("Select two sequences for alignment")
            return
        _, s1 = self._sequences[i1]
        _, s2 = self._sequences[i2]
        m = self._align_match.value()
        mm = self._align_mismatch.value()
        g = self._align_gap.value()

        # Limit length for performance
        max_len = 5000
        if len(s1) > max_len or len(s2) > max_len:
            self._log(f"Sequences truncated to {max_len} for alignment performance")
            s1 = s1[:max_len]
            s2 = s2[:max_len]

        method = self._align_method.currentText()
        if "Needleman" in method:
            a1, a2, score = _needleman_wunsch(s1, s2, m, mm, g)
        else:
            a1, a2, score = _smith_waterman(s1, s2, m, mm, g)

        # Build match line
        match_line = []
        for c1, c2 in zip(a1, a2):
            if c1 == c2:
                match_line.append("|")
            elif c1 == "-" or c2 == "-":
                match_line.append(" ")
            else:
                match_line.append(".")
        match_str = "".join(match_line)

        # Format in blocks of 60
        block = 60
        lines = [f"Method: {method}", f"Score:  {score}", ""]
        for start in range(0, len(a1), block):
            lines.append(f"Query:   {a1[start:start + block]}")
            lines.append(f"         {match_str[start:start + block]}")
            lines.append(f"Subject: {a2[start:start + block]}")
            lines.append("")

        identity = sum(1 for c1, c2 in zip(a1, a2) if c1 == c2)
        total = len(a1)
        lines.append(f"Identity: {identity}/{total} ({identity / total * 100:.1f}%)")
        gaps = a1.count("-") + a2.count("-")
        lines.append(f"Gaps:     {gaps}/{total * 2} ({gaps / (total * 2) * 100:.1f}%)")
        self._align_output.setPlainText("\n".join(lines))
        self._log(f"Alignment complete, score={score}")

    def _on_dot_plot(self):
        i1 = self._align_seq1_combo.currentIndex()
        i2 = self._align_seq2_combo.currentIndex()
        if i1 < 0 or i2 < 0 or i1 >= len(self._sequences) or i2 >= len(self._sequences):
            self._log("Select two sequences for dot plot")
            return
        n1, s1 = self._sequences[i1]
        n2, s2 = self._sequences[i2]
        max_len = 2000
        s1 = s1[:max_len]
        s2 = s2[:max_len]
        mat = _dot_matrix(s1, s2, window=1)
        self._align_fig.clear()
        ax = self._align_fig.add_subplot(111)
        ax.imshow(mat, cmap="Greys", aspect="auto", interpolation="nearest")
        ax.set_xlabel(n2)
        ax.set_ylabel(n1)
        ax.set_title("Dot Plot")
        self._align_fig.tight_layout()
        self._align_canvas.draw()
        self._log("Dot plot generated")

    # ---- ORF Tab ----------------------------------------------------------

    def _build_orf_tab(self):
        tab = QWidget()
        vl = QVBoxLayout(tab)

        tb = QHBoxLayout()
        tb.addWidget(QLabel("Min ORF length (nt):"))
        self._orf_min = QSpinBox()
        self._orf_min.setRange(9, 3000)
        self._orf_min.setValue(75)
        tb.addWidget(self._orf_min)
        btn = QPushButton("Find ORFs")
        btn.clicked.connect(self._on_find_orfs)
        tb.addWidget(btn)
        tb.addStretch()
        vl.addLayout(tb)

        self._orf_table = QTableWidget(0, 4)
        self._orf_table.setHorizontalHeaderLabels(["Frame", "Start", "End", "Protein"])
        self._orf_table.horizontalHeader().setStretchLastSection(True)
        vl.addWidget(self._orf_table)

        self._tabs.addTab(tab, "ORF Finder")

    def _on_find_orfs(self):
        name, seq = self._get_selected_seq()
        if seq is None:
            self._log("No sequence selected for ORF finding")
            return
        min_len = self._orf_min.value()
        orfs = _find_orfs(seq, min_length=min_len)
        # Also search reverse complement
        stype = self._effective_type(seq)
        if stype != "Protein":
            rc = _reverse_complement(seq, stype)
            rc_orfs = _find_orfs(rc, min_length=min_len)
            for frame, start, end, prot in rc_orfs:
                orfs.append((-frame, start, end, prot))

        self._orf_table.setRowCount(len(orfs))
        for row, (frame, start, end, prot) in enumerate(orfs):
            self._orf_table.setItem(row, 0, QTableWidgetItem(str(frame)))
            self._orf_table.setItem(row, 1, QTableWidgetItem(str(start)))
            self._orf_table.setItem(row, 2, QTableWidgetItem(str(end)))
            preview = prot[:60] + ("..." if len(prot) > 60 else "")
            self._orf_table.setItem(row, 3, QTableWidgetItem(preview))
        self._log(f"Found {len(orfs)} ORFs in {name} (min {min_len} nt)")

    # ---- Restriction Enzymes Tab ------------------------------------------

    def _build_restriction_tab(self):
        tab = QWidget()
        vl = QVBoxLayout(tab)

        tb = QHBoxLayout()
        btn = QPushButton("Find Restriction Sites")
        btn.clicked.connect(self._on_find_restriction)
        tb.addWidget(btn)
        tb.addStretch()
        vl.addLayout(tb)

        self._re_table = QTableWidget(0, 4)
        self._re_table.setHorizontalHeaderLabels(["Enzyme", "Position", "Cut Position", "Recognition"])
        self._re_table.horizontalHeader().setStretchLastSection(True)
        vl.addWidget(self._re_table)

        self._re_fig = Figure(figsize=(5, 2), dpi=100)
        style_figure(self._re_fig)
        self._re_canvas = FigureCanvas(self._re_fig)
        vl.addWidget(self._re_canvas)

        self._tabs.addTab(tab, "Restriction Sites")

    def _on_find_restriction(self):
        name, seq = self._get_selected_seq()
        if seq is None:
            self._log("No sequence selected")
            return
        sites = _find_restriction_sites(seq)
        self._re_table.setRowCount(len(sites))
        for row, (enzyme, pos, cut_pos, recog) in enumerate(sites):
            self._re_table.setItem(row, 0, QTableWidgetItem(enzyme))
            self._re_table.setItem(row, 1, QTableWidgetItem(str(pos)))
            self._re_table.setItem(row, 2, QTableWidgetItem(str(cut_pos)))
            self._re_table.setItem(row, 3, QTableWidgetItem(recog))
        self._log(f"Found {len(sites)} restriction sites in {name}")

        # Map visualization
        self._re_fig.clear()
        ax = self._re_fig.add_subplot(111)
        if sites:
            enzyme_names = list(set(s[0] for s in sites))
            enz_y = {e: i for i, e in enumerate(enzyme_names)}
            for enzyme, pos, cut_pos, _ in sites:
                ax.plot(pos, enz_y[enzyme], "|", color="red", markersize=12)
            ax.set_yticks(range(len(enzyme_names)))
            ax.set_yticklabels(enzyme_names, fontsize=7)
            ax.set_xlabel("Position (bp)")
            ax.set_title(f"Restriction Map: {name}")
            ax.set_xlim(0, len(seq))
        self._re_fig.tight_layout()
        self._re_canvas.draw()

    # ---- Codon Usage Tab --------------------------------------------------

    def _build_codon_tab(self):
        tab = QWidget()
        vl = QVBoxLayout(tab)

        tb = QHBoxLayout()
        btn = QPushButton("Compute Codon Usage")
        btn.clicked.connect(self._on_codon_usage)
        tb.addWidget(btn)
        tb.addStretch()
        vl.addLayout(tb)

        self._codon_table = QTableWidget(0, 4)
        self._codon_table.setHorizontalHeaderLabels(["Codon", "AA", "Count", "Frequency"])
        self._codon_table.horizontalHeader().setStretchLastSection(True)
        vl.addWidget(self._codon_table, stretch=1)

        self._codon_fig = Figure(figsize=(5, 3), dpi=100)
        style_figure(self._codon_fig)
        self._codon_canvas = FigureCanvas(self._codon_fig)
        vl.addWidget(self._codon_canvas, stretch=1)

        self._tabs.addTab(tab, "Codon Usage")

    def _on_codon_usage(self):
        name, seq = self._get_selected_seq()
        if seq is None:
            self._log("No sequence selected")
            return
        stype = self._effective_type(seq)
        if stype == "Protein":
            self._log("Codon usage requires nucleotide sequence")
            return
        rna = seq.upper().replace("T", "U")
        counts = Counter()
        for i in range(0, len(rna) - 2, 3):
            codon = rna[i:i + 3]
            if codon in CODON_TABLE:
                counts[codon] += 1
        total = sum(counts.values()) or 1

        codons_sorted = sorted(CODON_TABLE.keys())
        self._codon_table.setRowCount(len(codons_sorted))
        for row, codon in enumerate(codons_sorted):
            c = counts.get(codon, 0)
            aa = CODON_TABLE[codon]
            freq = c / total
            self._codon_table.setItem(row, 0, QTableWidgetItem(codon))
            self._codon_table.setItem(row, 1, QTableWidgetItem(aa))
            self._codon_table.setItem(row, 2, QTableWidgetItem(str(c)))
            self._codon_table.setItem(row, 3, QTableWidgetItem(f"{freq:.4f}"))

        # Grouped bar chart by amino acid
        self._codon_fig.clear()
        ax = self._codon_fig.add_subplot(111)
        aa_groups = defaultdict(list)
        for codon in codons_sorted:
            aa = CODON_TABLE[codon]
            aa_groups[aa].append((codon, counts.get(codon, 0)))
        x_labels = []
        x_values = []
        x_colors = []
        palette = plt.cm.tab20(np.linspace(0, 1, 20))
        ci = 0
        for aa in sorted(aa_groups.keys()):
            for codon, cnt in aa_groups[aa]:
                x_labels.append(codon)
                x_values.append(cnt)
                x_colors.append(palette[ci % 20])
            ci += 1
        ax.bar(range(len(x_labels)), x_values, color=x_colors, edgecolor="none")
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=90, fontsize=5)
        ax.set_ylabel("Count")
        ax.set_title(f"Codon Usage: {name}")
        self._codon_fig.tight_layout()
        self._codon_canvas.draw()
        self._log(f"Codon usage computed for {name} ({total} codons)")

    # ---- BLAST-like Search Tab --------------------------------------------

    def _build_blast_tab(self):
        tab = QWidget()
        vl = QVBoxLayout(tab)

        form = QHBoxLayout()
        form.addWidget(QLabel("Query:"))
        self._blast_query_combo = QComboBox()
        self._blast_query_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        form.addWidget(self._blast_query_combo)
        form.addWidget(QLabel("Word size:"))
        self._blast_word = QSpinBox()
        self._blast_word.setRange(3, 30)
        self._blast_word.setValue(11)
        form.addWidget(self._blast_word)
        form.addWidget(QLabel("Min score:"))
        self._blast_threshold = QSpinBox()
        self._blast_threshold.setRange(1, 500)
        self._blast_threshold.setValue(20)
        form.addWidget(self._blast_threshold)
        btn = QPushButton("Search")
        btn.clicked.connect(self._on_blast)
        form.addWidget(btn)
        form.addStretch()
        vl.addLayout(form)

        self._blast_table = QTableWidget(0, 5)
        self._blast_table.setHorizontalHeaderLabels(["Subject", "Position", "Score", "Query Seg", "Subject Seg"])
        self._blast_table.horizontalHeader().setStretchLastSection(True)
        vl.addWidget(self._blast_table)

        self._tabs.addTab(tab, "BLAST Search")

    def _on_blast(self):
        qi = self._blast_query_combo.currentIndex()
        if qi < 0 or qi >= len(self._sequences):
            self._log("Select a query sequence")
            return
        query_name, query_seq = self._sequences[qi]
        database = [(n, s) for i, (n, s) in enumerate(self._sequences) if i != qi]
        if not database:
            self._log("Need at least one other sequence as database")
            return
        ws = self._blast_word.value()
        th = self._blast_threshold.value()
        results = _blast_local_search(query_seq, database, word_size=ws, threshold=th)

        self._blast_table.setRowCount(len(results))
        for row, (name, pos, score, qseg, sseg) in enumerate(results):
            self._blast_table.setItem(row, 0, QTableWidgetItem(name))
            self._blast_table.setItem(row, 1, QTableWidgetItem(str(pos)))
            self._blast_table.setItem(row, 2, QTableWidgetItem(str(score)))
            self._blast_table.setItem(row, 3, QTableWidgetItem(qseg[:50]))
            self._blast_table.setItem(row, 4, QTableWidgetItem(sseg[:50]))
        self._log(f"BLAST search: {len(results)} hit(s) for {query_name}")

    # ---- Phylogenetics Tab ------------------------------------------------

    def _build_phylo_tab(self):
        tab = QWidget()
        vl = QVBoxLayout(tab)

        tb = QHBoxLayout()
        self._phylo_combo = QComboBox()
        tb.addWidget(QLabel("Sequences (multi-select all loaded):"))
        tb.addWidget(self._phylo_combo)
        self._phylo_method = QComboBox()
        self._phylo_method.addItems(["UPGMA", "Neighbor-Joining"])
        tb.addWidget(self._phylo_method)
        btn = QPushButton("Build Tree")
        btn.clicked.connect(self._on_build_tree)
        tb.addWidget(btn)
        tb.addStretch()
        vl.addLayout(tb)

        self._phylo_output = QPlainTextEdit()
        self._phylo_output.setReadOnly(True)
        self._phylo_output.setFont(QFont("Consolas", 10))
        vl.addWidget(self._phylo_output, stretch=1)

        self._phylo_fig = Figure(figsize=(5, 4), dpi=100)
        style_figure(self._phylo_fig)
        self._phylo_canvas = FigureCanvas(self._phylo_fig)
        vl.addWidget(self._phylo_canvas, stretch=1)

        self._tabs.addTab(tab, "Phylogenetics")

    def _on_build_tree(self):
        if len(self._sequences) < 3:
            self._log("Need at least 3 sequences for phylogenetic tree")
            return
        names = [n for n, _ in self._sequences]
        seqs = [s for _, s in self._sequences]
        n = len(seqs)

        # Build distance matrix using p-distance
        dm = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = _pairwise_distance(seqs[i], seqs[j])
                dm[i][j] = d
                dm[j][i] = d

        method = self._phylo_method.currentText()
        if method == "UPGMA":
            newick = _upgma(names, dm)
        else:
            newick = _neighbor_joining(names, dm)

        lines = [f"Method: {method}", f"Newick: {newick}", "", "Distance Matrix:"]
        header = "".ljust(15) + "".join(n[:12].ljust(13) for n in names)
        lines.append(header)
        for i, name in enumerate(names):
            row = name[:14].ljust(15) + "".join(f"{dm[i][j]:.4f}".ljust(13) for j in range(n))
            lines.append(row)
        self._phylo_output.setPlainText("\n".join(lines))

        # Simple dendrogram-like visualization
        self._phylo_fig.clear()
        ax = self._phylo_fig.add_subplot(111)
        # Render distance matrix as heatmap
        im = ax.imshow(dm, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        short_names = [nm[:15] for nm in names]
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(short_names, fontsize=7)
        ax.set_title(f"Pairwise Distance Matrix ({method})")
        self._phylo_fig.colorbar(im, ax=ax, shrink=0.8)
        self._phylo_fig.tight_layout()
        self._phylo_canvas.draw()
        self._log(f"Phylogenetic tree built ({method}) for {n} sequences")

    # ---- Random Sequence Generator Tab ------------------------------------

    def _build_generator_tab(self):
        tab = QWidget()
        vl = QVBoxLayout(tab)

        form = QFormLayout()
        self._gen_type = QComboBox()
        self._gen_type.addItems(["DNA", "RNA", "Protein"])
        form.addRow("Sequence type:", self._gen_type)

        self._gen_length = QSpinBox()
        self._gen_length.setRange(10, 100000)
        self._gen_length.setValue(500)
        form.addRow("Length:", self._gen_length)

        self._gen_gc = QDoubleSpinBox()
        self._gen_gc.setRange(0.0, 1.0)
        self._gen_gc.setDecimals(2)
        self._gen_gc.setValue(0.50)
        self._gen_gc.setSingleStep(0.05)
        form.addRow("GC content (DNA/RNA):", self._gen_gc)

        self._gen_name = QLineEdit("Generated_Seq")
        form.addRow("Name:", self._gen_name)

        self._gen_count = QSpinBox()
        self._gen_count.setRange(1, 100)
        self._gen_count.setValue(1)
        form.addRow("Number of sequences:", self._gen_count)

        vl.addLayout(form)

        btn = QPushButton("Generate Sequence(s)")
        btn.clicked.connect(self._on_generate_seq)
        vl.addWidget(btn)
        vl.addStretch()
        self._tabs.addTab(tab, "Seq Generator")

    def _on_generate_seq(self):
        stype = self._gen_type.currentText()
        length = self._gen_length.value()
        gc = self._gen_gc.value()
        name = self._gen_name.text().strip() or "Generated"
        count = self._gen_count.value()

        composition = None
        if stype in ("DNA", "RNA"):
            gc_half = gc / 2
            at_half = (1.0 - gc) / 2
            if stype == "DNA":
                composition = {"A": at_half, "T": at_half, "G": gc_half, "C": gc_half}
            else:
                composition = {"A": at_half, "U": at_half, "G": gc_half, "C": gc_half}

        for i in range(count):
            seq = _generate_random_sequence(stype, length, composition)
            seq_name = f"{name}_{i+1}" if count > 1 else name
            self._sequences.append((seq_name, seq))

        self._refresh_sequence_list()
        self._log(f"Generated {count} {stype} sequence(s), length={length}, GC={gc:.2f}")

    # ---- Primer Design Tab ------------------------------------------------

    def _build_primer_tab(self):
        tab = QWidget()
        vl = QVBoxLayout(tab)

        form = QFormLayout()
        self._primer_start = QSpinBox()
        self._primer_start.setRange(0, 1000000)
        self._primer_start.setValue(100)
        form.addRow("Target region start:", self._primer_start)

        self._primer_end = QSpinBox()
        self._primer_end.setRange(0, 1000000)
        self._primer_end.setValue(500)
        form.addRow("Target region end:", self._primer_end)

        self._primer_len_min = QSpinBox()
        self._primer_len_min.setRange(10, 40)
        self._primer_len_min.setValue(18)
        form.addRow("Min primer length:", self._primer_len_min)

        self._primer_len_max = QSpinBox()
        self._primer_len_max.setRange(10, 40)
        self._primer_len_max.setValue(25)
        form.addRow("Max primer length:", self._primer_len_max)

        self._primer_tm_min = QDoubleSpinBox()
        self._primer_tm_min.setRange(30, 80)
        self._primer_tm_min.setValue(55.0)
        form.addRow("Min Tm:", self._primer_tm_min)

        self._primer_tm_max = QDoubleSpinBox()
        self._primer_tm_max.setRange(30, 80)
        self._primer_tm_max.setValue(65.0)
        form.addRow("Max Tm:", self._primer_tm_max)

        self._primer_gc_min = QDoubleSpinBox()
        self._primer_gc_min.setRange(0.0, 1.0)
        self._primer_gc_min.setDecimals(2)
        self._primer_gc_min.setValue(0.40)
        form.addRow("Min GC:", self._primer_gc_min)

        self._primer_gc_max = QDoubleSpinBox()
        self._primer_gc_max.setRange(0.0, 1.0)
        self._primer_gc_max.setDecimals(2)
        self._primer_gc_max.setValue(0.60)
        form.addRow("Max GC:", self._primer_gc_max)

        vl.addLayout(form)

        btn = QPushButton("Design Primers")
        btn.clicked.connect(self._on_design_primers)
        vl.addWidget(btn)

        self._primer_table = QTableWidget(0, 5)
        self._primer_table.setHorizontalHeaderLabels(["Direction", "Sequence", "Start", "Tm", "GC%"])
        self._primer_table.horizontalHeader().setStretchLastSection(True)
        vl.addWidget(self._primer_table)

        self._tabs.addTab(tab, "Primer Design")

    def _on_design_primers(self):
        name, seq = self._get_selected_seq()
        if seq is None:
            self._log("No sequence selected for primer design")
            return
        start = self._primer_start.value()
        end = self._primer_end.value()
        plen = (self._primer_len_min.value(), self._primer_len_max.value())
        tm = (self._primer_tm_min.value(), self._primer_tm_max.value())
        gc = (self._primer_gc_min.value(), self._primer_gc_max.value())

        primers = _design_primers(seq, start, end, plen, tm, gc)
        self._primer_table.setRowCount(len(primers))
        for row, (direction, pseq, pos, ptm, pgc) in enumerate(primers):
            self._primer_table.setItem(row, 0, QTableWidgetItem(direction))
            self._primer_table.setItem(row, 1, QTableWidgetItem(pseq))
            self._primer_table.setItem(row, 2, QTableWidgetItem(str(pos)))
            self._primer_table.setItem(row, 3, QTableWidgetItem(f"{ptm:.1f}"))
            self._primer_table.setItem(row, 4, QTableWidgetItem(f"{pgc*100:.1f}%"))
        self._log(f"Found {len(primers)} primer candidates for {name}")

    # ---- Plasmid Map Viewer Tab -------------------------------------------

    def _build_plasmid_tab(self):
        tab = QWidget()
        vl = QVBoxLayout(tab)

        tb = QHBoxLayout()
        btn = QPushButton("Draw Plasmid Map")
        btn.clicked.connect(self._on_draw_plasmid)
        tb.addWidget(btn)
        self._plasmid_show_re = QCheckBox("Show restriction sites")
        self._plasmid_show_re.setChecked(True)
        tb.addWidget(self._plasmid_show_re)
        self._plasmid_show_orfs = QCheckBox("Show ORFs")
        self._plasmid_show_orfs.setChecked(True)
        tb.addWidget(self._plasmid_show_orfs)
        tb.addStretch()
        vl.addLayout(tb)

        self._plasmid_fig = Figure(figsize=(6, 6), dpi=100)
        style_figure(self._plasmid_fig)
        self._plasmid_canvas = FigureCanvas(self._plasmid_fig)
        vl.addWidget(self._plasmid_canvas)

        self._tabs.addTab(tab, "Plasmid Map")

    def _on_draw_plasmid(self):
        name, seq = self._get_selected_seq()
        if seq is None:
            self._log("No sequence selected for plasmid map")
            return

        stype = self._effective_type(seq)
        seq_len = len(seq)
        self._plasmid_fig.clear()
        ax = self._plasmid_fig.add_subplot(111, polar=True)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        # Draw the circular backbone
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(theta, np.ones_like(theta), 'k-', linewidth=2.5)

        # Annotate restriction sites
        if self._plasmid_show_re.isChecked() and stype != "Protein":
            sites = _find_restriction_sites(seq)
            for enzyme, pos, cut_pos, recog in sites:
                angle = 2 * np.pi * pos / seq_len
                ax.plot([angle, angle], [0.9, 1.1], 'r-', linewidth=1.5)
                ax.text(angle, 1.15, enzyme, fontsize=5, ha='center', va='bottom',
                        rotation=np.degrees(angle) - 90 if angle < np.pi else np.degrees(angle) + 90)

        # Annotate ORFs
        if self._plasmid_show_orfs.isChecked() and stype != "Protein":
            orfs = _find_orfs(seq, min_length=75)
            colors = plt.cm.Set2(np.linspace(0, 1, max(len(orfs), 1)))
            for idx, (frame, start, end, prot) in enumerate(orfs[:8]):
                a_start = 2 * np.pi * start / seq_len
                a_end = 2 * np.pi * end / seq_len
                if a_end < a_start:
                    a_end += 2 * np.pi
                arc = np.linspace(a_start, a_end, 50)
                radius = 0.85 - 0.05 * (frame - 1)
                ax.plot(arc, np.full_like(arc, radius), linewidth=4,
                        color=colors[idx % len(colors)], solid_capstyle='butt')
                mid_angle = (a_start + a_end) / 2
                ax.text(mid_angle, radius - 0.08, f"ORF{idx+1}({len(prot)}aa)",
                        fontsize=5, ha='center', color=colors[idx % len(colors)])

        # Labels
        for frac, label in [(0, "0"), (0.25, f"{seq_len//4}"), (0.5, f"{seq_len//2}"), (0.75, f"{3*seq_len//4}")]:
            angle = 2 * np.pi * frac
            ax.text(angle, 1.25, label, fontsize=7, ha='center', va='center')

        ax.set_rticks([])
        ax.set_title(f"Plasmid Map: {name} ({seq_len} bp)", pad=20)
        ax.set_ylim(0, 1.4)
        self._plasmid_fig.tight_layout()
        self._plasmid_canvas.draw()
        self._log(f"Plasmid map drawn for {name} ({seq_len} bp)")

    # ---- Multiple Sequence Alignment Tab ----------------------------------

    def _build_msa_tab(self):
        tab = QWidget()
        vl = QVBoxLayout(tab)

        tb = QHBoxLayout()
        btn_msa = QPushButton("Run MSA (all sequences)")
        btn_msa.clicked.connect(self._on_run_msa)
        tb.addWidget(btn_msa)
        btn_export_fasta = QPushButton("Export FASTA")
        btn_export_fasta.clicked.connect(self._on_export_fasta)
        tb.addWidget(btn_export_fasta)
        btn_export_clustal = QPushButton("Export Clustal")
        btn_export_clustal.clicked.connect(self._on_export_clustal)
        tb.addWidget(btn_export_clustal)
        tb.addStretch()
        vl.addLayout(tb)

        self._msa_output = QPlainTextEdit()
        self._msa_output.setReadOnly(True)
        self._msa_output.setFont(QFont("Consolas", 9))
        vl.addWidget(self._msa_output)

        self._tabs.addTab(tab, "MSA")
        self._msa_result = []  # stored alignment

    def _on_run_msa(self):
        if len(self._sequences) < 2:
            self._log("Need at least 2 sequences for MSA")
            return
        names = [n for n, _ in self._sequences]
        seqs = [s for _, s in self._sequences]
        # Truncate for performance
        max_len = 3000
        seqs = [s[:max_len] for s in seqs]
        if any(len(s) > max_len for _, s in self._sequences):
            self._log(f"Sequences truncated to {max_len} for MSA performance")

        self._log(f"Running progressive MSA on {len(seqs)} sequences...")
        QApplication.processEvents()
        self._msa_result = _progressive_msa(seqs, names)

        # Display in Clustal-like format
        output = _export_clustal(self._msa_result)
        self._msa_output.setPlainText(output)
        self._log(f"MSA complete: {len(self._msa_result)} sequences aligned")

    def _on_export_fasta(self):
        if not self._msa_result:
            self._log("Run MSA first before exporting")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export FASTA", "alignment.fasta",
            "FASTA Files (*.fasta *.fa);;All Files (*)"
        )
        if path:
            text = _export_fasta(self._msa_result)
            with open(path, "w") as f:
                f.write(text)
            self._log(f"Alignment exported as FASTA: {path}")

    def _on_export_clustal(self):
        if not self._msa_result:
            self._log("Run MSA first before exporting")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Clustal", "alignment.aln",
            "Clustal Files (*.aln);;All Files (*)"
        )
        if path:
            text = _export_clustal(self._msa_result)
            with open(path, "w") as f:
                f.write(text)
            self._log(f"Alignment exported as Clustal: {path}")

    # ---- Sequence Logo Tab ------------------------------------------------

    def _build_logo_tab(self):
        tab = QWidget()
        vl = QVBoxLayout(tab)

        tb = QHBoxLayout()
        btn = QPushButton("Generate Sequence Logo")
        btn.clicked.connect(self._on_sequence_logo)
        tb.addWidget(btn)
        self._logo_range_start = QSpinBox()
        self._logo_range_start.setRange(0, 100000)
        self._logo_range_start.setValue(0)
        tb.addWidget(QLabel("Start pos:"))
        tb.addWidget(self._logo_range_start)
        self._logo_range_end = QSpinBox()
        self._logo_range_end.setRange(0, 100000)
        self._logo_range_end.setValue(50)
        tb.addWidget(QLabel("End pos:"))
        tb.addWidget(self._logo_range_end)
        tb.addStretch()
        vl.addLayout(tb)

        self._logo_fig = Figure(figsize=(8, 3), dpi=100)
        style_figure(self._logo_fig)
        self._logo_canvas = FigureCanvas(self._logo_fig)
        vl.addWidget(self._logo_canvas)

        self._tabs.addTab(tab, "Seq Logo")

    def _on_sequence_logo(self):
        if not self._msa_result and len(self._sequences) < 2:
            self._log("Need MSA result or at least 2 sequences for sequence logo")
            return

        # Use MSA result if available, otherwise use raw sequences
        if self._msa_result:
            seqs = [s for _, s in self._msa_result]
        else:
            seqs = [s for _, s in self._sequences]

        stype = _detect_seq_type(seqs[0]) if seqs else "DNA"
        start = self._logo_range_start.value()
        end = self._logo_range_end.value()
        end = min(end, max(len(s) for s in seqs))

        # Trim to range
        seqs_trimmed = [s[start:end] for s in seqs]
        positions, logo_data = _compute_logo_data(seqs_trimmed, stype)

        color_map = {
            "A": "#2ecc71", "T": "#e74c3c", "U": "#e74c3c",
            "G": "#f39c12", "C": "#3498db",
        }
        # Protein coloring
        hydrophobic = set("AILMFWV")
        polar = set("STNQYC")
        positive = set("KRH")
        negative = set("DE")

        self._logo_fig.clear()
        ax = self._logo_fig.add_subplot(111)

        for i, heights in enumerate(logo_data):
            sorted_chars = sorted(heights.items(), key=lambda x: x[1])
            y_offset = 0
            for ch, h in sorted_chars:
                if stype in ("DNA", "RNA"):
                    color = color_map.get(ch, "#999999")
                else:
                    if ch in hydrophobic: color = "#2c3e50"
                    elif ch in polar: color = "#27ae60"
                    elif ch in positive: color = "#2980b9"
                    elif ch in negative: color = "#c0392b"
                    else: color = "#999999"
                ax.bar(i + start, h, bottom=y_offset, width=0.9, color=color,
                       edgecolor='none', linewidth=0)
                if h > 0.15:
                    ax.text(i + start, y_offset + h / 2, ch, ha='center', va='center',
                            fontsize=max(5, min(10, int(h * 12))), fontweight='bold', color='white')
                y_offset += h

        ax.set_xlabel("Position")
        ax.set_ylabel("Information (bits)")
        max_bits = 2.0 if stype in ("DNA", "RNA") else math.log2(20)
        ax.set_ylim(0, max_bits)
        ax.set_title(f"Sequence Logo ({len(seqs)} sequences, positions {start}-{end})")
        ax.set_xlim(start - 0.5, end - 0.5)
        self._logo_fig.tight_layout()
        self._logo_canvas.draw()
        self._log(f"Sequence logo generated for positions {start}-{end}")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = GenomicsWidget()
    w.setWindowTitle("Genomics Widget - Standalone Test")
    w.resize(1100, 800)
    w.show()

    # Add a sample sequence for quick testing
    sample_dna = (
        "ATGGCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCC"
        "GAATTCGGATCCAAGCTTGCGGCCGCCTCGAGGTCGACCTGCAGCCCGGG"
        "ATGAAAGCTATCGGCATCGACCTGGGCACTGTGTCCGCTGATAAAGCTGAC"
        "TAAATGCCTGCAGTTAGCAGGCTTAACCTTTAGGAGCAATCTTGCCAGTTT"
    )
    w._sequences.append(("Sample_DNA", sample_dna))
    w._sequences.append(("Sample_DNA_2", sample_dna[:100] + "NNNN" + sample_dna[104:]))
    w._sequences.append(("Short_DNA", sample_dna[:60]))
    w._refresh_sequence_list()

    sys.exit(app.exec_())
