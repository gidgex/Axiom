"""
Axiom Scientific Suite — Universal File Format Registry
Maps file extensions to modules and provides import/export utilities.
"""
import os
import csv
import json

# Comprehensive file type routing: extension -> (module_tab_name, description)
IMPORT_FORMATS = {
    # Data files
    '.csv': ('Data Analysis', 'Comma-separated values'),
    '.tsv': ('Data Analysis', 'Tab-separated values'),
    '.dat': ('Data Analysis', 'Generic data file'),
    '.txt': ('Data Analysis', 'Text data file'),
    '.xlsx': ('Data Analysis', 'Microsoft Excel'),
    '.xls': ('Data Analysis', 'Microsoft Excel (legacy)'),
    '.json': ('Data Analysis', 'JSON data'),
    '.hdf5': ('Data Analysis', 'HDF5 scientific data'),
    '.h5': ('Data Analysis', 'HDF5 scientific data'),
    '.nc': ('Data Analysis', 'NetCDF scientific data'),
    '.mat': ('Data Analysis', 'MATLAB data file'),
    '.npy': ('Data Analysis', 'NumPy array'),
    '.npz': ('Data Analysis', 'NumPy compressed arrays'),
    '.parquet': ('Data Analysis', 'Apache Parquet'),
    '.feather': ('Data Analysis', 'Arrow Feather'),

    # Python / Code
    '.py': ('Python Console', 'Python script'),
    '.r': ('Python Console', 'R script'),
    '.m': ('Python Console', 'MATLAB/Octave script'),
    '.jl': ('Python Console', 'Julia script'),

    # Images
    '.png': ('Image Processor', 'PNG image'),
    '.jpg': ('Image Processor', 'JPEG image'),
    '.jpeg': ('Image Processor', 'JPEG image'),
    '.tif': ('Image Processor', 'TIFF image'),
    '.tiff': ('Image Processor', 'TIFF image'),
    '.bmp': ('Image Processor', 'Bitmap image'),
    '.gif': ('Image Processor', 'GIF image'),
    '.svg': ('Image Processor', 'SVG vector image'),
    '.webp': ('Image Processor', 'WebP image'),

    # Crystal / Molecular structures
    '.cif': ('Crystal Viewer', 'Crystallographic Information File'),
    '.xyz': ('Molecule Viewer', 'XYZ coordinates'),
    '.pdb': ('Molecule Viewer', 'Protein Data Bank'),
    '.mol': ('Molecule Viewer', 'MDL Molfile'),
    '.mol2': ('Molecule Viewer', 'Tripos Mol2'),
    '.sdf': ('Molecule Viewer', 'Structure Data File'),
    '.poscar': ('Crystal Viewer', 'VASP POSCAR'),
    '.vasp': ('Crystal Viewer', 'VASP structure'),

    # CAD files
    '.dxf': ('2D CAD', 'AutoCAD Drawing Exchange'),
    '.dwg': ('2D CAD', 'AutoCAD Drawing'),
    '.stl': ('3D CAD', 'Stereolithography mesh'),
    '.obj': ('3D CAD', 'Wavefront OBJ mesh'),
    '.step': ('3D CAD', 'STEP CAD model'),
    '.stp': ('3D CAD', 'STEP CAD model'),
    '.iges': ('3D CAD', 'IGES CAD model'),
    '.igs': ('3D CAD', 'IGES CAD model'),
    '.3mf': ('3D CAD', '3D Manufacturing Format'),
    '.ply': ('3D CAD', 'Polygon File Format'),
    '.off': ('3D CAD', 'Object File Format'),

    # IC Layout
    '.gds': ('IC Layout', 'GDSII layout'),
    '.gdsii': ('IC Layout', 'GDSII layout'),
    '.oasis': ('IC Layout', 'OASIS layout'),
    '.lef': ('IC Layout', 'Library Exchange Format'),
    '.def': ('IC Layout', 'Design Exchange Format'),

    # Documents
    '.tex': ('LaTeX Editor', 'LaTeX source'),
    '.bib': ('LaTeX Editor', 'BibTeX bibliography'),
    '.pdf': ('PDF Tools', 'PDF document'),
    '.ps': ('PDF Tools', 'PostScript'),
    '.eps': ('PDF Tools', 'Encapsulated PostScript'),
    '.docx': ('PDF Tools', 'Microsoft Word'),
    '.doc': ('PDF Tools', 'Microsoft Word (legacy)'),
    '.odt': ('PDF Tools', 'OpenDocument Text'),
    '.rtf': ('PDF Tools', 'Rich Text Format'),

    # Genomics
    '.fasta': ('Genomics', 'FASTA sequence'),
    '.fa': ('Genomics', 'FASTA sequence'),
    '.fastq': ('Genomics', 'FASTQ sequence + quality'),
    '.fq': ('Genomics', 'FASTQ sequence + quality'),
    '.gb': ('Genomics', 'GenBank record'),
    '.gbk': ('Genomics', 'GenBank record'),
    '.sam': ('Genomics', 'Sequence Alignment Map'),
    '.bam': ('Genomics', 'Binary Alignment Map'),
    '.vcf': ('Genomics', 'Variant Call Format'),
    '.bed': ('Genomics', 'BED annotation'),
    '.gff': ('Genomics', 'General Feature Format'),

    # GIS
    '.shp': ('GIS / Mapping', 'Shapefile'),
    '.kml': ('GIS / Mapping', 'Keyhole Markup Language'),
    '.kmz': ('GIS / Mapping', 'KML compressed'),
    '.gpx': ('GIS / Mapping', 'GPS Exchange Format'),
    '.geojson': ('GIS / Mapping', 'GeoJSON'),

    # Notebooks
    '.ipynb': ('Notebook', 'Jupyter Notebook'),
    '.qnb': ('Notebook', 'Axiom Notebook'),

    # Signal / Audio
    '.wav': ('Signal Processing', 'WAV audio'),
    '.mp3': ('Signal Processing', 'MP3 audio'),
    '.flac': ('Signal Processing', 'FLAC audio'),

    # Fractal
    '.axiom': ('Dashboard', 'Axiom Project File'),

    # Circuit
    '.cir': ('Circuit Sim', 'SPICE netlist'),
    '.spice': ('Circuit Sim', 'SPICE netlist'),
    '.net': ('Circuit Sim', 'Netlist'),
}

# Export format groups by module category
EXPORT_FORMATS = {
    'data': {
        'CSV (.csv)': '.csv',
        'TSV (.tsv)': '.tsv',
        'Excel (.xlsx)': '.xlsx',
        'JSON (.json)': '.json',
        'NumPy (.npy)': '.npy',
        'Parquet (.parquet)': '.parquet',
        'HTML Table (.html)': '.html',
        'LaTeX Table (.tex)': '.tex',
        'Markdown (.md)': '.md',
        'SQL (.sql)': '.sql',
    },
    'plot': {
        'PNG Image (.png)': '.png',
        'SVG Vector (.svg)': '.svg',
        'PDF (.pdf)': '.pdf',
        'EPS (.eps)': '.eps',
        'TIFF (.tiff)': '.tiff',
        'JPEG (.jpg)': '.jpg',
    },
    'document': {
        'PDF (.pdf)': '.pdf',
        'LaTeX (.tex)': '.tex',
        'HTML (.html)': '.html',
        'Plain Text (.txt)': '.txt',
        'Markdown (.md)': '.md',
        'Rich Text (.rtf)': '.rtf',
    },
    'cad2d': {
        'DXF (.dxf)': '.dxf',
        'SVG (.svg)': '.svg',
        'PDF (.pdf)': '.pdf',
        'PNG (.png)': '.png',
    },
    'cad3d': {
        'STL ASCII (.stl)': '.stl',
        'OBJ (.obj)': '.obj',
        'PLY (.ply)': '.ply',
        'PNG Render (.png)': '.png',
        'SVG (.svg)': '.svg',
    },
    'structure': {
        'XYZ (.xyz)': '.xyz',
        'PDB (.pdb)': '.pdb',
        'CIF (.cif)': '.cif',
        'POSCAR (.vasp)': '.vasp',
        'MOL2 (.mol2)': '.mol2',
    },
    'layout': {
        'GDSII (.gds)': '.gds',
        'JSON (.json)': '.json',
        'PNG (.png)': '.png',
        'SVG (.svg)': '.svg',
    },
    'image': {
        'PNG (.png)': '.png',
        'JPEG (.jpg)': '.jpg',
        'TIFF (.tiff)': '.tiff',
        'BMP (.bmp)': '.bmp',
        'WebP (.webp)': '.webp',
        'PDF (.pdf)': '.pdf',
    },
    'sequence': {
        'FASTA (.fasta)': '.fasta',
        'GenBank (.gb)': '.gb',
        'Plain Text (.txt)': '.txt',
    },
    'circuit': {
        'SPICE Netlist (.cir)': '.cir',
        'CSV Results (.csv)': '.csv',
        'PNG Plot (.png)': '.png',
    },
}


def get_import_filter():
    """Build a QFileDialog filter string for all supported import formats."""
    all_exts = ' '.join(f'*{ext}' for ext in sorted(IMPORT_FORMATS.keys()))
    filters = [f'All Supported Files ({all_exts})']

    categories = {}
    for ext, (mod, desc) in IMPORT_FORMATS.items():
        cat = mod
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((ext, desc))

    for cat in sorted(categories.keys()):
        exts = ' '.join(f'*{e}' for e, _ in categories[cat])
        filters.append(f'{cat} Files ({exts})')

    filters.append('All Files (*)')
    return ';;'.join(filters)


def get_export_filter(category):
    """Get QFileDialog filter for a specific export category."""
    fmts = EXPORT_FORMATS.get(category, EXPORT_FORMATS.get('plot', {}))
    parts = [f'{name}' for name in fmts.keys()]
    parts.append('All Files (*)')
    return ';;'.join(parts)


def route_file(path):
    """Given a file path, return the module tab name that should handle it."""
    ext = os.path.splitext(path)[1].lower()
    if ext in IMPORT_FORMATS:
        return IMPORT_FORMATS[ext][0]
    return 'Python Console'


def write_data_csv(data, path):
    """Write a list-of-lists or numpy array to CSV."""
    import numpy as np
    if isinstance(data, np.ndarray):
        np.savetxt(path, data, delimiter=',')
    elif hasattr(data, 'to_csv'):
        data.to_csv(path, index=False)
    else:
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in data:
                writer.writerow(row)


def write_data_json(data, path):
    """Write data to JSON."""
    import numpy as np

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return super().default(obj)

    with open(path, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


def write_html_table(df, path, title="Axiom Data Export"):
    """Write a pandas DataFrame to a styled HTML file."""
    html = f"""<!DOCTYPE html>
<html><head><title>{title}</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; background: #1e1e1e; color: #ddd; padding: 20px; }}
h1 {{ color: #4a90d9; }}
table {{ border-collapse: collapse; width: 100%; }}
th {{ background: #2a2a2a; color: #4a90d9; padding: 8px 12px; border: 1px solid #444; text-align: left; }}
td {{ padding: 6px 12px; border: 1px solid #333; }}
tr:nth-child(even) {{ background: #252525; }}
tr:hover {{ background: #2a3a4a; }}
</style></head><body>
<h1>{title}</h1>
{df.to_html(index=False, classes='axiom-table')}
<p style="color:#666; margin-top:20px;">Exported from Axiom Scientific Suite</p>
</body></html>"""
    with open(path, 'w') as f:
        f.write(html)
