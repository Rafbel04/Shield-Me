"""
Convert binary STL files to OpenSCAD polyhedron() code.

This module reads STL geometry and generates standalone OpenSCAD polyhedron
definitions, enabling fully portable SCAD files without external STL dependencies.
"""

import struct
import numpy as np
from pathlib import Path


def stl_to_polyhedron(stl_path, center=True, indent="  "):
    """Convert binary STL file to OpenSCAD polyhedron code.

    Args:
        stl_path: Path to binary STL file
        center: If True, center the geometry at origin (default True)
        indent: Indentation string for formatting (default "  ")

    Returns:
        str: OpenSCAD polyhedron() code with proper formatting
    """
    vertices, faces = parse_binary_stl(stl_path)

    if center:
        vertices = center_vertices(vertices)

    # Build OpenSCAD code
    lines = []
    lines.append(f"{indent}polyhedron(")

    # Points array
    lines.append(f"{indent}  points = [")
    for i, v in enumerate(vertices):
        comma = "," if i < len(vertices) - 1 else ""
        lines.append(f"{indent}    [{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}]{comma}")
    lines.append(f"{indent}  ],")

    # Faces array
    lines.append(f"{indent}  faces = [")
    for i, f in enumerate(faces):
        comma = "," if i < len(faces) - 1 else ""
        lines.append(f"{indent}    [{f[0]}, {f[1]}, {f[2]}]{comma}")
    lines.append(f"{indent}  ]")

    lines.append(f"{indent});")

    return "\n".join(lines)


def parse_binary_stl(stl_path):
    """Parse binary STL file and extract unique vertices and face indices.

    Args:
        stl_path: Path to binary STL file

    Returns:
        tuple: (vertices, faces)
            - vertices: List of [x, y, z] coordinates
            - faces: List of [v1_idx, v2_idx, v3_idx] indices
    """
    with open(stl_path, 'rb') as f:
        # Skip 80-byte header
        f.read(80)

        # Read number of triangles (uint32)
        num_triangles = struct.unpack('<I', f.read(4))[0]

        # Read all triangles
        raw_vertices = []
        for _ in range(num_triangles):
            # Skip normal vector (3 floats)
            f.read(12)

            # Read 3 vertices (9 floats total)
            v1 = struct.unpack('<fff', f.read(12))
            v2 = struct.unpack('<fff', f.read(12))
            v3 = struct.unpack('<fff', f.read(12))

            raw_vertices.append([v1, v2, v3])

            # Skip attribute byte count (uint16)
            f.read(2)

    # Deduplicate vertices and build face index list
    vertices, faces = deduplicate_vertices(raw_vertices)

    return vertices, faces


def deduplicate_vertices(raw_vertices, epsilon=1e-6):
    """Deduplicate vertices and build face index list.

    Args:
        raw_vertices: List of triangles, each containing 3 vertices
        epsilon: Tolerance for considering vertices identical

    Returns:
        tuple: (unique_vertices, faces)
    """
    unique_vertices = []
    vertex_to_index = {}
    faces = []

    for triangle in raw_vertices:
        face_indices = []
        for vertex in triangle:
            # Round to handle floating point comparison
            vertex_tuple = tuple(round(coord / epsilon) * epsilon for coord in vertex)

            if vertex_tuple not in vertex_to_index:
                vertex_to_index[vertex_tuple] = len(unique_vertices)
                unique_vertices.append(list(vertex))

            face_indices.append(vertex_to_index[vertex_tuple])

        faces.append(face_indices)

    return unique_vertices, faces


def center_vertices(vertices):
    """Align vertices so bottom-left-lowest corner is at origin (0,0,0).

    Args:
        vertices: List of [x, y, z] coordinates

    Returns:
        List of aligned [x, y, z] coordinates with min corner at origin
    """
    vertices_array = np.array(vertices)

    # Find minimum coordinates (bottom-left-lowest corner)
    min_coords = vertices_array.min(axis=0)

    # Translate so minimum corner is at origin
    aligned = vertices_array - min_coords

    return aligned.tolist()


def main():
    """Convert Shield-Me blank shield STL to Python template."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python stl_to_polyhedron.py <stl_file>")
        print("\nExample:")
        print("  python stl_to_polyhedron.py out/Blank_IO_Shield-No_Grill.stl")
        sys.exit(1)

    stl_path = sys.argv[1]
    if not Path(stl_path).exists():
        print(f"Error: File not found: {stl_path}")
        sys.exit(1)

    print(f"Converting {stl_path} to OpenSCAD polyhedron...")
    polyhedron_code = stl_to_polyhedron(stl_path, center=True, indent="  ")

    # Generate Python template file
    output_path = Path(__file__).parent / "shield_polyhedron_template.py"

    with open(output_path, 'w') as f:
        f.write('"""\n')
        f.write('Auto-generated shield geometry template.\n')
        f.write('This is appended to SCAD files for an AIO openSCAD solution.\n')
        f.write('DO NOT EDIT - regenerate with: python stl_to_polyhedron.py <stl_file>\n')
        f.write('"""\n\n')
        f.write('SHIELD_POLYHEDRON = """\\\n')
        f.write(polyhedron_code)
        f.write('\n"""\n')

    print(f"Generated: {output_path}")
    print(f"Shield geometry ready for use in opencv_handler.py")


if __name__ == "__main__":
    main()
