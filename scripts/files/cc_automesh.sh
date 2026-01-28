#!/bin/bash

# ==============================================================================
#  AUTOMATIC MESHING SCRIPT USING CLOUDCOMPARE (CLI)
#  Usage: ./cc_automesh.sh <input_cloud.xyz> <output_folder>
# ==============================================================================

# --- 1. CONFIGURATION PARAMETERS (Adjust these values as needed) ---

# Path to the executable (Snap version detected in your system)
CC_BIN="/snap/bin/cloudcompare.CloudCompare"

# Step A: SOR Filter (Statistical Outlier Removal) - Noise Cleaning
SOR_K=50               # Number of neighbors to analyze
SOR_SIGMA=1.0          # Standard Deviation (Lower value = more strict cleaning)

# Step B: Subsampling (Voxel Grid) - Uniformize density
VOXEL_SIZE=0.1         # Voxel size in meters (e.g., 0.1 = 10cm)

# Step C: MLS Smoothing (2.5D Quadric Fit) - OMITIDO EN CLI POR INESTABILIDAD
MLS_RADIUS=0.4         # Smoothing Radius (Parameter kept for comments)

# Step D: Triangulation (Delaunay 2.5D) - Generate Mesh
DELAUNAY_MAX_EDGE=1.5  # Max edge length of the triangle (prevents connecting large gaps)

# ==============================================================================

# --- 2. ARGUMENT AND DIRECTORY MANAGEMENT ---

INPUT_FILE="$1"
OUTPUT_DIR="$2"

# Basic validation
if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "❌ Error: Missing arguments."
    echo "Correct usage: ./cc_automesh.sh <input_file.xyz> <output_folder>"
    echo "Example: ./cc_automesh.sh my_data/cloud.xyz my_results/"
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: The input file '$INPUT_FILE' does not exist."
    exit 1
fi

# 1. CORRECCIÓN DEL PATH (Elimina la barra final si existe)
OUTPUT_DIR_CLEANED="${OUTPUT_DIR%/}"

# Create output directory if it does not exist
if [ ! -d "$OUTPUT_DIR_CLEANED" ]; then
    echo "g Creating output directory: $OUTPUT_DIR_CLEANED"
    mkdir -p "$OUTPUT_DIR_CLEANED"
fi

# Prepare output filename (Añade una sola barra entre el directorio limpio y el archivo)
FILENAME=$(basename -- "$INPUT_FILE")
FILENAME_NO_EXT="${FILENAME%.*}"
OUTPUT_PATH="${OUTPUT_DIR_CLEANED}/${FILENAME_NO_EXT}_mesh_processed.ply" # <-- PATH CORREGIDO

# --- 3. EXECUTION OF THE PROCESS ---

echo "========================================="
echo "Starting CloudCompare Processing"
echo "-----------------------------------------"
echo "📄 Input:  $INPUT_FILE"
echo "💾 Output: $OUTPUT_PATH"
echo "⚙️  Config: Voxel=$VOXEL_SIZE | SOR_K=$SOR_K"
echo "========================================="

# Execute the command in silent mode
$CC_BIN \
    -SILENT \
    -AUTO_SAVE OFF \
    -O -GLOBAL_SHIFT AUTO "$INPUT_FILE" \
    -C_EXPORT_FMT PLY \
    \
    -SOR $SOR_K $SOR_SIGMA \
    -SS SPATIAL $VOXEL_SIZE \
    \
    -DELAUNAY -MAX_EDGE_LENGTH $DELAUNAY_MAX_EDGE \
    \
    -SAVE_MESHES FILE "$OUTPUT_PATH"

# --- 4. RESULT ---

if [ $? -eq 0 ]; then
    echo "✅ Success! Mesh generated correctly."
    echo "📍 File saved at: $OUTPUT_PATH"
else
    echo "❌ An error occurred during CloudCompare execution. (Check CloudCompare logs for specific module error.)"
fi