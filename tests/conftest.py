import sys
from pathlib import Path

# Los modulos viven en src/ sin empaquetar; se agrega al path para importarlos.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
